import utils
from utils import detector_utils as detector_utils
from utils import pose_classification_utils as classifier
import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import Queue, Pool
import time
from utils.detector_utils import WebcamVideoStream
import datetime
import argparse
import numpy as np
import os
import keyboard
from collections import deque

os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
import gui

frame_processed = 0
score_thresh = 0.18
frame_start = 0
frame_mid = 3
frame_end = 5


# Create a worker thread that loads graph and
# does detection on images in an input queue and puts it on an output queue
#
def worker(input_q, output_q, cap_params, frame_count, poses):
    centroid = None
    predicted_label = ""
    print(">> loading frozen model for worker")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    print(">> loading keras model for worker")
    try:
        model, classification_graph, session = classifier.load_KerasGraph(
            "F:/Github/Hand_pose_DKE/cnn/models/hand_poses_wGarbage_10.h5")
    except Exception as e:
        print(e)
    centroid_list = deque(maxlen=5)
    direction = ""
    (dX, dY) = (0, 0)
    while True:
        # print("> ===== in worker loop, frame ", frame_count)
        frame = input_q.get()
        if (frame is not None):
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
            boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

            # print(boxes[0])

            # get region of interest
            res = detector_utils.get_box_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
                                               scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)

            # get boundary box
            detector_utils.draw_box_on_image(cap_params['num_hands_detect'], cap_params["score_thresh"],
                                             scores, boxes, cap_params['im_width'], cap_params['im_height'], frame)

            # classify hand pose
            if res is not None and frame_count == 0:
                class_res = classifier.classify(model, classification_graph, session, res)
                class_pred = class_res.argmax(axis=-1)
                predicted_label = poses[int(class_pred)]
                #print(predicted_label)

            if predicted_label == "Start" and frame_count <= frame_end:
                centroid = detector_utils.get_centroid(cap_params['num_hands_detect'], cap_params["score_thresh"],
                                                   scores, boxes, cap_params['im_width'], cap_params['im_height'],
                                                   frame)

            if centroid is not None:
                #if frame_count in range(frame_start,frame_end-1):
                centroid_list.appendleft(centroid)

                for i in np.arange(1, len(centroid_list)):
                    if centroid_list[i-1] is None or centroid_list[i] is None:
                        continue
                    if frame_count >= frame_end and centroid_list[-5] != None and i == 1:
                        dX = centroid_list[-5][0] - centroid_list[i][0]
                        dY = centroid_list[-5][1] - centroid_list[i][1]
                        (dirX,dirY) = ("","")

                        if np.abs(dX) > 20:
                            dirX = "Right" if np.sign(dX) == 1 else "Left"

                        if np.abs(dY) > 20:
                            dirY = "DOWN" if np.sign(dY) == 1 else "UP"

                        if dirX != "" and dirY != "":
                            direction = "{}-{}".format(dirY, dirX)

                        else:
                            direction = dirX if dirX != "" else dirY

                    thickness = int(np.sqrt(frame_end / float(i + 1)) * 2.5)
                    cv2.line(frame, centroid_list[i - 1], centroid_list[i], (0, 0, 255), thickness)

                cv2.putText(frame, direction, (20, 50), cv2.FONT_HERSHEY_SIMPLEX,
                            0.65, (77, 255, 9), 1)
                cv2.putText(frame, "dx: {}, dy: {}".format(dX, dY),
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 0, 255), 1)
                if direction == "Left":
                    keyboard.press_and_release('left')
                elif direction == "Right":
                    keyboard.press_and_release('right')

                frame_count += 1
                if frame_count > frame_end:
                    frame_count = 0
                    centroid_list.clear()
                    direction = ""
                    flag = 1


        #print(frame_count)

        output_q.put(frame)  # print(frame_processed)
# else:
#     output_q.put(frame)
    sess.close()

if __name__ == '__main__':

    vid_src = 1
    num_hands = 1
    fps = 1
    width = 300
    height = 200
    display = 1
    num_workers = 4
    queue_size = 5

    input_q = Queue(maxsize=queue_size)
    output_q = Queue(maxsize=queue_size)
    # cropped_output_q    = Queue(maxsize=args.queue_size)
    # inferences_q        = Queue(maxsize=args.queue_size)

    video_capture = WebcamVideoStream(
        src=vid_src, width=width, height=height).start()

    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    print(cap_params['im_width'], cap_params['im_height'])
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = num_hands

    # Count number of files to increment new example directory
    poses = []
    _file = open("poses.txt", "r")
    lines = _file.readlines()
    for line in lines:
        line = line.strip()
        if (line != ""):
            print(line)
            poses.append(line)

    # spin up workers to paralleize detection.
    pool = Pool(num_workers, worker,
                (input_q, output_q, cap_params, frame_processed, poses))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    cv2.namedWindow('Handpose', cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = video_capture.read()
            frame = cv2.flip(frame, 1)
            index += 1

            input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            output_frame = output_q.get()

            inferences = None

            elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
            num_frames += 1
            fps = num_frames / elapsed_time

            if (output_frame is not None):
                output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                if (display > 0):
                    # if (fps > 0):
                    #     detector_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                    #                                      output_frame)
                    cv2.imshow('Handpose', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    if (num_frames == 400):
                        num_frames = 0
                        start_time = datetime.datetime.now()
                    else:
                        print("frames processed: ", index, "elapsed time: ",
                              elapsed_time, "fps: ", str(int(fps)))
            else:
                print("video end")
                break
    except KeyboardInterrupt:
        pass
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()
