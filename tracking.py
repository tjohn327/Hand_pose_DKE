import cv2
import tensorflow as tf
import multiprocessing
from multiprocessing import  Queue,Pool
import time
import datetime
import os
import keras
import detector
from detector import WebcamVideoStream
from classification import classify

frame_processed = 0
score_thresh = 0.18

# Create a worker thread that loads graph and
# does detection on images in an input queue
def worker(input_queue,output_queue,cap_params,frame_processed):
    sess = tf.Session()
    try:
        model,classification_graph,session = classify()

if __name__ == '__main__':
    vid_src = 0
    num_hands = 1
    fps = 1
    width = 300
    height = 200
    display = 1
    num_workers = 4
    queue_size = 5

    input_queue = Queue(maxsize=queue_size)
    output_queue = Queue(maxsize=queue_size)

    video_capture = WebcamVideoStream(src=vid_src,width = width,height = height).start()

    cap_param = {}
    frame_processed = 0
    cap_param['image_width'],cap_param['image_height'] = video_capture.size()
    cap_param['score_threshold'] = score_thresh

    pool = Pool(num_workers,worker,(input_queue,output_queue,cap_param,frame_processed))
