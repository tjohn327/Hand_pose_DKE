import cv2
import os
import numpy as np
import keras
from sklearn.utils import shuffle
from PIL import Image
from keras.preprocessing.image import img_to_array, array_to_img


def read_poses(poses):
    count_image = 0
    count_classes = 0
    poses_directory = os.listdir('F:/Hand_pose/Poses/')

    if (poses[0] == 'all'):
        poses = poses_directory.copy()

    for pose in poses_directory:
        if pose in poses:
            subdirs = os.listdir("F:/Hand_pose/Poses/" + pose + '/')
            count_classes += 1
            for subdir in subdirs:
                files = os.listdir("F:/Hand_pose/Poses/" + pose + '/' + subdir + '/')
                for file in files:
                    if (file.endswith('.png')):
                        path = 'F:/Hand_pose/Poses/' + pose + '/' + subdir + '/' + file
                        count_image += 1
    print("Classes = " + str(count_classes))
    print("Images = " + str(count_image))
    x = np.empty(shape=(count_image, 28, 28, 1))
    y = np.empty(count_image)

    count_image = 0
    count_classes = 0
    for pose in poses_directory:
        print("hello")
        if pose in poses:
            print(" Current Pose: " + pose)
            subdirs = os.listdir("F:/Hand_pose/Poses/" + pose + '/')
            for subdir in subdirs:
                files = os.listdir("F:/Hand_pose/Poses/" + pose + '/' + subdir + '/')
                print("Current example :" + subdir)
                for file in files:
                    if file.endswith('.png'):
                        path = 'F:/Hand_pose/Poses/' + pose + '/' + subdir + '/' + file
                        # Reading the image and normalizing
                        image = cv2.imread(path)
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        image = image.astype(dtype="float32")
                        image = np.reshape(image, (28, 28, 1))
                        x[count_image][:][:][:] = image
                        y[count_image] = count_classes
                        count_image += 1
            count_classes += 1
    x = x / 255

    return x, y


def load_poses(poses=['all']):
    x, y = read_poses(poses)
    x, y = shuffle(x, y, random_state=0)

    x_train, y_train, x_test, y_test = split_poses(x, y)

    return x_train, y_train, x_test, y_test


def split_poses(x, y, split=0.8):
    index = int(split * x.shape[0])
    x_train = x[:index][:][:][:]
    x_test = x[index:][:][:][:]
    y_train = y[:index]
    y_test = y[index:]
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_poses()

    print(x_train, x_test,y_train,y_test)


