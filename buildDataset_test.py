import cv2
import os
import numpy as np
from sklearn.utils import shuffle
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.preprocessing import image as image_util
from keras.utils import to_categorical
from PIL import Image


def read_poses(poses):
    poses_directory = os.listdir('F:/Hand_pose/Poses/')
    if (poses[0] == 'all'):
        poses = poses_directory.copy()

    x_data = []
    y_data = []

    lookup = dict()
    reverselookup = dict()
    count = 0
    for j in os.listdir('F:/Hand_pose/Poses/'):
        if not j.startswith('.'):
            lookup[j] = count
            reverselookup[count] = j
            count = count + 1

    datacount = 0
    for pose in poses_directory:
        count = 0
        if pose in poses:
            print(" Current Pose: " + pose)
            subdirs = os.listdir("F:/Hand_pose/Poses/" + pose + '/')
            for subdir in subdirs:
                files = os.listdir("F:/Hand_pose/Poses/" + pose + '/' + subdir + '/')
                print("Current example :" + subdir)
                for file in files:
                    if (file.endswith('.png')):
                        path = 'F:/Hand_pose/Poses/' + pose + '/' + subdir + '/' + file
                        # Reading the image and normalizing
                        image = Image.open(path).convert('RGB')
                        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        #image = image.astype(dtype="float64")
                        image = image.resize((224,224))
                        arr = np.array(image)
                        x_data.append(arr)
                        count = count + 1
            y_values = np.full((count,1),lookup[pose])
            y_data.append(y_values)
            datacount = datacount + count
    #x = x / 255
    x_data = np.array(x_data, dtype='float32')
    x_data = x_data.reshape((len(x_data), 224, 224, 3))
    x_data /= 255
    print(x_data)
    y_data = np.array(y_data)
    y_data = y_data.reshape(len(y_data), 1)
    y_data = to_categorical(y_data)
    print(y_data)
    return x_data,y_data


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
    print(y_train.shape, y_test.shape)
