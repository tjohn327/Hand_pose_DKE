import os
import keras
import matplotlib.style as style
import numpy as np
# import tables
from PIL import Image
from keras import optimizers
from keras.applications import VGG16
from keras.callbacks import Callback
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Input
from keras.models import Model
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# %matplotlib inline
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

style.use('seaborn-whitegrid')

from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

def load_data():
    lookup = dict()
    reverselookup = dict()
    count = 0
    count_image = 0
    for j in os.listdir('F:/Github/Hand_pose_DKE/Poses/'):
        for k in os.listdir('F:/Github/Hand_pose_DKE/Poses/' + j + '/'):
            for l in os.listdir('F:/Github/Hand_pose_DKE/Poses/' + j + '/' + k + '/'):
                count_image = count_image + 1
        if not j.startswith('.'):
            lookup[j] = count
            reverselookup[count] = j
            count = count + 1

    # print(count_image)

    print(lookup)

    x_data = []
    y_data = np.empty(count_image)
    datacount = 0
    poses_directory = os.listdir('F:/Hand_pose/Poses/')

    # if (poses[0] == 'all'):
    poses = poses_directory.copy()

    for pose in poses_directory:  # dang

        if pose in poses:
            subdirs = os.listdir("F:/Github/Hand_pose_DKE/Poses/" + pose + '/')
            for subdir in subdirs:  # dang1
                files = os.listdir("F:/Github/Hand_pose_DKE/Poses/" + pose + '/' + subdir + '/')
                count = 0
                for file in files:  # image
                    if (file.endswith('.png')):
                        path = 'F:/Github/Hand_pose_DKE/Poses/' + pose + '/' + subdir + '/' + file

                        img = Image.open(path).convert('RGB')

                        img = img.resize((224, 224))
                        arr = np.array(img)
                        x_data.append(arr)
                        count = count + 1
                        y_data[count] = lookup[pose]
                #datacount = datacount + count

    #print(y_data)

    return x_data, y_data


def data_processing(x_data, y_data):
    x_data = np.array(x_data, dtype='float32')
    x_data = x_data.reshape((len(x_data), 224, 224, 3))
    x_data /= 255
    #print(x_data)

    y_data = np.array(y_data, dtype='float32')
    #print(y_data)
    y_data = y_data.reshape(len(y_data), 1)
    y_data = to_categorical(y_data)
    #print(y_data)
    return x_data, y_data


def load_dataset():
    x, y = load_data()
    x, y = data_processing(x, y)

    index = int(0.8 * x.shape[0])
    x_train = x[:index][:][:][:]
    x_test = x[index:][:][:][:]
    y_train = y[:index]
    y_test = y[index:]

    return x_train,y_train,x_test,y_test

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_dataset()