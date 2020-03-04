import os
import keras
import matplotlib.style as style
import numpy as np
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

from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

style.use('seaborn-whitegrid')

lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('F:/Hand_gesture/dataset/leapGestRecog/00/'):
    if not j.startswith('.'):
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1

def load_data(init,end):
    x_data = []
    y_data = []
    datacount = 0
    for i in range(init, end):
        for j in os.listdir('F:/Hand_gesture/dataset/leapGestRecog/0' + str(i) + '/'): #01
            if not j.startswith('.'):
                count = 0
                for k in os.listdir('F:/Hand_gesture/dataset/leapGestRecog/0' +
                                    str(i) + '/' + j + '/'):                       #palm

                    img = Image.open('F:/Hand_gesture/dataset/leapGestRecog/0' +
                                     str(i) + '/' + j + '/' + k).convert('RGB')    #image

                    img = img.resize((224, 224))
                    arr = np.array(img)
                    x_data.append(arr)
                    count = count + 1
                y_values = np.full((count, 1), lookup[j])
                y_data.append(y_values)
                datacount = datacount + count

    print(y_data)
    return x_data, y_data


def data_processing(x_data, y_data):
    x_data = np.array(x_data, dtype='float32')
    x_data = x_data.reshape((len(x_data), 224, 224, 3))
    x_data /= 255

    y_data = np.array(y_data)
    y_data = y_data.reshape(len(y_data), 1)
    print(y_data)
    y_data = to_categorical(y_data)

    return x_data, y_data

x_train, y_train = load_data(0,3)
x_test, y_test = load_data(3,5)
