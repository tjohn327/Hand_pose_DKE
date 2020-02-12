import cv2
import numpy as np
from tensorflow import Graph,Session
import  tensorflow as tf
import os
import keras

os.environ['KERAS_BACKEND'] = 'tensorflow'

def classify(model,graph,sess,image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.flip(image,1)

    #Reshape
    res = cv2.resize(image,(28,28), interpolation=cv2.INTER_AREA)

    #Convert to float values between 0 and 1
    res = res.atype(dtype="float64")
    res = res /  255
    res = np.reshape(res, (1,28,28,1))

    with graph.as_default():
        with sess.as_default():
            prediction = model.predict(res)

    return prediction[0]

