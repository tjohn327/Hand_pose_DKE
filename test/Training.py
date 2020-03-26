import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

import os, sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import buildDataset as dataset


def train():
    batch_size = 128
    epochs = 10
    learning_rate = 0.01
    model_name_to_save = "F:/Hand_pose/handpose_weights" + str(epochs) + ".h5"

    # image dimensions
    row, column = 28, 28

    # import dataset
    x_train, y_train, x_test, y_test = dataset.load_poses()
    num_classes = len(np.unique(y_test))

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, row, column)
        x_test = x_test.reshape(x_test.shape[0], 1, row, column)
        input_shape = (1, row, column)
    else:
        x_train = x_train.reshape(x_train.shape[0], row, column, 1)
        x_test = x_test.reshape(x_test.shape[0], row, column, 1)
        input_shape = (row, column, 1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)

    #x_validate, x_test, y_validate, y_test = train_test_split(x_test, y_test, test_size=0.5)

    # MODEL#
    # model building
    model = Sequential()

    # add layers
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=learning_rate),
                  metrics=['accuracy'])

    ##TRAIN##
    hist = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2,
                     validation_data=(x_test, y_test))

    # Evaluation results
    score = model.evaluate(x_test, y_test, verbose=1)

    # list all data in history
    print(hist.history.keys())
    print(hist.history['acc'])
    print(hist.history['val_acc'])
    print(hist.history['loss'])
    print(hist.history['val_loss'])

    print("Test loss:", score[0])
    print("Test accuracy", score[1])

    model.save(model_name_to_save)

    # Plot the loss and accuracy
    # to increase accuracy we show it in legends

    fig = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper right')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    train()
