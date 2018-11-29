"""
Testing for homework report
"""
from sys import argv
from contextlib import redirect_stdout

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':
    MODEL = Sequential()
    MODEL.add(Dense(1509, activation='relu'))

    for _ in range(4):
        MODEL.add(Dense(1509, activation='relu'))
        MODEL.add(Dense(1509, activation='relu'))
        MODEL.add(Dropout(0.5))
    MODEL.add(Dense(7, activation='softmax'))

    MODEL.summary()
