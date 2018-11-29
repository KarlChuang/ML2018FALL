"""
CNN training
"""
from sys import argv
from contextlib import redirect_stdout

import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

TRAINING_FILE_NAME = argv[1]
OUTPUT_FILE_PATH = argv[2]
OUTPUT_MODEL_SUMMARY_PATH = argv[2].split('.hs')[0] + '.txt'

class Data():
    """
    Class to read data and create it as an array
    """

    def __init__(self):
        self.data_matrix = np.array([])
        self.data_y = np.array([])

    def get_data_matrix(self):
        """
        get data matrix create from data
        """
        return self.data_matrix

    def get_real_y(self):
        """
        get label value
        """
        return self.data_y

    def read_data(self, file_path):
        """
        read data from train.csv file
        """
        data = []
        data_y = []
        # open train_x.csv
        with open(file_path, 'rb') as file:
            for idx, line in enumerate(file):
                if (idx == 0):
                    continue
                row = str(line).split("'")[1].split('\\n')[0]
                items = row.split(',')
                temp_y = []
                for idx2 in range(7):
                    if idx2 == int(items[0]):
                        temp_y.append(1)
                    else:
                        temp_y.append(0)
                data_y.append(temp_y)
                temp_x = items[1].split(' ')
                temp_x_3d = [[[float(y) / 255] for y in temp_x[x:x + 48]]
                             for x in range(0, len(temp_x), 48)]
                data.append(temp_x_3d)
        self.data_matrix = np.array(data)
        self.data_y = np.array(data_y)

class Model():
    """
    Use to create a model which can train
    """
    def __init__(self):
        self.data = Data()
        self.model = Sequential()

    def set_data(self, data):
        """
        set model data set
        """
        self.data = data

    def create_model(self):
        """
        use Keras to build a network
        """
        # Convolutional layers
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                              activation='relu',
                              input_shape=(48, 48, 1)))
        self.model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                              activation='relu'))
        self.model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
                              activation='relu'))
        self.model.add(Conv2D(filters=512, kernel_size=(2, 2), strides=(1, 1),
                              activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))

        # Fully connection layers
        self.model.add(Flatten())
        self.model.add(Dense(2048, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1024, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(7, activation='softmax'))

    def compile(self):
        """
        compile the model
        """
        sgd = SGD(lr=0.005, decay=0.000001, momentum=0.9)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=sgd,
            metrics=['accuracy'])

    def training(self, output_file_path, output_model_path):
        """
        training the model
        """
        training_data_x = self.data.get_data_matrix()
        training_data_y = self.data.get_real_y()
        with open(output_model_path, 'w') as file:
            with redirect_stdout(file):
                self.model.summary()
        # for idx2 in range(20):
            # print("%3s / 20" % (idx2 + 1))
        self.model.fit(
            training_data_x,
            training_data_y,
            validation_split=0.2,
            batch_size=100,
            epochs=100,
            verbose=1)
        self.model.save(output_file_path)

    def train_with_data_generator(self, output_file_path, output_model_path):
        """
        Use Image data generator to generate new data
        """
        validation_split = 0.2
        data_x = self.data.get_data_matrix()
        data_y = self.data.get_real_y()
        validation_data_x = data_x[:int(len(data_x) * validation_split)]
        validation_data_y = data_y[:int(len(data_x) * validation_split)]
        training_data_x = data_x[int(len(data_x) * validation_split):]
        training_data_y = data_y[int(len(data_x) * validation_split):]
        with open(output_model_path, 'w') as file:
            with redirect_stdout(file):
                self.model.summary()
        train_datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True)
        train_datagen.fit(training_data_x)
        train_generator = train_datagen.flow(
            training_data_x,
            training_data_y,
            batch_size=100)
        # for idx2 in range(20):
            # print("%3s / 20" % (idx2 + 1))
        self.model.fit_generator(
            generator=train_generator,
            steps_per_epoch=500,
            validation_data=(validation_data_x, validation_data_y),
            epochs=100)
        self.model.save(output_file_path)

    def load_model(self, model_file_path):
        """
        load model from outside file
        """
        self.model = load_model(model_file_path)


if __name__ == '__main__':
    DATA = Data()
    DATA.read_data(TRAINING_FILE_NAME)
    MODEL = Model()
    MODEL.set_data(DATA)
    MODEL.create_model()
    MODEL.compile()
    MODEL.train_with_data_generator(
        output_file_path=OUTPUT_FILE_PATH,
        output_model_path=OUTPUT_MODEL_SUMMARY_PATH)
