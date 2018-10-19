"""
Use probability generative model to train the model
"""
# from statistics import mean, stdev
# from sys import argv
from time import time
from sys import argv
from os import path
from random import randint, sample
from statistics import mean, stdev
import math

import numpy as np


class Data():
    """
    Class to read data and create it as an array
    """

    def __init__(self):
        self.data_matrix = np.array([])
        self.data_y = np.array([])
        self.mean = []
        self.stdev = []
        self.max = []

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

    def read_data(self, file_path_x, file_path_y):
        """
        read data from train.csv file
        """
        data = []
        data_y = []
        # open train_x.csv
        with open(file_path_x, 'rb') as file:
            for idx, line in enumerate(file):
                if (idx == 0):
                    continue
                row = str(line).split("'")[1].split('\\r')[0]
                items = row.split(',')
                data.append([float(x) for x in items])
        data = np.array(data)
        self.data_matrix = np.transpose(data)
        # open train_y.csv
        with open(file_path_y, 'rb') as file:
            for idx, line in enumerate(file):
                if idx == 0:
                    continue
                row = str(line).split("'")[1].split('\\r')[0]
                data_y.append(int(row))
        data_y = np.array(data_y)
        data_y = data_y.reshape(-1, 1)
        self.data_y = data_y

    def feature_scaling(self):
        """
        feature scaling
        """
        for dimension in self.data_matrix:
            max_data = dimension[0]
            for data in dimension:
                if data > max_data:
                    max_data = data
            self.max.append(max_data)
        for idx, dimension in enumerate(self.data_matrix):
            self.data_matrix[idx] = dimension / self.max[idx]
        self.data_matrix = np.array(self.data_matrix)
        for dimension in self.data_matrix:
            self.mean.append(mean(dimension))
            self.stdev.append(stdev(dimension))
        for idx, dimension in enumerate(self.data_matrix):
            self.data_matrix[idx] = (dimension - self.mean[idx]) / self.stdev[idx]
        self.data_matrix = np.array(self.data_matrix)

    def get_means(self):
        """
        get means of all dimension of x
        """
        return self.mean

    def get_stdevs(self):
        """
        get stdevs of all dimension of x
        """
        return self.stdev

    def get_max(self):
        """
        get max of all dimension of x
        """
        return self.max

    def get_corr(self):
        """
        get correlation coefficient to find the relationship between parameter and output
        """
        temp_y_data = np.ravel(self.data_y)
        temp_y_data = np.array(temp_y_data)
        for idx1, dimension in enumerate(self.data_matrix):
            corr = np.corrcoef([dimension, temp_y_data])
            print("d_%s  %.2f" % (idx1, corr[0, 1]), end='\t')
            if idx1 % 5 == 4:
                print()
        print()
        exit()

class Logestic():
    """
    A class to create a logestic model
    """

    def __init__(self, data):
        self.weights = []
        self.bias = 0.0
        self.dimension = 0
        self.data = data

    def default_rule(self):
        """
        default function for loss function's augument
        """
        return True

    def estimate_1_probability(self, x_value, weights, bias):
        """
        get Pr{x in class 1}
        """
        z_value = np.inner(weights, x_value)
        z_value += bias
        sigmoid = 1 / (1 + math.exp(-z_value))
        return sigmoid

    def loss(self,
             data_matrix_t,
             y_real_array,
             weights,
             bias,
             order=1,
             train_dimensions=range(23),
             good_data_rule=default_rule):
        """
        Get value of loss funtion from data in 'months' (list of months)
        """
        total_loss = 0.0
        data_number = 0
        for idx3, x_value in enumerate(data_matrix_t):
            x_train = []
            for dimension in train_dimensions:
                x_train.append(x_value[dimension])
            x_train = np.array(x_train)
            y_real = y_real_array[idx3][0]
            if not good_data_rule(x_train):
                continue
            if order == 4:
                x_train_temp = np.append(x_train**4, x_train**3)
                x_train_temp = np.append(x_train_temp, x_train**2)
                x_train = np.append(x_train_temp, x_train)
            if order == 3:
                x_train_temp = np.append(x_train**3, x_train**2)
                x_train = np.append(x_train_temp, x_train)
            if order == 2:
                x_train = np.append(x_train**2, x_train)
            x_train = np.array(x_train)
            y_sigmoid = self.estimate_1_probability(x_train, weights, bias)
            if y_sigmoid == 0 or y_sigmoid == 1:
                loss_predict = 10000
            else:
                loss_predict = -(y_real * math.log(y_sigmoid) +
                                 (1 - y_real) * math.log(1 - y_sigmoid))
            total_loss += loss_predict
            data_number += 1
        return total_loss / data_number / 2

    def get_correct_rate(self,
                         data_matrix_t,
                         y_real_array,
                         weights,
                         bias,
                         order=1,
                         train_dimensions=range(23),
                         threshold=0.5):
        """
        Get rate of correction in data_matrix
        """
        correct_number = 0
        data_number = 0
        for idx3, x_value in enumerate(data_matrix_t):
            x_train = []
            for dimension in train_dimensions:
                x_train.append(x_value[dimension])
            x_train = np.array(x_train)
            y_real = y_real_array[idx3][0]
            if order == 4:
                x_train_temp = np.append(x_train**4, x_train**3)
                x_train_temp = np.append(x_train_temp, x_train**2)
                x_train = np.append(x_train_temp, x_train)
            if order == 3:
                x_train_temp = np.append(x_train**3, x_train**2)
                x_train = np.append(x_train_temp, x_train)
            if order == 2:
                x_train = np.append(x_train**2, x_train)
            x_train = np.array(x_train)
            y_sigmoid = self.estimate_1_probability(x_train, weights, bias)
            result = 1
            if y_sigmoid < threshold:
                result = 0
            if result == y_real:
                correct_number += 1
            data_number += 1
        return correct_number / data_number

    def create_model(self, data_matrix, y_real_array, banch_num, epochs, rate, order=1,
                     train_dimensions=range(23), good_data_rule=default_rule,
                     regularization_parameter=0):
        """
        create generative model from data
        """
        data_matrix_t = np.transpose(data_matrix)
        temp_weights = []
        divider = []
        temp_bias = 0.1
        bias_divider = 0.0
        for _ in range(len(train_dimensions) * order):
            temp_weights.append(0.1)
            divider.append(0.0)
        temp_weights = np.array(temp_weights)
        divider = np.array(divider)

        best_loss = self.loss(data_matrix_t, y_real_array, temp_weights, temp_bias, order,
                              train_dimensions, good_data_rule)
        self.weights = np.copy(temp_weights)
        self.bias = temp_bias

        for j in range(epochs):

            data_amount = np.shape(data_matrix)[1]
            random_dimension = sample(range(0, data_amount), data_amount)

            for banch in range(int(data_amount / banch_num)):
                x_collection = []
                y_real_collection = []
                # train_banch = random_dimension[banch * banch_num: (banch + 1) *
                #                                banch_num]
                for data_idx in random_dimension[banch * banch_num:
                                                 (banch + 1) * banch_num]:
                    # randomly pick 'banch_num' number for a banch
                    # for _ in range(banch_num):
                    #     data_idx = randint(0, np.shape(data_matrix)[1] - 1)
                    # x_train = data_matrix_t[data_idx]
                    x_train = []
                    for dimension in train_dimensions:
                        x_train.append(data_matrix_t[data_idx][dimension])
                    x_train = np.array(x_train)
                    while not good_data_rule(x_train):
                        data_idx = randint(0, np.shape(data_matrix)[1] - 1)
                        x_train = []
                        for dimension in train_dimensions:
                            x_train.append(data_matrix_t[data_idx][dimension])
                        x_train = np.array(x_train)
                    if order == 4:
                        x_train_temp = np.append(x_train**4, x_train**3)
                        x_train_temp = np.append(x_train_temp, x_train**2)
                        x_train = np.append(x_train_temp, x_train)
                    if order == 3:
                        x_train_temp = np.append(x_train**3, x_train**2)
                        x_train = np.append(x_train_temp, x_train)
                    if order == 2:
                        x_train = np.append(x_train**2, x_train)
                    x_train = np.array(x_train)
                    y_real = y_real_array[data_idx][0]

                    x_collection.append(x_train)
                    y_real_collection.append(y_real)
                x_collection = np.array(x_collection)
                y_real_collection = np.array(y_real_collection)

                # gradient = (y_predict - y_real) * x
                y_predict = []
                for x_value in x_collection:
                    y_predict.append(self.estimate_1_probability(x_value, temp_weights, temp_bias))
                y_predict = np.array(y_predict)
                y_gap = y_predict - y_real_collection
                gradients = []
                for dimension in range(len(train_dimensions) * order):
                    x_dimension = x_collection[:, dimension:dimension + 1]
                    x_dimension = np.ravel(x_dimension)
                    x_dimension = np.array(x_dimension)
                    gradients.append(np.inner(x_dimension, y_gap))
                gradients = np.array(gradients)
                gradient_bias = np.sum(y_gap)

                # use adagrad for adjusting learning rate
                divider = divider + gradients**2
                bias_divider = bias_divider + gradient_bias**2
                temp_weights = temp_weights - (
                    (regularization_parameter * temp_weights + gradients) * rate /
                    (divider**(1 / 2)))
                temp_bias = temp_bias - (gradient_bias * rate / (bias_divider**(1/2)))

                # detect loss every 100 epochs calculation
                # if j % 100 == 99:
            temp_loss = self.loss(data_matrix_t, y_real_array, temp_weights, temp_bias,
                                  order, train_dimensions, good_data_rule)
            if temp_loss < best_loss:
                best_loss = temp_loss
                self.weights = np.copy(temp_weights)
                self.bias = temp_bias
            if regularization_parameter != 0:
                best_loss = temp_loss
                self.weights = np.copy(temp_weights)
                self.bias = temp_bias
            print(
                "Training %6s/%s epochs, min loss= %.7e" % (j + 1, epochs,
                                                            best_loss), end='\r')
        print()

    def export_model_parameter(self, output_file_path):
        """
        export the model to output file
        """
        file = open(output_file_path, "w+")
        file.write('weights,')
        for weight in self.weights:
            file.write(str(weight))
            file.write(',')
        file.write('\r\n')
        file.write('bias,')
        file.write(str(self.bias))
        file.write('\r\n')
        file.write('means,')
        for mean_value in self.data.get_means():
            file.write(str(mean_value) + ',')
        file.write('\r\n')
        file.write('stdevs,')
        for stdev_value in self.data.get_stdevs():
            file.write(str(stdev_value) + ',')
        file.write('\r\n')
        file.write('max,')
        for max_value in self.data.get_max():
            file.write(str(max_value) + ',')
        file.write('\r\n')
        file.close()

if __name__ == '__main__':
    START = time()

    PATH, FILENAME = path.split(__file__)
    OUTPUT_PATH = argv[1]

    RATE = 0.05
    EPOCHS = 300
    BANCH_NUM = 100
    ORDER = 3
    TRAINING_DIMENSION = [5, 6, 7, 8, 9, 10, 0]

    TRAINING_DATA = Data()
    TRAINING_DATA.read_data(
        file_path_x='./' + PATH + '/data/train_x.csv',
        file_path_y='./' + PATH + '/data/train_y.csv')
    TRAINING_DATA.feature_scaling()
    # TRAINING_DATA.get_corr()
    # exit()
    # def good_data(x_train):
    #     """
    #     define what kind of data is good
    #     """
    #     if any(x > 1.5 or x < -1.5 for x in x_train):
    #         return False
    #     return True

    # TEMP_MATRIX_T = np.transpose(TRAINING_DATA.get_data_matrix())
    # # RANGE = sample(range(20000), 20000)
    # TEMP_Y = TRAINING_DATA.get_real_y()
    # TRAIN_Y = np.array(TEMP_Y[:15000])
    # TEST_Y = np.array(TEMP_Y[15000:])
    # TRAIN_MATRIX_T = np.array(TEMP_MATRIX_T[:15000])
    # TEST_MATRIX_T = np.array(TEMP_MATRIX_T[15000:])


    # for i in range(10):
    #     print('___', end='')
    #     print(10**(5-i), end='')
    #     print('___')
    #     MODEL = Logestic(TRAINING_DATA)
    #     MODEL.create_model(
    #         np.transpose(TRAIN_MATRIX_T),
    #         TRAIN_Y,
    #         BANCH_NUM,
    #         EPOCHS,
    #         RATE,
    #         ORDER,
    #         train_dimensions=TRAINING_DIMENSION,
    #         regularization_parameter=10**(-i))
    #     LOSS = MODEL.loss(
    #         TEST_MATRIX_T,
    #         TEST_Y,
    #         MODEL.weights,
    #         MODEL.bias,
    #         ORDER,
    #         train_dimensions=TRAINING_DIMENSION)
    #     CORRECT_RATE = MODEL.get_correct_rate(
    #         TEST_MATRIX_T,
    #         TEST_Y,
    #         MODEL.weights,
    #         MODEL.bias,
    #         ORDER,
    #         train_dimensions=TRAINING_DIMENSION,
    #         threshold=0.4)
    #     print('loss=', LOSS)
    #     print('correct rate=', CORRECT_RATE)
    # exit()

    MODEL = Logestic(TRAINING_DATA)
    MODEL.create_model(
        TRAINING_DATA.get_data_matrix(),
        TRAINING_DATA.get_real_y(),
        BANCH_NUM,
        EPOCHS,
        RATE,
        ORDER,
        train_dimensions=TRAINING_DIMENSION)
        # regularization_parameter=10)
    # good_data_rule=good_data)
    MODEL.export_model_parameter(OUTPUT_PATH)
    STOP = time()
    for i in range(30):
        print('__%s__' % (0.35 + 0.01 * i))
        CORRECT_RATE = MODEL.get_correct_rate(
            np.transpose(TRAINING_DATA.get_data_matrix()),
            TRAINING_DATA.get_real_y(),
            MODEL.weights,
            MODEL.bias,
            ORDER,
            train_dimensions=TRAINING_DIMENSION,
            threshold=0.35 + 0.01 * i)
        print('correct rate=', CORRECT_RATE)
    print("Training finished!")
    print("Time used:", round((STOP - START) / 60), "min",
          round((STOP - START) % 60), "sec")
