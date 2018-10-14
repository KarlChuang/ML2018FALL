"""
Use probability generative model to train the model
"""
# from statistics import mean, stdev
# from sys import argv
from time import time
from os import path
from random import randint
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

    def create_model(self, data_matrix, y_real_array, banch_num, times, rate, order=1,
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

        for j in range(times):
            x_collection = []
            y_real_collection = []

            # randomly pick 'banch_num' number for a banch
            for _ in range(banch_num):
                data_idx = randint(0, np.shape(data_matrix)[1] - 1)
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

            # detect loss every 100 times calculation
            if j % 100 == 99:
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
                    "Training %6s/%s times, min loss= %.3e" %
                    (j + 1, times, best_loss),
                    end='\r')
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
        file.close()

if __name__ == '__main__':
    START = time()

    PATH, FILENAME = path.split(__file__)

    RATE = 0.05
    TIMES = 10000
    BANCH_NUM = 100
    ORDER = 1

    TRAINING_DATA = Data()
    TRAINING_DATA.read_data(
        file_path_x='./' + PATH + '/data/train_x.csv',
        file_path_y='./' + PATH + '/data/train_y.csv')
    TRAINING_DATA.feature_scaling()
    MODEL = Logestic(TRAINING_DATA)
    MODEL.create_model(TRAINING_DATA.get_data_matrix(),
                       TRAINING_DATA.get_real_y(), BANCH_NUM, TIMES, RATE,
                       ORDER)
    MODEL.export_model_parameter('./' + PATH + 'model/logestic.csv')
    STOP = time()
    print("Training finished!")
    print("Time used:", round((STOP - START) / 60), "min",
          round((STOP - START) % 60), "sec")
