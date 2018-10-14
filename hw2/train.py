"""
Use probability generative model to train the model
"""
from statistics import mean, stdev
from sys import argv
from time import time
from os import path
import math

import numpy as np

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

    def divide_matrix(self):
        """
        According to y value to divide the data to two part
        """
        data_matrix_0 = []
        data_matrix_1 = []
        temp_data_matrix = np.transpose(self.data_matrix)
        for idx, row in enumerate(temp_data_matrix):
            if self.data_y[idx][0] == 0:
                data_matrix_0.append(row)
            else:
                data_matrix_1.append(row)
        data_matrix_0 = np.transpose(np.array(data_matrix_0))
        data_matrix_1 = np.transpose(np.array(data_matrix_1))
        return data_matrix_0, data_matrix_1

class Generative():
    """
    A class to create a generative model
    """
    def __init__(self):
        self.mean_0 = np.array([])
        self.mean_1 = np.array([])
        self.covariance_0 = np.array([])
        self.covariance_1 = np.array([])
        self.dimension = 0
        self.probability_class_0 = 0
        self.probability_class_1 = 0

    def create_model(self, data):
        """
        create generative model from data
        """
        (data_matrix_0, data_matrix_1) = data.divide_matrix()
        class_0_num = np.shape(data_matrix_0)[1]
        class_1_num = np.shape(data_matrix_1)[1]
        self.probability_class_0 = class_0_num / (class_0_num + class_1_num)
        self.probability_class_1 = class_1_num / (class_0_num + class_1_num)
        mean_0 = []
        mean_1 = []
        self.dimension = len(data_matrix_0)
        covariance_temp = np.zeros((self.dimension, self.dimension))
        data_num = 0
        for row in data_matrix_0:
            mean_0.append(mean(row))
        self.mean_0 = np.array(mean_0)
        for row in data_matrix_1:
            mean_1.append(mean(row))
        self.mean_1 = np.array(mean_1)
        self.mean_0 = self.mean_0.reshape(-1, 1)
        self.mean_1 = self.mean_1.reshape(-1, 1)
        for row in np.transpose(data_matrix_0):
            row = row.reshape(-1, 1)
            data_num += 1
            covariance_temp += np.dot((row - self.mean_0), np.transpose(row - self.mean_0))
        self.covariance_0 = covariance_temp / data_num
        data_num = 0
        covariance_temp = np.zeros((self.dimension, self.dimension))
        for row in np.transpose(data_matrix_1):
            row = row.reshape(-1, 1)
            data_num += 1
            covariance_temp += np.dot((row - self.mean_1), np.transpose(row - self.mean_1))
        self.covariance_1 = covariance_temp / data_num

    def use_equal_covariance(self):
        """
        use the same covariance for class0 and class1
        """
        probability_0 = self.probability_class_0 * self.covariance_0
        probability_1 = self.probability_class_1 * self.covariance_1
        self.covariance_0 = probability_0 + probability_1
        self.covariance_1 = self.covariance_0

    def estimate_y(self, x_value):
        """
        get likelihood value from generative model
        """
        covariance_deter_0 = 0
        exponent_0 = np.array([])

        covariance_deter_0 = np.linalg.det(self.covariance_0)
        exponent_0 = np.transpose(x_value - self.mean_0)
        exponent_0 = np.dot(exponent_0, np.linalg.inv(self.covariance_0))
        exponent_0 = np.dot(exponent_0, (x_value - self.mean_0))
        exponent_0 = -exponent_0[0][0] / 2

        covariance_deter_1 = 0
        exponent_1 = np.array([])
        covariance_deter_1 = np.linalg.det(self.covariance_1)
        exponent_1 = np.transpose(x_value - self.mean_1)
        exponent_1 = np.dot(exponent_1, np.linalg.inv(self.covariance_1))
        exponent_1 = np.dot(exponent_1, (x_value - self.mean_1))
        exponent_1 = -exponent_1[0][0] / 2
        covariance_deter_ratio = covariance_deter_0 / covariance_deter_1
        probability_class_ratio = self.probability_class_1 / self.probability_class_0
        likelihood_0 = 1 / (1 + covariance_deter_ratio * math.exp(
            exponent_1 - exponent_0) * probability_class_ratio)
        if likelihood_0 < 0.5:
            return 1
        return 0

    def export_model_parameter(self, output_file_path):
        """
        export the model to output file
        """
        file = open(output_file_path, "w+")
        file.write('mean_0,')
        for mean_value in self.mean_0:
            file.write(str(mean_value[0]))
            file.write(',')
        file.write('\r\n')
        file.write('mean_1,')
        for mean_value in self.mean_1:
            file.write(str(mean_value[0]))
            file.write(',')
        file.write('\r\n')
        file.write('covariance_0\r\n')
        for cov_row in self.covariance_0:
            for cov in cov_row:
                file.write(str(cov))
                file.write(',')
            file.write('\r\n')
        file.write('covariance_1\r\n')
        for cov_row in self.covariance_1:
            for cov in cov_row:
                file.write(str(cov))
                file.write(',')
            file.write('\r\n')
        file.write('probability_class_0,')
        file.write(str(self.probability_class_0))
        file.write('\r\n')
        file.write('probability_class_1,')
        file.write(str(self.probability_class_1))
        file.write('\r\n')
        file.close()

if __name__ == '__main__':
    PATH, FILENAME = path.split(__file__)
    TRAINING_DATA = Data()
    TRAINING_DATA.read_data(file_path_x='./' + PATH + '/data/train_x.csv',
                            file_path_y='./' + PATH + '/data/train_y.csv')
    MODEL = Generative()
    MODEL.create_model(TRAINING_DATA)
    MODEL.use_equal_covariance()
    MODEL.export_model_parameter('./' + PATH + 'model/generative.csv')
