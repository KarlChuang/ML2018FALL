"""
estimate the testing data
"""
from sys import argv
from os import path
import math

import numpy as np


class Data():
    """
    Class to read data and create it as an array
    """

    def __init__(self):
        self.data_matrix = np.array([])

    def get_data_matrix(self):
        """
        get data matrix create from data
        """
        return self.data_matrix

    def read_data(self, file_path_x):
        """
        read data from train.csv file
        """
        data = []
        # open train_x.csv
        with open(file_path_x, 'rb') as file:
            for idx1, line in enumerate(file):
                if (idx1 == 0):
                    continue
                row = str(line).split("'")[1].split('\\r')[0]
                items = row.split(',')
                data.append([float(x) for x in items])
        data = np.array(data)
        self.data_matrix = np.transpose(data)


class Logestic():
    """
    A class to create a logestic model
    """

    def __init__(self):
        self.weights = []
        self.means = []
        self.stdevs = []
        self.bias = 0.0
        self.dimension = 0

    def create_model(self, model_file_path):
        """
        create generative model from data
        """
        with open(model_file_path, 'rb') as file:
            temp_weights = []
            temp_means = []
            temp_stdevs = []
            for idx2, line in enumerate(file):
                if idx2 == 0:
                    row = str(line).split('\\r')[0].split(',')[1:]
                    row = row[:len(row) - 1]
                    temp_weights = [float(x) for x in row]
                elif idx2 == 1:
                    value = str(line).split('\\r')[0].split(',')[1]
                    self.bias = float(value)
                elif idx2 == 2:
                    row = str(line).split('\\r')[0].split(',')[1:]
                    row = row[:len(row) - 1]
                    temp_means = [float(x) for x in row]
                elif idx2 == 3:
                    row = str(line).split('\\r')[0].split(',')[1:]
                    row = row[:len(row) - 1]
                    temp_stdevs = [float(x) for x in row]
            self.weights = np.array(temp_weights)
            self.means = np.array(temp_means)
            self.stdevs = np.array(temp_stdevs)

    def estimate_y(self, x_value):
        """
        get likelihood value from generative model
        """
        x_value = np.array(x_value)
        x_feature_scaling = (x_value - self.means) / self.stdevs
        z_value = np.inner(self.weights, x_feature_scaling)
        z_value = z_value + self.bias
        sigmoid = 1 / (1 + math.exp(-z_value))
        if sigmoid < 0.5:
            return 0
        return 1

if __name__ == '__main__':
    PATH, FILENAME = path.split(__file__)
    TRAINING_DATA = Data()
    TRAINING_DATA.read_data(file_path_x=argv[1])
    MODEL = Logestic()
    MODEL.create_model('./' + PATH + '/model/logestic.csv')

    print('id,Value')
    for idx, x in enumerate(np.transpose(TRAINING_DATA.get_data_matrix())):
        print('id_' + str(idx), end=',')
        result = MODEL.estimate_y(x)
        print(result)
