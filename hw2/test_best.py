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

    def one_hot_encoding(self):
        """
        make several dimension to one hot dimension
        """
        new_data_matrix = []
        for idx1, dimension in enumerate(self.data_matrix):
            if idx1 == 1:
                new_dimension_1 = []
                new_dimension_2 = []
                for k in dimension:
                    if k == 1:
                        new_dimension_1.append(1)
                        new_dimension_2.append(0)
                    else:
                        new_dimension_1.append(0)
                        new_dimension_2.append(1)
                new_data_matrix.append(new_dimension_1)
                new_data_matrix.append(new_dimension_2)
            elif idx1 == 2:
                new_dimension = []
                for _ in range(7):
                    new_dimension.append([])
                for k in dimension:
                    for idx2, list1 in enumerate(new_dimension):
                        if idx2 == k:
                            list1.append(1)
                        else:
                            list1.append(0)
                for list1 in new_dimension:
                    new_data_matrix.append(list1)
            elif idx1 == 3:
                new_dimension = []
                for _ in range(4):
                    new_dimension.append([])
                for k in dimension:
                    for idx2, list1 in enumerate(new_dimension):
                        if idx2 == k:
                            list1.append(1)
                        else:
                            list1.append(0)
                for list1 in new_dimension:
                    new_data_matrix.append(list1)
            else:
                new_data_matrix.append(dimension)
        new_data_matrix = np.array(new_data_matrix)
        self.data_matrix = new_data_matrix

class Logestic():
    """
    A class to create a logestic model
    """

    def __init__(self):
        self.weights = []
        self.means = []
        self.stdevs = []
        self.max = []
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
            temp_max = []
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
                elif idx2 == 4:
                    row = str(line).split('\\r')[0].split(',')[1:]
                    row = row[:len(row) - 1]
                    temp_max = [float(x) for x in row]
            self.weights = np.array(temp_weights)
            self.means = np.array(temp_means)
            self.stdevs = np.array(temp_stdevs)
            self.max = np.array(temp_max)

    def estimate_y(self, x_value, test_dimensions=range(23), order=1, threshold=0.5):
        """
        get likelihood value from generative model
        """
        x_temp = []
        mean_temp = []
        stdev_temp = []
        max_temp = []
        for idx1 in test_dimensions:
            x_temp.append(x_value[idx1])
            mean_temp.append(self.means[idx1])
            stdev_temp.append(self.stdevs[idx1])
            max_temp.append(self.max[idx1])
        x_temp = np.array(x_temp)
        mean_temp = np.array(mean_temp)
        stdev_temp = np.array(stdev_temp)
        max_temp = np.array(max_temp)
        x_feature_scaling = ((x_temp - mean_temp) / stdev_temp) / max_temp
        if order == 4:
            x_train_temp = np.append(x_feature_scaling**4, x_feature_scaling**3)
            x_train_temp = np.append(x_train_temp, x_feature_scaling**2)
            x_feature_scaling = np.append(x_train_temp, x_feature_scaling)
        if order == 3:
            x_train_temp = np.append(x_feature_scaling**3, x_feature_scaling**2)
            x_feature_scaling = np.append(x_train_temp, x_feature_scaling)
        if order == 2:
            x_feature_scaling = np.append(x_feature_scaling**2, x_feature_scaling)
        x_feature_scaling = np.array(x_feature_scaling)
        z_value = np.inner(self.weights, x_feature_scaling)
        z_value = z_value + self.bias
        sigmoid = 1 / (1 + math.exp(-z_value))
        if sigmoid < threshold:
            return 0
        return 1

if __name__ == '__main__':
    PATH, FILENAME = path.split(__file__)
    TRAINING_DATA = Data()
    TRAINING_DATA.read_data(file_path_x=argv[1])
    TRAINING_DATA.one_hot_encoding()

    MODEL = Logestic()
    MODEL.create_model('./' + PATH + '/model/logestic_5.csv')
    # TEST_DIMENSION = [5, 6, 7, 8, 9, 10, 0]
    TEST_DIMENSION = range(33)
    print('id,Value')
    for idx, x in enumerate(np.transpose(TRAINING_DATA.get_data_matrix())):
        print('id_' + str(idx), end=',')
        result = MODEL.estimate_y(x, test_dimensions=TEST_DIMENSION, order=3, threshold=0.4)
        print(result)
