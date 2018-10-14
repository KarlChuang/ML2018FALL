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

    def create_model(self, model_file_path):
        """
        create generative model from data
        """
        with open(model_file_path, 'rb') as file:
            temp_covariance_0 = []
            temp_covariance_1 = []
            for idx2, line in enumerate(file):
                if idx2 < 2:
                    row = str(line).split('\\r')[0].split(',')[1:]
                    row = row[:len(row) - 1]
                    row = [float(x) for x in row]
                    if idx2 == 0:
                        self.mean_0 = np.array(row)
                        self.mean_0 = self.mean_0.reshape(-1, 1)
                    if idx2 == 1:
                        self.mean_1 = np.array(row)
                        self.mean_1 = self.mean_1.reshape(-1, 1)
                elif idx2 > 2 and idx2 < 50:
                    row = str(line).split("'")[1].split('\\r')[0].split(',')
                    row = row[:len(row) - 1]
                    row = [float(x) for x in row]
                    if idx2 < 26 and idx2 > 2:
                        temp_covariance_0.append(row)
                    elif idx2 > 26 and idx2 < 50:
                        temp_covariance_1.append(row)
                elif idx2 == 50:
                    row = str(line).split('\\r')[0].split(',')
                    self.probability_class_0 = float(row[1])
                elif idx2 == 51:
                    row = str(line).split('\\r')[0].split(',')
                    self.probability_class_1 = float(row[1])
            self.covariance_0 = np.array(temp_covariance_0)
            self.covariance_1 = np.array(temp_covariance_1)

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

if __name__ == '__main__':
    PATH, FILENAME = path.split(__file__)
    TRAINING_DATA = Data()
    TRAINING_DATA.read_data(file_path_x=argv[1])
    MODEL = Generative()
    MODEL.create_model('./' + PATH + '/model/generative.csv')
    MODEL.use_equal_covariance()

    print('id,Value')
    for idx, x in enumerate(np.transpose(TRAINING_DATA.get_data_matrix())):
        print('id_' + str(idx), end=',')
        result = MODEL.estimate_y(x.reshape(-1, 1))
        print(result)
