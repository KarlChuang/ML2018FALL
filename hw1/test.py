"""
Use to predict result from testing data
"""
from sys import argv
from os import path

import numpy as np

def read_model_data(file_path):
    """
    read file to create model
    """
    data = []
    with open(file_path, 'rb') as file:
        for line in file:
            line = str(line).split("'")[1].split('\\r')[0]
            data.append(float(line))
    return data

def read_file(file_path):
    """
    read file from testing data
    """
    data_collection = []
    name_collection = []
    for _ in range(18):
        data_collection.append([])

    with open(file_path, 'rb') as file:
        for idx, line in enumerate(file):
            line = str(line).split('\\r')[0]
            items = line.split(',')[2:]
            if idx % 18 == 0:
                name_collection.append(line.split("'")[1].split(',')[0])
            for item in items:
                try:
                    num = float(item)
                except ValueError:
                    data_collection[(idx) % 18].append(0.0)
                else:
                    data_collection[(idx) % 18].append(num)
    return data_collection, name_collection


def feature_scaling(data, means, stdevs):
    """
    feature scaling for 'data'
    """
    for idx1, row in enumerate(data):
        for idx2, column in enumerate(row):
            try:
                row[idx2] = (column - means[idx1]) / stdevs[idx1]
            except ZeroDivisionError:
                row[idx2] = (column - means[idx1])

def predict_result(weights, data_matrix):
    """
    calculate the predict result
    """
    x_collection = []
    for i in range(len(NAME)):
        x_temp = data_matrix[:, i * 9:i * 9 + 9]
        x_temp = x_temp.ravel()

        x_temp = np.append(x_temp**2, x_temp)
        x_temp = np.append(x_temp, [1.0])
        x_collection.append(x_temp)
    x_collection = np.array(x_collection)
    y_predict = np.dot(x_collection, weights)
    return y_predict

def print_result(name, y_predict):
    """
    print the result
    """
    print('id', end=',')
    print('value')
    for idx, data in enumerate(y_predict):
        print(name[idx], end=',')
        print("%.1f" % (data[0]))

if __name__ == '__main__':
    PATH, FILENAME = path.split(__file__)

    # File Name
    try:
        TESTING_FILE = argv[1]
    except IndexError:
        print('You do not enter the file path to predict result!')
        exit()
    try:
        MODEL_NUM = argv[2]
    except IndexError:
        MODEL_NUM = '0'

    # get weight matrix, mean list, and stdev list
    WEIGHTS = read_model_data('./' + PATH + '/model/' + MODEL_NUM + '_weight.csv')
    WEIGHTS = np.array(WEIGHTS)
    WEIGHTS = WEIGHTS.reshape(-1, 1)
    MEANS = read_model_data('./' + PATH + '/model/' + MODEL_NUM + '_mean.csv')
    STDEVS = read_model_data('./' + PATH + '/model/' + MODEL_NUM + '_stdev.csv')

    # read testing data
    (DATA, NAME) = read_file(TESTING_FILE)

    # feature sacling
    feature_scaling(DATA, MEANS, STDEVS)

    # create data to a matrix
    DATA_MATRIX = np.array(DATA)

    # calculate predict result
    Y_PREDICT = predict_result(WEIGHTS, DATA_MATRIX)

    # print result
    print_result(NAME, Y_PREDICT)
