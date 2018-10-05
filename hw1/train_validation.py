"""
Use to train a good model for PM2.5 prediction
"""
from statistics import mean, stdev
from sys import argv
from time import time
from os import path
from random import randint, sample

import numpy as np

def read_file(file_path):
    """
    Read training data from 'file_path' and return a data matrix
    """
    data = []
    for _ in range(18):
        data.append([])
    with open(file_path, 'rb') as file:
        for idx, line in enumerate(file):
            if (idx == 0):
                continue
            row = str(line).split('\\r')[0]
            items = row.split(',')[3:]
            for item in items:
                try:
                    num = float(item)
                except ValueError:
                    data[(idx - 1) % 18].append(0.0)
                else:
                    data[(idx - 1) % 18].append(num)
    return data


def data_fixing(data):
    """
    fix data to make sense for actual situation
    """
    for month in range(12):
        for hour in range(480):
            for air in range(18):
                try:
                    data1 = data[air][(month % 12) * 480 + hour - 1]
                    data2 = data[air][(month % 12) * 480 + hour]
                    data3 = data[air][(month % 12) * 480 + hour + 1]
                except IndexError:
                    continue
                if data2 <= 0:
                    if hour == 0:
                        data[air][(month % 12) * 480 + hour] = data3
                    elif hour == 479:
                        data[air][(month % 12) * 480 + hour] = data1
                    else:
                        if data3 <= 0:
                            data[air][(month % 12) * 480 + hour] = data1
                        else:
                            data[air][(month % 12) * 480 + hour] = data1 * 0.5 + data3 * 0.5



def print_corr(data):
    """
    find correlation coefficient for all xi to y
    """
    data_matrix = np.array(data)
    for air in range(18):
        print("\nAIR__" + str(air))
        for dimension in range(9):
            x_collection = []
            y_collection = []
            for month in range(12):
                for day in range(471):
                    x_data = data_matrix[air, day + (month % 12) * 480 + dimension]
                    y_data = data_matrix[9, day + (month % 12) * 480 + 9]
                    x_collection.append(x_data)
                    y_collection.append(y_data)

            x_collection = np.array(x_collection)
            y_collection = np.array(y_collection)
            corr = np.corrcoef([x_collection, y_collection])
            print(str(dimension), "%.2f" % corr[0, 1], end='\t\t')
            if dimension % 5 == 4:
                print()
        print()

def feature_scaling(data):
    """
    feature scaling for 'data'
    """
    means = []
    stdevs = []
    for row in data:
        means.append(mean(row))
        stdevs.append(stdev(row))
    for idx1, row in enumerate(data):
        for idx2, column in enumerate(row):
            try:
                row[idx2] = (column - means[idx1]) / stdevs[idx1]
            except ZeroDivisionError:
                row[idx2] = (column - means[idx1])
    return (means, stdevs)


def loss(data_matrix, weights, months, mean_y, stdev_y):
    """
    Get value of loss funtion from data in 'months' (list of months)
    """
    total_loss = 0.0
    for month in months:
        for day in range(471):
            x_train = np.array([])
            x_train = data_matrix[:, day + (month % 12) * 480:day + (month% 12) * 480 + 9]
            x_train = np.ravel(x_train)
            y_real = data_matrix[9][day + (month % 12) * 480 + 9] * stdev_y + mean_y
            # x_train = np.append(x_train**2, x_train)
            x_train = np.append(x_train, [1.0])
            x_train = np.array(x_train)

            y_predict = np.inner(x_train, weights)

            total_loss += (y_predict - y_real)**2
    return total_loss


def train(data_matrix, banch_num, times, rate, months, mean_y, stdev_y):
    """
    Training for good weight from 'data_matrix' in 'months' (list if month)
    'times' (integer) times with learning rate 'rate' and banch number 'banch_num'
    """
    weights = []
    divider = []
    for _ in range(163):
        weights.append(0.0)
        divider.append(0.0)
    weights = np.array(weights)
    divider = np.array(divider)
    best_loss = loss(data_matrix, weights, months, mean_y, stdev_y)
    best_weights = np.copy(weights)

    for j in range(times):
        x_collection = []
        y_real_collection = []

        # randomly pick 'banch_num' number for a banch
        for _ in range(banch_num):
            month = sample(months, 1)[0]
            day = randint(0, 470)
            x_train = data_matrix[:, day + (month % 12) * 480:day + (month % 12) * 480 + 9]
            x_train = np.ravel(x_train)
            # x_train = np.append(x_train**2, x_train)
            x_train = np.append(x_train, [1.0])
            x_train = np.array(x_train)
            y_real = data_matrix[9][day + (month % 12) * 480 + 9] * stdev_y + mean_y

            x_collection.append(x_train)
            y_real_collection.append(y_real)
        x_collection = np.array(x_collection)
        y_real_collection = np.array(y_real_collection)

        # gradient = (y_predict - y_real) * x
        y_predict = np.dot(x_collection, weights)
        y_gap = y_predict - y_real_collection
        gradients = []
        for dimension in range(163):
            x_dimension = x_collection[:, dimension:dimension + 1]
            x_dimension = np.ravel(x_dimension)
            x_dimension = np.array(x_dimension)
            gradients.append(np.inner(x_dimension, y_gap))
        gradients = np.array(gradients)

        # use adagrad for adjusting learning rate
        divider = divider + gradients**2
        weights = weights - ((rate * gradients) / (divider**(1 / 2)))

        # detect loss every 100 times calculation
        if j % 100 == 99:
            temp_loss = loss(data_matrix, weights, months, mean_y, stdev_y)
            if temp_loss < best_loss:
                best_loss = temp_loss
                best_weights = np.copy(weights)
            print("Training %6s/%s times, min loss= %.3e" % (j + 1, times, best_loss), end='\r')
    return best_weights

def get_weight_gradient_0(data_matrix):
    """
    Use mathematical way to calculate best weight when gradient equals to zero
    """
    x_collection = []
    y_real_collection = []
    for month in range(12):
        for day in range(471):
            x_temp = data_matrix[:, day + (month % 12) * 480:day + (month% 12) * 480 + 9]
            x_temp = x_temp.ravel()
            x_temp = np.append(x_temp, [1.0])
            y_real = data_matrix[9][day + (month % 12) * 480 + 9]
            x_temp = np.array(x_temp)
            x_collection.append(x_temp)
            y_real_collection.append(y_real)
    x_collection = np.array(x_collection)
    y_real_collection = np.array(y_real_collection)
    x_collection_transpose = np.transpose(x_collection)
    best_weights = np.dot(x_collection_transpose, x_collection)
    best_weights = np.dot(np.linalg.inv(best_weights), x_collection_transpose)
    best_weights = np.dot(best_weights, y_real_collection)
    return best_weights

def write_file(file_path, data_array):
    """"
    write file
    """
    file = open(file_path, "w+")
    for data in data_array:
        file.write(str(data))
        file.write('\r\n')
    file.close()

if __name__ == '__main__':
    # system setting
    PATH, FILENAME = path.split(__file__)
    START = time()

    try:
        MODEL_NUM = argv[1]
    except IndexError:
        print('You do not choose the which result number for output.')
        print('Use \"0\" by default')
        MODEL_NUM = '0'

    # initial parameters
    RATE = 10.0
    TIMES = 25000
    BANCH_NUM = 100
    WEIGHTS_LOSS_PAIR = []
    TRAIN_NUM = 40

    # read data
    DATA = read_file('./' + PATH + '/data/train.csv')

    # feature scaling
    (MEANS, STDEVS) = feature_scaling(DATA)


    # data_fixing(DATA)
    # print_corr(DATA)
    # exit()

    # create data to a matrix
    DATA_MATRIX = np.array(DATA)

    for i in range(TRAIN_NUM):
        print("___"+ str(i + 1) + "___")
        MONTHS = sample(range(12), 12)
        MONTHS_1 = MONTHS[0:6]
        MONTHS_2 = MONTHS[6:12]
        BEST_WEIGHTS = train(DATA_MATRIX, BANCH_NUM, TIMES, RATE, MONTHS_1,
                             MEANS[9], STDEVS[9])
        LOSS = loss(DATA_MATRIX, BEST_WEIGHTS, MONTHS_2, MEANS[9], STDEVS[9])
        WEIGHTS_LOSS_PAIR.append((BEST_WEIGHTS, LOSS))
        print("Testing...  Loss= %.3e                                      " % LOSS)

    # Find smallest error weight
    MIN_WEIGHTS = WEIGHTS_LOSS_PAIR[0][0]
    MIN_LOSS = WEIGHTS_LOSS_PAIR[0][1]
    for pair in WEIGHTS_LOSS_PAIR:
        if MIN_LOSS < pair[1]:
            MIN_WEIGHTS = pair[0]

    # write file
    write_file('./' + PATH + '/model/' + MODEL_NUM + '_mean.csv', MEANS)
    write_file('./' + PATH + '/model/' + MODEL_NUM + '_stdev.csv', STDEVS)
    write_file('./' + PATH + '/model/' + MODEL_NUM + '_weight.csv', MIN_WEIGHTS)

    # Finish
    STOP = time()
    print("Training finished!")
    print("Time used:", round((STOP - START) / 60), "min", round((STOP - START) % 60), "sec")
