"""
Use to train a good model for PM2.5 prediction
"""
from statistics import mean, stdev
from sys import argv
from time import time
from os import path
from random import sample, randint
import math

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
                idx2 = (month % 12) * 480 + hour
                if data[air][idx2] <= 0:
                    if hour == 0:
                        data[air][idx2] = 0.75 * data[air][idx2 + 1] + 0.25 * data[air][idx2 + 2]
                    elif hour == 479:
                        data[air][idx2] = 0.75 * data[air][idx2 - 1] + 0.25 * data[air][idx2 - 2]
                    else:
                        data1 = data[air][idx2 - 1]
                        data3 = data[air][idx2 + 1]
                        if data3 <= 0:
                            data[air][idx2] = data1
                        else:
                            data[air][idx2] = data1 * 0.5 + data3 * 0.5

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

def default_rule(_):
    """
    default function for loss function's augument
    """
    return True

def loss(data_matrix,
         weights,
         mean_y,
         stdev_y,
         order=1,
         train_dimensions=range(18),
         good_data_rule=default_rule,
         get_rmse=False,
         get_l2_norm=False,
         testing_days=range(471)):
    """
    Get value of loss funtion from data in 'months' (list of months)
    """
    total_loss = 0.0
    data_number = 0
    for month in range(12):
        for day in testing_days:
            x_train = []
            for dimension in train_dimensions:
                x_train.append(data_matrix[dimension, day + (month % 12) * 480:day +
                                           (month % 12) * 480 + 9])
            # x_train = data_matrix[:, day + (month % 12) * 480:day + (month% 12) * 480 + 9]
            x_train = np.array(x_train)
            x_train = np.ravel(x_train)
            y_real = data_matrix[9][day + (month % 12) * 480 + 9] * stdev_y + mean_y

            # if isinstance(reasonable_data, list):
            #     if any(x < reasonable_data[0] or x > reasonable_data[1] for x in x_train):
            #         continue
            # data_number += 1
            if not good_data_rule(x_train):
                continue

            if order == 3:
                x_train_temp = np.append(x_train**3, x_train**2)
                x_train = np.append(x_train_temp, x_train)
            if order == 2:
                x_train = np.append(x_train**2, x_train)
            x_train = np.append(x_train, [1.0])
            x_train = np.array(x_train)
            y_predict = np.inner(x_train, weights)

            total_loss += (y_predict - y_real)**2
            data_number += 1
    # print(data_number)
    if get_rmse:
        return math.sqrt(total_loss / data_number)
    if get_l2_norm:
        return total_loss
    return total_loss / data_number / 2

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

def train(data_matrix,
          banch_num,
          times,
          rate,
          mean_y,
          stdev_y,
          order=1,
          train_dimensions=range(18),
          good_data_rule=default_rule,
          regularization_parameter=0,
          training_days=range(471)):
    """
    Training for good weight from 'data_matrix' in 'months' (list if month)
    'times' (integer) times with learning rate 'rate' and banch number 'banch_num'
    with dimension reduced dataset
    """
    weights = []
    divider = []
    for _ in range(len(train_dimensions) * 9 * order + 1):
        weights.append(0.0)
        divider.append(0.0)
    weights = np.array(weights)
    divider = np.array(divider)
    best_loss = loss(data_matrix, weights, mean_y, stdev_y, order,
                     train_dimensions, good_data_rule,
                     testing_days=training_days)
    best_weights = np.copy(weights)

    for j in range(times):
        x_collection = []
        y_real_collection = []

        # randomly pick 'banch_num' number for a banch
        for _ in range(banch_num):
            month = randint(0, 11)
            day = sample(training_days, 1)[0]
            x_train = []
            for dimension in train_dimensions:
                x_train.append(data_matrix[dimension, day + (month % 12) * 480:day +
                                           (month % 12) * 480 + 9])
            x_train = np.array(x_train)
            x_train = np.ravel(x_train)

            # if isinstance(reasonable_data, list):
            #     while any(x < reasonable_data[0] or x > reasonable_data[1] for x in x_train):
            while not good_data_rule(x_train):
                month = randint(0, 11)
                day = sample(training_days, 1)[0]
                x_train = []
                for dimension in train_dimensions:
                    x_train.append(
                        data_matrix[dimension, day + (month % 12) * 480:day +
                                    (month % 12) * 480 + 9])
                x_train = np.array(x_train)
                x_train = np.ravel(x_train)

            if order == 3:
                x_train_temp = np.append(x_train**3, x_train**2)
                x_train = np.append(x_train_temp, x_train)
            if order == 2:
                x_train = np.append(x_train**2, x_train)
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
        for dimension in range(len(train_dimensions) * 9 * order + 1):
            x_dimension = x_collection[:, dimension:dimension + 1]
            x_dimension = np.ravel(x_dimension)
            x_dimension = np.array(x_dimension)
            gradients.append(np.inner(x_dimension, y_gap))
        gradients = np.array(gradients)

        # use adagrad for adjusting learning rate
        divider = divider + gradients**2
        weights＿no_bias = np.copy(weights)
        weights＿no_bias[len(train_dimensions) * 9 * order] = 0
        weights = weights - (
            (regularization_parameter * weights＿no_bias + gradients) * rate / (divider**(1 / 2)))

        # detect loss every 100 times calculation
        if j % 100 == 99:
            temp_loss = loss(data_matrix, weights, mean_y, stdev_y,
                             order, train_dimensions, good_data_rule,
                             testing_days=training_days)
            if temp_loss < best_loss:
                best_loss = temp_loss
                best_weights = np.copy(weights)
            if regularization_parameter != 0:
                best_loss = temp_loss
                best_weights = np.copy(weights)
            print(
                "Training %6s/%s times, min loss= %.3e" %
                (j + 1, times, best_loss),
                end='\r')
    print()
    return best_weights

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
    RATE = 1.0
    TIMES = 25000
    BANCH_NUM = 100
    WEIGHTS_LOSS_PAIR = []
    TRAIN_NUM = 10
    ORDER = 1

    # read data
    DATA = read_file('./' + PATH + '/data/train.csv')

    # feature scaling
    (MEANS, STDEVS) = feature_scaling(DATA)


    # data_fixing(DATA)
    # print_corr(DATA)
    # exit()

    # create data to a matrix
    DATA_MATRIX = np.array(DATA)

    def data_good(train_data):
        """
        define what kind if data is good
        """
        # for air in range(int(len(train_data) / 9)):
        #     for hour in range(1, 8):
        #         idx = air * 9 + hour
        #         if train_data[idx] == 0:
        #             continue
        #         average = (train_data[idx - 1] + train_data[idx + 1]) / 2
        #         relative = abs((train_data[idx] - average) / train_data[idx])
        #         if relative > 20:
        #             return False
        if any(x < -2 or x > 2 for x in train_data):
            return False
        return True

    TRAIN_DIMENSIONS = [2, 3, 5, 6, 8, 9, 12, 13]
    # TRAIN_DIMENSIONS = [9]
    for i in range(5):
        ALL_DAYS = sample(range(471), 471)
        TRAINING_DAYS = ALL_DAYS[0:314]
        TESTING_DAYS = ALL_DAYS[314:471]
        print("___train___ (%d)" % (i))
        print("Training days= ", end='')
        print(TRAINING_DAYS)
        BEST_WEIGHTS = train(DATA_MATRIX, BANCH_NUM, TIMES, RATE, MEANS[9],
                             STDEVS[9], ORDER, TRAIN_DIMENSIONS,
                             good_data_rule=data_good,
                             regularization_parameter=0.1,
                             training_days=TRAINING_DAYS)
        LOSS = loss(
            DATA_MATRIX,
            BEST_WEIGHTS,
            MEANS[9],
            STDEVS[9],
            ORDER,
            TRAIN_DIMENSIONS,
            good_data_rule=data_good,
            testing_days=TESTING_DAYS)
        WEIGHTS_LOSS_PAIR.append((BEST_WEIGHTS, LOSS))


    # Find smallest error weight
    MIN_WEIGHTS = WEIGHTS_LOSS_PAIR[0][0]
    MIN_LOSS = WEIGHTS_LOSS_PAIR[0][1]
    for pair in WEIGHTS_LOSS_PAIR:
        if MIN_LOSS < pair[1]:
            MIN_WEIGHTS = pair[0]
            MIN_LOSS = pair[1]

    TOTAL_LOSS = loss(
        DATA_MATRIX,
        MIN_WEIGHTS,
        MEANS[9],
        STDEVS[9],
        ORDER,
        TRAIN_DIMENSIONS)

    print()
    print(' ' * 80, end='\r')
    print(
        "Testing...  Loss= %.3e  Full data loss= %.3e"
        % (MIN_LOSS, TOTAL_LOSS))

    # write file
    write_file('./' + PATH + '/model/' + MODEL_NUM + '_mean.csv', MEANS)
    write_file('./' + PATH + '/model/' + MODEL_NUM + '_stdev.csv', STDEVS)
    write_file('./' + PATH + '/model/' + MODEL_NUM + '_weight.csv', MIN_WEIGHTS)

    # Finish
    STOP = time()
    print("Training finished!")
    print("Time used:", round((STOP - START) / 60), "min", round((STOP - START) % 60), "sec")
