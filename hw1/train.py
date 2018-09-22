import numpy as np
import math
from statistics import mean, stdev
from sys import argv

try:
    MODEL_NUM = argv[1]
except:
    MODEL_NUM = '0'

# read training data (train.csv)
DATA = []
for i in range(18):
    DATA.append([])

with open('./data/train.csv', 'rb') as file:
    for idx, line in enumerate(file):
        if (idx == 0):
            continue
        line = str(line)
        line = line.split('\\r')[0]
        items = line.split(',')[3:]
        for idx2, item in enumerate(items):
            try:
                num = float(item)
            except:
                DATA[(idx - 1) % 18].append(0.0)
            else:
                DATA[(idx - 1) % 18].append(num)

# feature scaling
MEAN = []
STDEV = []
for row in DATA:
    MEAN.append(mean(row))
    STDEV.append(stdev(row))
for idx1, row in enumerate(DATA):
    for idx2, data in enumerate(row):
        row[idx2] = (data - MEAN[idx1]) / STDEV[idx1]

# linear regression

# initial regression parameter
DATA_MATRIX = np.array(DATA)
RATE = 10.0
TIMES = 200
W_ERR_PAIR = []

# Loss(month_1, month_2): get Loss from data of month_1 to month_2
def Loss(W, month_1, month_2):
    err = 0.0
    for k in range(month_1, month_2):
        for i in range(471):
            x = DATA_MATRIX[:, i + (k%12) * 480:i + (k%12) * 480 + 9]
            x = x.ravel()
            x = np.append(x, [1.0])
            x = np.array(x)
            y = np.inner(x, W)
            y_real = DATA_MATRIX[9][i + (k%12) * 480 + 9]
            err += (y - y_real)**2
            # print(y_real * STDEV[9] + MEAN[9], y * STDEV[9] + MEAN[9])
    return err

# Train(times, month_1, month_2): train W from data of month_1 to month_2 times times
def Train(times, rate, month_1, month_2):
    W = []
    for i in range(163):
        W.append(1.0)
    W = np.array(W)
    divider = 0.0

    for j in range(times):
        for k in range(month_1, month_2):
            for i in range(471):
                x = DATA_MATRIX[:, i + (k % 12) * 480:i + (k % 12) * 480 + 9]
                x = x.ravel()
                x = np.append(x, [1.0])
                x = np.array(x)
                y = np.inner(x, W)
                y_real = DATA_MATRIX[9][i + (k % 12) * 480 + 9]
                gra = 2 * (y - y_real) * x
                divider = divider + np.sum(np.square(gra))
                W = W - rate * (y - y_real) / math.sqrt(divider) * x
    return W

# Training
for month in range(6):
    weight = Train(TIMES, RATE, month, month + 10)
    err = Loss(weight, month + 10, month + 12)
    W_ERR_PAIR.append((weight, err))

# Find smallest error weight
MIN_W = W_ERR_PAIR[0][0]
MIN_ERR = W_ERR_PAIR[0][1]
for pair in W_ERR_PAIR:
    if MIN_ERR < pair[1]:
        MIN_W = pair[0]

# write into file
f = open('./model/' + MODEL_NUM + '_mean.csv', 'w+')
for i in MEAN:
    f.write(str(i))
    f.write('\r\n')
f.close()

f = open('./model/' + MODEL_NUM + '_stdev.csv', 'w+')
for i in STDEV:
    f.write(str(i))
    f.write('\r\n')
f.close()

f = open("./model/" + MODEL_NUM + "_weight.csv", "w+")
for w in MIN_W:
    f.write(str(w))
    f.write('\r\n')
f.close()
