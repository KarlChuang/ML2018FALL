import numpy as np

# get weight matrix
W = []
MEAN = []
STDEV = []

with open('./hw1/model/1_weight.csv', 'rb') as file:
    for idx, line in enumerate(file):
        line = str(line).split("'")[1].split('\\r')[0]
        W.append(float(line))
W = np.array(W)
W = W.reshape(-1, 1)

with open('./hw1/model/1_mean.csv', 'rb') as file:
    for idx, line in enumerate(file):
        line = str(line).split("'")[1].split('\\r')[0]
        MEAN.append(float(line))

with open('./hw1/model/1_stdev.csv', 'rb') as file:
    for idx, line in enumerate(file):
        line = str(line).split("'")[1].split('\\r')[0]
        STDEV.append(float(line))

# read testing data (test.csv)
DATA = []
NAME = []
for i in range(18):
    DATA.append([])

with open('./hw1/data/test.csv', 'rb') as file:
    for idx, line in enumerate(file):
        line = str(line).split('\\n')[0]
        items = line.split(',')[2:]
        if idx % 18 == 0:
            NAME.append(line.split("'")[1].split(',')[0])
        for idx2, item in enumerate(items):
            try:
                num = float(item)
            except:
                DATA[(idx) % 18].append(0.0)
            else:
                DATA[(idx) % 18].append((num - MEAN[idx % 18]) / STDEV[idx % 18])

DATA_MATRIX = np.array(DATA)

# calculating the result
X = []
for i in range(len(NAME)):
    x = DATA_MATRIX[:, i * 9:i * 9 + 9]
    x = x.ravel()
    x = np.append(x, [1.0])
    X.append(x)
X = np.array(X)

Y = np.dot(X, W)
Y = Y * STDEV[9] + MEAN[9]
for idx, data in enumerate(Y):
    print(NAME[idx], end=',')
    print(int(round(data[0])))
