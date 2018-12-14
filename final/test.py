import numpy as np
import pandas as pd
import torch


def read_data(file_name):
    data = pd.read_csv(file_name, encoding="big5")
    data = data.values
    label_only = True
    test_x = data[:, 0]
    return test_x


def genLabels_Partition(train_X, train_Y, valid_ratio=0.9):
    data_size = len(train_Y)
    labels = {train_X[i]: train_Y[i] for i in range(len(train_Y))}
    train_ids = [train_X[i] for i in range(int(data_size * (1 - valid_ratio)))]
    valid_ids = [
        train_X[i]
        for i in range(int(data_size * (1 - valid_ratio)), data_size)
    ]
    partition = {'train': train_ids, 'validation': valid_ids}

    return labels, partition

read_data('./data/test.csv')
