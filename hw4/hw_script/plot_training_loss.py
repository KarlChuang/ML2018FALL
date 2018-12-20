from os import path

import numpy as np
import matplotlib.pyplot as plt

def loading_log(file_path):
    """
    loading training loss
    """
    loss_list = []
    accuracy_list = []
    with open(file_path) as file:
        temp_loss_list = []
        temp_accuracy_list = []
        for idx, line in enumerate(file):
            if idx == 0:
                continue
            loss = float(line.split('(')[1].split(',')[0])
            accuracy = float(line.split(',')[2].split('\n')[0])
            temp_loss_list.append(loss)
            temp_accuracy_list.append(accuracy)
            if idx % 100 == 0:
                loss_list.append(sum(temp_loss_list) / len(temp_loss_list))
                accuracy_list.append(sum(temp_accuracy_list) / len(temp_accuracy_list))
                temp_loss_list = []
                temp_accuracy_list = []
        if len(temp_loss_list) > 0:
            loss_list.append(sum(temp_loss_list) / len(temp_loss_list))
            accuracy_list.append(sum(temp_accuracy_list) / len(temp_accuracy_list))
    return loss_list, accuracy_list

def ploting(data_list, title, x_label, y_label):
    """
    ploting data to graph
    """
    plt.plot(range(1, 88), data_list)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    plt.show()


if __name__ == "__main__":
    LOSS_LIST, ACC_LIST = loading_log('model_10.modellog')
    # print(len(LOSS_LIST), len(ACC_LIST))
    ploting(LOSS_LIST, 'training loss', 'step (per 1000 step)', 'loss')
    ploting(ACC_LIST, 'training accuracy', 'step (per 1000 step)', 'accuracy')
