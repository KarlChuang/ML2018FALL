"""
Use to draw confusion matrix
"""
from sys import argv
import itertools

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import Sequential, load_model

TESTING_FILE_PATH = argv[1]

class Data():
    """
    Class to read data and create it as an array
    """

    def __init__(self):
        self.data_matrix = np.array([])
        self.predict_y = np.array([])
        self.data_y = np.array([])
        self.predict_label = np.array([])

    def get_data_matrix(self):
        """
        get data matrix create from data
        """
        return self.data_matrix

    def get_predict_y(self):
        """
        get predict y
        """
        return self.predict_y

    def set_predict_y(self, predict_data_y):
        """
        set the predict_y
        """
        self.predict_y = np.argmax(predict_data_y, axis=1)
        self.predict_label = predict_data_y

    def read_data(self, file_path):
        """
        read data from train.csv file
        """
        data = []
        data_y = []
        # open train_x.csv
        with open(file_path, 'rb') as file:
            for idx, line in enumerate(file):
                if (idx == 0):
                    continue
                if (idx == 5600):
                    break
                row = str(line).split("'")[1].split('\\n')[0]
                items = row.split(',')
                temp_y = []
                for idx2 in range(7):
                    if idx2 == int(items[0]):
                        temp_y.append(1)
                    else:
                        temp_y.append(0)
                data_y.append(temp_y)
                temp_x = items[1].split(' ')
                temp_x_3d = [[[float(y) / 255] for y in temp_x[x:x + 48]]
                             for x in range(0, len(temp_x), 48)]
                data.append(temp_x_3d)
        self.data_matrix = np.array(data)
        self.data_y = np.array(data_y)

    def get_confision_matrix(self):
        """
        ploting confision matrix
        """
        real_y = np.argmax(self.data_y, axis=1)
        return confusion_matrix(real_y, self.predict_y)

class Model():
    """
    Use to load model
    """

    def __init__(self):
        self.model = Sequential()

    def predict(self, data):
        """
        predict result from data
        """
        testing_data_x = data.get_data_matrix()
        return self.model.predict(testing_data_x)

    def load_model(self, model_file_path):
        """
        load model from outside file
        """
        self.model = load_model(model_file_path)


def plot_confusion_matrix(cm,
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


if __name__ == '__main__':
    MODEL = Model()
    MODEL.load_model(model_file_path='./model/model_8.hs')
    DATA = Data()
    DATA.read_data(file_path=TESTING_FILE_PATH)
    DATA.set_predict_y(predict_data_y=MODEL.predict(DATA))
    # DATA.write_file(OUTPUT_FILE_PATH)
    CONFUSION_MATIRX = DATA.get_confision_matrix()
    plot_confusion_matrix(CONFUSION_MATIRX, classes='aaa', normalize=True)
    plt.show()
