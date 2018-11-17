"""
testing file
"""
from sys import argv

import numpy as np
from keras.models import Sequential, load_model

TESTING_FILE_PATH = argv[1]
OUTPUT_FILE_PATH = argv[2]

class Data():
    """
    Class to read data and create it as an array
    """

    def __init__(self):
        self.data_matrix = np.array([])
        self.data_id = np.array([])
        self.predict_y = np.array([])

    def get_data_matrix(self):
        """
        get data matrix create from data
        """
        return self.data_matrix

    def get_data_id_array(self):
        """
        get data id array
        """
        return self.data_id

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

    def read_data(self, file_path):
        """
        read data from train.csv file
        """
        data = []
        data_id = []
        # open train_x.csv
        with open(file_path, 'rb') as file:
            for idx, line in enumerate(file):
                if (idx == 0):
                    continue
                row = str(line).split("'")[1].split('\\n')[0]
                items = row.split(',')
                data_id.append(items[0])
                temp_x = items[1].split(' ')
                temp_x_3d = [[[float(y) / 255] for y in temp_x[x:x + 48]]
                             for x in range(0, len(temp_x), 48)]
                data.append(temp_x_3d)
        self.data_matrix = np.array(data)
        self.data_id = np.array(data_id)

    def write_file(self, output_file_path):
        """
        write predict data to output file
        """
        with open(output_file_path, 'w') as file:
            file.write('id,label\r\n')
            for idx1, id_value in enumerate(self.data_id):
                file.write(id_value + ',' + str(self.predict_y[idx1]) + '\r\n')

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

if __name__ == '__main__':
    MODEL = Model()
    MODEL.load_model(model_file_path='./model/model_6.hs')
    DATA = Data()
    DATA.read_data(file_path=TESTING_FILE_PATH)
    DATA.set_predict_y(predict_data_y=MODEL.predict(DATA))
    DATA.write_file(OUTPUT_FILE_PATH)
