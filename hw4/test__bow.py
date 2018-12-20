"""
Testing
"""
from os import path
from sys import argv

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model_3 import Model_3

MODEL_PATH = argv[1]
TESTING_FILE_PATH = argv[2]
OUTPUT_FILE_PATH = argv[3]
EMBEDDING_DIMENSION = 250
TESTING_FILE_NUMBER = 80
TEST_X_VECTOR_PATH = path.join('data', 'test_x_bow.pt', )
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')

from model_0 import Model_0
from model_1 import Model_1

# def get_test_x_vector_path(num):
#     """
#     use for cutting test_x file
#     """
#     return path.join('data', 'test_x_vector', 'part%s.pt' % (int(num)))


# def input_fitting(vector_x_batch):
#     """
#     fit the LSTM input
#     """
#     longest_sent = max([len(v) for v in vector_x_batch])
#     temp_x_batch = [
#         (torch.cat((torch.stack(v), torch.zeros(longest_sent - len(v), EMBEDDING_DIMENSION)))
#          if len(v)> 0 else torch.zeros(longest_sent - len(v), EMBEDDING_DIMENSION))
#         for v in vector_x_batch
#     ]
#     x_tensor = torch.stack(temp_x_batch, 0)
#     x_tensor_t = torch.transpose(x_tensor, 0, 1)
#     if USE_GPU:
#         return x_tensor_t.cuda()
#     return x_tensor_t

def predict(model, output_file_path, batches):
    vector_x_list = torch.load('./data/a_bow.csv')
    # vector_x_list = torch.load(TEST_X_VECTOR_PATH)
    # vector_x_list = []
    # for path_idx in range(TESTING_FILE_NUMBER):
    #     file_path = get_test_x_vector_path(path_idx + 1)
    #     if USE_GPU:
    #         vector_x_list_temp = torch.load(file_path)
    #     else:
    #         vector_x_list_temp = torch.load(file_path, map_location='cpu')
    #     vector_x_list = vector_x_list + vector_x_list_temp
    #     print('\rConcating testing file...', path_idx + 1, 'success', end='')
    # print()
    with open(output_file_path, 'w') as output_file:
        output_file.write('id,label\n')
        idx = 0
        while idx + batches <= len(vector_x_list):
            vector_x_batch = vector_x_list[idx:idx + batches]
            if USE_GPU:
                x_tensor = torch.FloatTensor(vector_x_batch).cuda()
                # y_tensor = torch.tensor(vector_y_batch).cuda()
            else:
                x_tensor = torch.FloatTensor(vector_x_batch)
                # y_tensor = torch.tensor(vector_y_batch)
            # x_tensor = input_fitting(vector_x_batch)
            model.zero_grad()
            scores = model(x_tensor)
            for idx2, score in enumerate(scores):
                output_file.write(str(idx + idx2) + ',')
                if score[0] > score[1]:
                    output_file.write('0\n')
                else:
                    output_file.write('1\n')
            idx += batches
            print('\rProgress: %s%%' % (int(100 * idx / len(vector_x_list))), end='')
        if idx < len(vector_x_list):
            vector_x_batch = vector_x_list[idx:]
            x_tensor = torch.FloatTensor(vector_x_batch)
            model.zero_grad()
            scores = model(x_tensor)
            for idx2, score in enumerate(scores):
                output_file.write(str(idx + idx2) + ',')
                if score[0] > score[1]:
                    output_file.write('0\n')
                else:
                    output_file.write('1\n')
        print()

if __name__ == "__main__":
    LSTM = Model_3(2)
    # LSTM = Model_1(1000, 2)
    if USE_GPU:
        LSTM.load_state_dict(torch.load(MODEL_PATH))
    else:
        LSTM.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    LSTM.eval()
    if USE_GPU:
        LSTM.cuda()
    predict(LSTM, OUTPUT_FILE_PATH, 50)
