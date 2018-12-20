"""
Testing
"""
from os import path
from sys import argv

import jieba
from gensim.models import word2vec
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model_1 import Model_1

MODEL_PATH = argv[1]
TESTING_FILE_PATH = argv[2]
DICT_TXT_BIG = argv[3]
OUTPUT_FILE_PATH = argv[4]
EMBEDDING_DIMENSION = 250
EMBEDDING_FILE_PATH = path.join(
    'model', 'embedding' + str(EMBEDDING_DIMENSION) + '.model')
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')
EMBEDDING_MODEL = word2vec.Word2Vec.load(EMBEDDING_FILE_PATH)

def transfer_test_to_vector(embedding_model, test_x_file_path):
    """
    cut testing data to small data and save as vector
    """
    temp_x_list = []
    with open(test_x_file_path, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            if idx == 0:
                continue
            sentence = line.split(',')[1].split('\n')[0]
            seg_list = jieba.lcut(sentence)
            vector_list = []
            for word in seg_list:
                try:
                    vector_list.append(
                        torch.FloatTensor(embedding_model[word]))
                except KeyError:
                    pass
            if len(vector_list) == 0:
                vector_list.append(torch.zeros(EMBEDDING_DIMENSION))
            temp_x_list.append(vector_list)
    return temp_x_list

def input_fitting(vector_x_batch):
    """
    fit the LSTM input
    """
    longest_sent = max([len(v) for v in vector_x_batch])
    temp_x_batch = [
        (torch.cat((torch.stack(v), torch.zeros(longest_sent - len(v), EMBEDDING_DIMENSION)))
         if len(v)> 0 else torch.zeros(longest_sent - len(v), EMBEDDING_DIMENSION))
        for v in vector_x_batch
    ]
    x_tensor = torch.stack(temp_x_batch, 0)
    x_tensor_t = torch.transpose(x_tensor, 0, 1)
    if USE_GPU:
        return x_tensor_t.cuda()
    return x_tensor_t

def predict(model, output_file_path, batches):
    vector_x_list = transfer_test_to_vector(EMBEDDING_MODEL, TESTING_FILE_PATH)
    with open(output_file_path, 'w') as output_file:
        output_file.write('id,label\n')
        idx = 0
        while idx + batches <= len(vector_x_list):
            vector_x_batch = vector_x_list[idx:idx + batches]
            x_tensor = input_fitting(vector_x_batch)
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
            x_tensor = input_fitting(vector_x_batch)
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
    #     LSTM = Model_0(1000, 2)
    LSTM = Model_1(1000, 2)
    if USE_GPU:
        LSTM.load_state_dict(torch.load(MODEL_PATH))
    else:
        LSTM.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    LSTM.eval()
    if USE_GPU:
        LSTM.cuda()
    predict(LSTM, OUTPUT_FILE_PATH, 50)
