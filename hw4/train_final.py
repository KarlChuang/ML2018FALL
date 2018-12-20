"""
RNN model training
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

# from model_0 import Model_0
from model_1 import Model_1
# from model_2 import Model_2

TRAIN_X_FILE_PATH = argv[1]
TRAIN_Y_FILE_PATH = argv[2]
DICT_TXT_BIG = argv[3]
OUTPUT_MODEL_PATH = argv[4]
EMBEDDING_DIMENSION = 250
EMBEDDING_FILE_PATH = path.join(
    'model', 'embedding' + str(EMBEDDING_DIMENSION) + '.model')
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_GPU else 'cpu')

jieba.load_userdict(DICT_TXT_BIG)
EMBEDDING_MODEL = word2vec.Word2Vec.load(EMBEDDING_FILE_PATH)

def transfer_train_x_to_vector(embedding_model, train_x_file_path,
                               train_y_file_path):
    """
    cut training data to small data and save as vector
    """
    temp_x_list = []
    y_list = []
    with open(train_x_file_path, 'r', encoding='utf-8') as file:
        with open(train_y_file_path, 'r') as file_y:
            for idx, line in enumerate(file):
                data_y = next(file_y)
                if idx == 0:
                    continue
                data_y = int(data_y.split(',')[1].split('\n')[0])
                sentence = line.split(',')[1].split('\n')[0]
                seg_list = jieba.lcut(sentence)
                vector_list = []
                for word in seg_list:
                    try:
                        vector_list.append(
                            torch.FloatTensor(embedding_model[word]))
                    except KeyError:
                        pass
                if len(vector_list) > 0:
                    temp_x_list.append(vector_list)
                    y_list.append(data_y)
    return (temp_x_list, y_list)

def input_fitting(vector_x_batch, vector_y_batch):
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
    y_tensor = torch.tensor(vector_y_batch)
    if USE_GPU:
        return x_tensor_t.cuda(), y_tensor.cuda()
    return x_tensor_t, y_tensor

def training(epochs, batches, model, loss_function, optimizer):
    """
    training the model
    """
    vector_x_list, vector_y_list = transfer_train_x_to_vector(
        EMBEDDING_MODEL, TRAIN_X_FILE_PATH, TRAIN_Y_FILE_PATH)
    loss_history = []
    accuracy_history = []
    for epoch in range(epochs):
        model.train()
        idx = 0
        while idx + batches <= len(vector_x_list):
            vector_x_batch = vector_x_list[idx:idx + batches]
            vector_y_batch = vector_y_list[idx:idx + batches]
            x_tensor, y_tensor = input_fitting(vector_x_batch, vector_y_batch)

            optimizer.zero_grad()
            model.zero_grad()
            score = model(x_tensor)
            loss = loss_function(score, y_tensor)
            accuracy = get_accuracy(score, y_tensor)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.data[0])
            accuracy_history.append(accuracy)
            if idx % 500 == 0 and len(loss_history) > 0:
                temp_loss = sum(loss_history) / len(loss_history)
                temp_accuracy = sum(accuracy_history) / len(accuracy_history)
                print(
                    '\rTrain epoch: {} ({:2.0f}%)\tLoss: {:.4f}\tAccuracy: {:.2f}'.format(
                        epoch + 1, 100. * (idx + 1) / len(vector_x_list), temp_loss, temp_accuracy),
                    end='')
                loss_history = []
                accuracy_history = []
            idx += batches
            if idx % 10000 == 0:
                torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
        print()

def get_accuracy(scores, labels):
    """
    get batch accuracy
    """
    correct_num = 0
    for idx, label in enumerate(labels):
        if scores[idx][0] > scores[idx][1] and label == 0:
            correct_num += 1
        if scores[idx][1] > scores[idx][0] and label == 1:
            correct_num += 1
    return correct_num / len(scores)

if __name__ == "__main__":
    LSTM = Model_1(1000, 2)
    # LSTM.load_state_dict(torch.load(OUTPUT_MODEL_PATH))
    # LSTM.eval()
    if USE_GPU:
        print('Use gpu')
        LSTM.cuda()
    training(
        epochs=10,
        batches=10,
        model=LSTM,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optim.Adam(LSTM.parameters(), lr=0.001))
#         optimizer=optim.SGD(LSTM.parameters(), lr=0.000000001, momentum=0.9, weight_decay=0.00001))
