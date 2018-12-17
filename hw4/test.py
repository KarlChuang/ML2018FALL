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
from gensim.models import word2vec
import jieba

MODEL_PATH = path.join('model', 'model_4.model')
TESTING_FILE_PATH = argv[1]
OUTPUT_FILE_PATH = argv[2]
EMBEDDING_MODEL_PATH = path.join('model', 'embedding.model')
EMBEDDING_MODEL = word2vec.Word2Vec.load(EMBEDDING_MODEL_PATH)
EMBEDDING_DIMENSION = 250

cuda = torch.device('cuda')


from train3 import LSTMModel


# class LSTMModel(nn.Module):
#     """
#     save the training model
#     """

#     def __init__(self, hidden_dimension, output_dimension):
#         super(LSTMModel, self).__init__()
#         self.hidden_dimension = hidden_dimension
#         self.output_dimension = output_dimension
#         self.hidden = self.init_hidden()
#         self.lstm = nn.LSTM(EMBEDDING_DIMENSION, hidden_dimension)
#         self.sequencial1 = nn.Linear(hidden_dimension, 50)
#         self.sequencial2 = nn.Linear(50, 25)
#         self.sequencial3 = nn.Linear(25, 10)
#         self.sequencial4 = nn.Linear(10, output_dimension)

#     def init_hidden(self):
#         """
#         initial the hidden value
#         """
#         return (autograd.Variable(torch.zeros(1, 1, self.hidden_dimension)),
#                 autograd.Variable(torch.zeros(1, 1, self.hidden_dimension)))

#     def forward(self, word_vector):
#         """
#         forward path for LSTM
#         """
#         lstm_output, self.hidden = self.lstm(
#             word_vector.view(len(word_vector), 1, -1), self.hidden)
#         tag_space = self.sequencial1(lstm_output.view(len(word_vector), -1))
#         tag_space = self.sequencial2(tag_space)
#         tag_space = self.sequencial3(tag_space)
#         tag_space = self.sequencial4(tag_space)
#         # print(tag_space[-1])
#         output = F.softmax(tag_space[-1], dim=0)
#         # print(output)
#         # exit()
#         return output


def data_x_generator(file_path):
    """
    use to generate iteration data
    """
    with open(file_path, 'r') as file:
        for idx, line in enumerate(file):
            if idx == 0:
                continue
            idd = line.split(',')[0]
            sentence = line.split(',')[1].split('\n')[0]
            yield (idd, sentence)

def embedding(sentence):
    """
    word embedding
    """
    seg_list = jieba.lcut(sentence, cut_all=True)
    vector_list = []
    for word in seg_list:
        try:
            vector_list.append(EMBEDDING_MODEL[word])
        except KeyError:
            # print(word + ' not in vocabulary')
            pass
    return torch.FloatTensor(vector_list)

def predict(model, testing_file_path, output_file_path):
    generator = data_x_generator(testing_file_path)
    with open(output_file_path, 'w') as output_file:
        output_file.write('id,label\n')
        for (idd, sentence) in generator:
            output_file.write(str(idd) + ',')
            model.zero_grad()
            model.hidden = model.init_hidden()
            word_vector = embedding(sentence)
            if len(word_vector) == 0:
                output_file.write(str(0) + '\n')
            else:
                score = model(word_vector).view(1, -1)
                if score[0][0] > score[0][1]:
                    output_file.write('0\n')
                else:
                    output_file.write('1\n')

if __name__ == "__main__":
    LSTM = LSTMModel(1000, 2)
    LSTM.load_state_dict(torch.load(MODEL_PATH))
    LSTM.eval()
    predict(LSTM, TESTING_FILE_PATH, OUTPUT_FILE_PATH)
