"""
RNN model training
"""
from os import path
from sys import argv

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

TRAIN_X_VECTOR_PATH = path.join('data', 'train_x_vector')
TRAIN_Y_VECTOR_PATH = path.join('data', 'train_y_vector.pt')
OUTPUT_MODEL_PATH = argv[1]
EMBEDDING_DIMENSION = 250

class LSTMModel(nn.Module):
    """
    save the training model
    """
    def __init__(self, hidden_dimension, output_dimension):
        super(LSTMModel, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(EMBEDDING_DIMENSION, hidden_dimension)
        self.sequencial1 = nn.Linear(hidden_dimension, 500)
        self.sequencial2 = nn.Linear(500, output_dimension)
#         self.sequencial3 = nn.Linear(100, output_dimension)

    def init_hidden(self):
        """
        initial the hidden value
        """
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dimension)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dimension)))

    def forward(self, word_vector):
        """
        forward path for LSTM
        """
        lstm_output, self.hidden = self.lstm(
            word_vector.view(len(word_vector), 1, -1), self.hidden)
        temp_idx = len(lstm_output)
#         while temp_idx < 3:
#             lstm_output = torch.cat(
#                 (lstm_output, torch.zeros(1, 1, self.hidden_dimension)), dim=0)
#             temp_idx += 1
#         tag_space = F.relu(self.sequencial1(lstm_output[-3:].view(1, -1)))
        tag_space = F.relu(self.sequencial1(lstm_output[-1:].view(1, -1)))
#         tag_space = F.relu(self.sequencial2(tag_space))
        tag_space = self.sequencial2(tag_space)
        output = F.softmax(tag_space[-1], dim=0)
        return output

def training(epochs, model, loss_function, optimizer):
    """
    training the model
    """
    for epoch in range(epochs):
        total_loss = 0
        total_correctness = 0
        predict_loss = [0]*1000
        predict_correctness = [0]*1000
        vector_x_list = []
        for path_i in range(12):
            vector_x_list_temp = torch.load(TRAIN_X_VECTOR_PATH + str(path_i) + '.pt')
            vector_x_list = vector_x_list + vector_x_list_temp
        vector_y_list = torch.load(TRAIN_Y_VECTOR_PATH)
        for idx, vector_list in enumerate(vector_x_list):
            print('\r%6s' % (idx + 1), end='')
            tag = torch.LongTensor(vector_y_list[idx])
            model.zero_grad()
            model.hidden = model.init_hidden()
            score = model(vector_list).view(1, -1)
            loss = loss_function(score, tag)
            loss.backward()
            optimizer.step()
            total_loss = total_loss - predict_loss[idx % 1000] + loss
            predict_loss[idx % 1000] = loss
            total_correctness -= predict_correctness[idx % 1000]
            predict_correctness[idx % 1000] = 0
            if ((score.data[0][0] > score.data[0][1] and tag == 0) or
                    (score.data[0][0] < score.data[0][1] and tag == 1)):
                total_correctness += 1
                predict_correctness[idx % 1000] = 1
            if idx > 999:
                print(' ave_loss=%.3f accuracy=%.3f' %
                      (total_loss / 1000, total_correctness / 1000), end='')
            if idx % 1000 == 999:
                torch.save(model.state_dict(), OUTPUT_MODEL_PATH)
        print()

if __name__ == "__main__":
    LSTM = LSTMModel(1000, 2)
    LSTM.load_state_dict(torch.load(OUTPUT_MODEL_PATH))
    LSTM.eval()
#     LSTM.cuda()
    training(
        epochs=10,
        model=LSTM,
        loss_function=nn.CrossEntropyLoss(),
        optimizer=optim.SGD(LSTM.parameters(), lr=0.001, momentum=0.9, weight_decay=0.000001))
