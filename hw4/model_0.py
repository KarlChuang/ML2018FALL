"""
testing result is output_8.csv
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIMENSION = 250


class Model_0(nn.Module):
    """
    save the training model
    """

    def __init__(self, hidden_dimension, output_dimension):
        super(LSTMModel, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        #         self.hidden = self.init_hidden()
        self.lstm = nn.LSTM(EMBEDDING_DIMENSION, hidden_dimension)
        self.sequencial1 = nn.Linear(hidden_dimension, 500)
        self.sequencial2 = nn.Linear(500, 250)
        self.sequencial3 = nn.Linear(250, output_dimension)

    def forward(self, word_vector):
        """
        forward path for LSTM
        """
        lstm_output, _ = self.lstm(word_vector)
        temp_idx = len(lstm_output)
        tag_space = F.relu(self.sequencial1(lstm_output[-1]))
        tag_space = F.relu(self.sequencial2(tag_space))
        tag_space = self.sequencial3(tag_space)
        output = F.softmax(tag_space, dim=0)
        return output
