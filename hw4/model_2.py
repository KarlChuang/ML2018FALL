import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIMENSION = 250

class Model_2(nn.Module):
    """
    save the training model
    """

    def __init__(self, hidden_dimension, output_dimension):
        super(Model_2, self).__init__()
        self.hidden_dimension = hidden_dimension
        self.output_dimension = output_dimension
        self.lstm = nn.LSTM(EMBEDDING_DIMENSION, hidden_dimension)
        self.sequencial1 = nn.Linear(hidden_dimension, output_dimension)

    def forward(self, word_vector):
        """
        forward path for LSTM
        """
        lstm_output, _ = self.lstm(word_vector)
        temp_idx = len(lstm_output)
        tag_space = self.sequencial1(lstm_output[-1])
        output = F.softmax(tag_space, dim=0)
        return output
