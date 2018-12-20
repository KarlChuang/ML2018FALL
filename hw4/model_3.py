import torch
import torch.nn as nn
import torch.nn.functional as F

EMBEDDING_DIMENSION = 3087


class Model_3(nn.Module):
    """
    save the training model
    """

    def __init__(self, output_dimension):
        super(Model_3, self).__init__()
        self.output_dimension = output_dimension
        self.sequencial1 = nn.Linear(EMBEDDING_DIMENSION, 1000)
        self.sequencial2 = nn.Linear(1000, 500)
        self.sequencial3 = nn.Linear(500, 250)
        self.sequencial4 = nn.Linear(250, 100)
        self.sequencial5 = nn.Linear(100, output_dimension)

    def forward(self, word_vector):
        """
        forward path for LSTM
        """
        tag_space = F.relu(self.sequencial1(word_vector))
        tag_space = F.relu(self.sequencial2(tag_space))
        tag_space = F.relu(self.sequencial3(tag_space))
        tag_space = F.relu(self.sequencial4(tag_space))
        tag_space = self.sequencial5(tag_space)
        output = F.softmax(tag_space, dim=0)
        return output
