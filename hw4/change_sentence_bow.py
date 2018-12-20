"""
use bag of word to transfer the sentence to vector
"""

from os import path

import jieba
from gensim.models import word2vec
import torch

EMBEDDING_DIMENSION = 250
TRAIN_X_FILE_PATH = path.join('data', 'train_x.csv')
TESTING_X_FILE_PATH = path.join('data', 'test_x.csv')
TRAIN_Y_FILE_PATH = path.join('data', 'train_y.csv')
TEST_X_BOW_PATH = path.join('data', 'test_x.csv')
TRAINING_X_BOW_PATH = path.join('data', 'train_x_bow.pt')
TRAIN_Y_BOW_PATH = path.join('data', 'train_y_bow.pt')
TESTING_X_BOW_PATH = path.join('data', 'test_x_bow.pt')

jieba.load_userdict(path.join('data', 'dict.txt.big'))

WORD_BAG = {}

def segmentation(input_file_path, threshold=100):
    """
    cut the sentences to words
    """
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for idx, line in enumerate(input_file):
            if idx == 0:
                continue
            sentence = line.split(',')[1].split('\n')[0]
            seg_list = jieba.lcut(sentence)
            for word in seg_list:
                if word in WORD_BAG:
                    WORD_BAG[word] += 1
                else:
                    WORD_BAG[word] = 1
    del_key = []
    for key, element in WORD_BAG.items():
        if element < 100:
            del_key.append(key)
    for key in del_key:
        del WORD_BAG[key]

def transfer_sentence_to_vector(sentence):
    """
    transfer sentence to vector
    """
    seg_list = jieba.lcut(sentence)
    word_vector = []
    for key in WORD_BAG:
        if key in seg_list:
            word_vector.append(1)
        else:
            word_vector.append(0)
    return word_vector

def transfer_train_x_to_vector(train_x_file_path, train_y_file_path):
    """
    save training data as vector data
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
                vector = transfer_sentence_to_vector(sentence)
                temp_x_list.append(vector)
                y_list.append(data_y)
    torch.save(temp_x_list, TRAINING_X_BOW_PATH)
    torch.save(y_list, TRAIN_Y_BOW_PATH)

def transfer_testing_x_to_vector(test_x_file_path):
    """
    save training data as vector data
    """
    temp_x_list = []
    with open(test_x_file_path, 'r', encoding='utf-8') as file:
        for idx, line in enumerate(file):
            if idx == 0:
                continue
            sentence = line.split(',')[1].split('\n')[0]
            vector = transfer_sentence_to_vector(sentence)
            temp_x_list.append(vector)
    torch.save(temp_x_list, './data/a_bow.csv')

if __name__ == "__main__":
    segmentation(TRAIN_X_FILE_PATH)
    # transfer_train_x_to_vector(TRAIN_X_FILE_PATH, TRAIN_Y_FILE_PATH)
    # transfer_testing_x_to_vector(TESTING_X_FILE_PATH)
    transfer_testing_x_to_vector('./data/a.csv')
    print('dimension:', len(WORD_BAG))
