"""
deal with chinese sentence with segmentation and embedding
"""
from os import path

import jieba
from gensim.models import word2vec
import torch

EMBEDDING_DIMENSION = 250
TRAIN_X_FILE_PATH = path.join('data', 'train_x.csv')
TRAIN_Y_FILE_PATH = path.join('data', 'train_y.csv')
TRAIN_Y_VECTOR_PATH = path.join('data', 'train_y_vector.pt')
TEST_X_FILE_PATH = path.join('data', 'test_x.csv')
SEGMENTATION_FILE_PATH = path.join('data', 'segmentation.txt')
EMBEDDING_FILE_PATH = path.join('model', 'embedding' + str(EMBEDDING_DIMENSION) + '.model')

def get_train_x_vector_path(num):
    """
    use for cutting train_x file
    """
    return path.join('data', 'train_x_vector', 'part%s.pt' % (int(num)))

def get_test_x_vector_path(num):
    """
    use for cutting test_x file
    """
    return path.join('data', 'test_x_vector', 'part%s.pt' % (int(num)))


jieba.load_userdict(path.join('data', 'dict.txt.big'))

def segmentation(input_file_path, output_file_path):
    """
    cut the sentences to words
    """
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        with open(input_file_path, 'r', encoding='utf-8') as input_file:
            for idx, line in enumerate(input_file):
                if idx == 0:
                    continue
                sentence = line.split(',')[1].split('\n')[0]
                seg_list = jieba.lcut(sentence)
                output_file.write(' '.join(seg_list) + '\n')

def embedding(input_file_path, output_file_path):
    """
    training word embedding model
    """
    sentences = word2vec.LineSentence(input_file_path)
    model = word2vec.Word2Vec(sentences, size=250)
    model.save(output_file_path)


def transfer_train_x_to_vector(embedding_model, train_x_file_path, train_y_file_path, batch_size):
    """
    cut training data to small data and save as vector
    """
    with open(train_x_file_path, 'r', encoding='utf-8') as file:
        with open(train_y_file_path, 'r') as file_y:
            temp_x_list = []
            y_list = []
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
                        vector_list.append(torch.FloatTensor(embedding_model[word]))
                    except KeyError:
                        pass
                if len(vector_list) > 0:
                    temp_x_list.append(vector_list)
                    y_list.append(data_y)
                if idx % batch_size == 0:
                    torch.save(temp_x_list, get_train_x_vector_path(idx / batch_size))
                    temp_x_list = []
                    print('\rProgress=%s' % int(idx / batch_size), end='')
            if len(temp_x_list) > 0:
                torch.save(temp_x_list, get_train_x_vector_path(idx / batch_size))
                print('\rProgress=%s' % int(idx / batch_size), end='')
            torch.save(y_list, TRAIN_Y_VECTOR_PATH)
            print()

def transfer_test_to_vector(embedding_model, test_x_file_path, batch_size):
    """
    cut testing data to small data and save as vector
    """
    with open(test_x_file_path, 'r', encoding='utf-8') as file:
        temp_x_list = []
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
            temp_x_list.append(vector_list)
            if idx % batch_size == 0:
                torch.save(temp_x_list,
                        get_test_x_vector_path(idx / batch_size))
                temp_x_list = []
                print('\rProgress=%s' % int(idx / batch_size), end='')
        if len(temp_x_list) > 0:
            torch.save(temp_x_list, get_test_x_vector_path(idx / batch_size))
            print('\rProgress=%s' % int(idx / batch_size), end='')
        print()

if __name__ == "__main__":
    # segmentation(TRAIN_X_FILE_PATH, SEGMENTATION_FILE_PATH)
    # embedding(SEGMENTATION_FILE_PATH, EMBEDDING_FILE_PATH)
    EMBEDDING_MODEL = word2vec.Word2Vec.load(EMBEDDING_FILE_PATH)
    transfer_train_x_to_vector(EMBEDDING_MODEL, TRAIN_X_FILE_PATH,
                               TRAIN_Y_FILE_PATH, 1000)
    # transfer_test_to_vector(EMBEDDING_MODEL, TEST_X_FILE_PATH, 1000)
