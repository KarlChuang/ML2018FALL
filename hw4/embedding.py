"""
deal with chinese sentence with segmentation and embedding
"""
from os import path

import jieba
from gensim.models import word2vec

TRAIN_X_FILE_PATH = path.join('data', 'train_x.csv')
SEGMENTATION_FILE_PATH = path.join('data', 'segmentation.txt')
EMBEDDING_FILE_PATH = path.join('model', 'embedding.model')

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

def testing_model(model_path):
    """
    testing the embedding model
    """
    word = ''
    with open(SEGMENTATION_FILE_PATH, 'r', encoding='utf-8') as input_file:
        for idx, line in enumerate(input_file):
            if idx == 10:
                word = line.split(' ')[0]
    model = word2vec.Word2Vec.load(model_path)
    res = model.most_similar(word, topn=100)
    print(word)
    for item in res:
        print(item[0]+","+str(item[1]))

if __name__ == "__main__":
    # segmentation(TRAIN_X_FILE_PATH, SEGMENTATION_FILE_PATH)
    embedding(SEGMENTATION_FILE_PATH, EMBEDDING_FILE_PATH)
    # testing_model(EMBEDDING_FILE_PATH)
