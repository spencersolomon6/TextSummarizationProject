import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim.downloader
from gensim.models import Word2Vec
import os


MODEL_NAME = "pretrained.model"


def preprocess_data(data: pd.Series, vector_model: Word2Vec):
    """
    Transforms a Series of data into a list of list of word vectors for each document

    :param data:
    :return: list of list of vectors
    """

    vectors = []
    for i, text in data.iteritems():
        tokenized_text = [word for word in word_tokenize(text) if word not in stopwords]

        vectors.append([vector_model.wv[word] for word in tokenized_text])

    return vectors


if __name__ == '__main__':
    train = pd.read_csv("data/train.csv")
    # test = pd.read_csv("data/test.csv")
    # validation = pd.read_csv("data/validation.csv")

    if os.path.exists(MODEL_NAME):
        vector_model = Word2Vec.load(MODEL_NAME)
    else:
        vector_model = gensim.downloader.load('fasttext-wiki-news-subwords-300')
        vector_model.save(MODEL_NAME)





    print(preprocess_data(train.head()["article"], vector_model))
