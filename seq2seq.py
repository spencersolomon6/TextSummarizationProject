from typing import List
from base_model import BaseModel
from nltk.tokenize import word_tokenize
from torch.nn import LSTM, Embedding, Module, Linear, Softmax, CrossEntropyLoss
from torch.optim import RMSprop
import torch
import numpy as np


class Seq2Seq(Module, BaseModel):
    learning_rate: float

    embedding_size: int
    vocab_size: int
    hidden_size: int
    vocabulary: list

    enc_embedding: Embedding
    enc_lstm1: LSTM
    enc_lstm2: LSTM
    enc_lstm3: LSTM
    dec_embedding: Embedding
    dec_lstm: LSTM
    fc_linear: Linear
    output_layer: Softmax

    def __init__(self, embedding_size, hidden_size, vocabulary, learning_rate=.01):
        super().__init__()
        self.learning_rate = learning_rate

        self.embedding_size = embedding_size
        self.vocab_size = len(vocabulary)
        self.hidden_size = hidden_size
        self.vocabulary = vocabulary

        self.init_model()

    def init_model(self):
        self.enc_embedding = Embedding(self.vocab_size, self.embedding_size)

        self.enc_lstm1 = LSTM(self.embedding_size, self.hidden_size, dropout=.4)
        self.enc_lstm2 = LSTM(self.hidden_size, self.hidden_size, dropout=.4)
        self.enc_lstm3 = LSTM(self.hidden_size, self.hidden_size, dropout=.4)

        self.dec_embedding = Embedding(self.vocab_size, self.embedding_size)

        self.dec_lstm = LSTM(self.embedding_size, self.hidden_size, dropout=.4)
        self.fc_linear = Linear(self.hidden_size, self.vocab_size)
        self.output_layer = Softmax()

    def forward(self, article):
        enc_embeddings = self.enc_embedding(article)

        enc_lstm1_out, _ = self.enc_lstm1(enc_embeddings)
        enc_lstm2_out, _ = self.enc_lstm2(enc_lstm1_out)
        enc_lstm3_out, (hn, cn) = self.enc_lstm3(enc_lstm2_out)

        dec_embeddings = self.dec_embedding(None)
        dec_lstm_out, _ = self.dec_lstm(dec_embeddings, (hn, cn))
        fc_linear_out = self.fc_linear(dec_lstm_out)
        output_layer_out = self.output_layer(fc_linear_out)

        return output_layer_out

    def sentence2tensor(self, sentence: str):
        indexes = [self.vocabulary.index(word) for word in word_tokenize(sentence)]
        return torch.tensor(indexes, dtype=torch.long)

    def tensor2sentence(self, tensor: torch.Tensor):
        predictions = []
        for i, row in enumerate(tensor):
            predictions.append(self.vocabulary[np.argmax(row)])

        return ' '.join(predictions)

    def train_model(self, train_data: List[str], references: List[str]) -> None:
        loss = CrossEntropyLoss()
        optimizer = RMSprop(self.parameters(), lr=self.learning_rate)

    def predict(self, data: List[str]) -> List[str]:
        predictions = []
        for sentence in data:
            with torch.no_grad():
                inputs = self.sentence2tensor(sentence)
                scores = self(inputs)
                predictions.append(self.tensor2sentence(scores))

        return predictions
