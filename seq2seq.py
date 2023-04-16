from collections import Counter
from typing import List
from base_model import BaseModel
from nltk.tokenize import word_tokenize
from torch.nn import LSTM, Embedding, Module, Linear, Softmax, CrossEntropyLoss
from torch.optim import RMSprop, Adam
import torch
import numpy as np
from gensim.models import KeyedVectors, Word2Vec
import random


class Encoder(Module):
    embedding_size: int
    hidden_size: int

    lstm: LSTM

    def __init__(self, embedding_size, hidden_size):
        super().__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.init_model()

    def init_model(self):
        self.lstm = LSTM(self.embedding_size, self.hidden_size, num_layers=1, dropout=.4)

    def forward(self, x, hidden=None):
        if hidden is not None:
            return self.lstm(x, hidden)
        else:
            return self.lstm(x)

    def init_hidden(self):
        return [torch.rand(1, self.hidden_size) for i in range(2)]


class Decoder(Module):
    vocab_size: int
    hidden_size: int

    lstm: LSTM
    linear: Linear
    activation: Softmax

    def __init__(self, vocab_size, hidden_size):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.init_model()

    def forward(self, x, hidden):
        cn, hn = hidden

        lstm_out, hidden = self.lstm(x, (cn, hn))
        linear_out = self.linear(lstm_out[0])
        return self.activation(linear_out), hidden

    def init_model(self):
        self.lstm = LSTM(1, self.hidden_size, dropout=.4)
        self.linear = Linear(self.hidden_size, self.vocab_size)
        self.activation = Softmax(dim=0)


class Seq2Seq(Module, BaseModel):
    learning_rate: float

    embedding_size: int
    vocab_size: int
    hidden_size: int
    max_length: int = 50

    encoder: Encoder
    decoder: Decoder
    vector_model: Word2Vec

    def __init__(self, embedding_size, hidden_size, vector_model, learning_rate=.01):
        super().__init__()
        self.learning_rate = learning_rate

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vector_model = vector_model
        self.vocab_size = len(vector_model.wv.index2word) + 2

        self.init_model()

    def init_model(self):
        self.encoder = Encoder(self.embedding_size, self.hidden_size)
        self.decoder = Decoder(self.vocab_size, self.hidden_size)

    def sentence2tensor(self, sentence: List[str]):
        vector = [self.vector_model.wv[word] for word in sentence if word in self.vector_model.wv]
        return torch.tensor(np.array(vector), dtype=torch.float)

    def train_model(self, train_data: List[List[str]], references: List[List[str]], epochs: int = 10) -> List[float]:
        loss_criteria = CrossEntropyLoss()
        encoder_optimizer = Adam(self.encoder.parameters(), lr=self.learning_rate)
        decoder_optimizer = Adam(self.decoder.parameters(), lr=self.learning_rate)

        epoch_losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            training_set = list(zip(train_data, references))
            random.shuffle(training_set)
            for article, reference in training_set:
                sentence = []
                loss = 0
                encoder_hidden = self.encoder.init_hidden()

                self.zero_grad()
                self.encoder.zero_grad()
                self.decoder.zero_grad()

                inputs = self.sentence2tensor(article)
                targets = self.sentence2tensor(reference)

                encoder_outputs, encoder_hidden = self.encoder(inputs, encoder_hidden)

                decoder_input = torch.tensor([[0]], dtype=torch.float)
                decoder_hidden = encoder_hidden

                for i in range(targets.shape[0]):
                    decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                    topv, topi = decoder_output.topk(10)
                    choice = random.randint(0, 9)
                    decoder_input = torch.tensor([[topi[choice].squeeze().detach()]], dtype=torch.float)

                    if decoder_input.item() == 1 or decoder_input.item() == 0:
                        break

                    word = self.vector_model.wv.index2word[int(topi[choice]) - 2]
                    if word not in self.vector_model.wv:
                        word = "<UNK>"
                        output_tensor = torch.zeros(self.embedding_size, dtype=torch.float)
                    else:
                        output_tensor = torch.tensor(self.vector_model.wv[word])

                    sentence.append(word)

                    loss += loss_criteria(output_tensor, targets[i])

                epoch_loss += float(loss)
                loss = loss.clone().detach().requires_grad_(True)
                loss.backward()
                encoder_optimizer.step()
                decoder_optimizer.step()

            epoch_loss /= len(train_data)
            epoch_losses.append(epoch_loss)
            print(f"Epoch {epoch} Loss: {epoch_loss}")
            print(f"Last training example: \n")
            print(f"Article: {' '.join(article)}")
            print(f"Reference: {' '.join(reference)}")
            print(f"Summary: {' '.join(sentence)}\n\n\n")

        return epoch_losses

    def predict(self, data: List[str]) -> List[List[str]]:
        output_sentences = []

        for article in data:
            sentence = []
            inputs = self.sentence2tensor(article)

            encoder_outputs, encoder_hidden = self.encoder(inputs)

            decoder_input = torch.tensor([[0]], dtype=torch.float)
            decoder_hidden = encoder_hidden

            for i in range(self.max_length):
                decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(10)
                choice = random.randint(0, 9)
                decoder_input = torch.tensor([[topi[choice].squeeze().detach()]], dtype=torch.float)

                word = self.vector_model.wv.index2word[int(topi[choice])]
                if word not in self.vector_model.wv:
                    word = "<UNK>"
                sentence.append(word)

                if decoder_input.item() == 1:
                    break

            output_sentences.append(sentence)

        return output_sentences

    def save(self):
        torch.save(self.encoder.state_dict(), "encoder.pt")
        torch.save(self.decoder.state_dict(), "decoder.pt")
