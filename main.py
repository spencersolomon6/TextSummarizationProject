import os

import gensim.downloader
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.summarization.summarizer import summarize
from rouge_score.rouge_scorer import RougeScorer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
from seq2seq import Seq2Seq
from collections import Counter

MODEL_NAME = "pretrained.model"


def preprocess_data(data: pd.Series, vector_model: KeyedVectors):
    """
    Transforms a Series of data into a list of list of word vectors for each document

    :param data:
    :return: list of list of vectors
    """

    vectors = []
    for i, text in data.iteritems():
        tokenized_text = [word for word in word_tokenize(text)]

        vectors.append([vector_model[word] if word in vector_model.key_to_index else np.zeros(300) for word in tokenized_text])

    return vectors


def rogue_score(predictions, references, scorer: RougeScorer):
    """
    Returns a DataFrame of scores for each prediction/reference pair

    :param predictions:
    :param references:
    :param scorer:
    :return: A DataFrame containing the prediction, reference and scores for each example
    """

    scores = pd.DataFrame(columns=['prediction', 'reference', 'precision', 'recall', 'fmeasure'])
    for predicted, reference in zip(predictions, references):
        score = scorer.score(predicted, reference)['rouge2']
        score_dict = {'prediction': predicted, 'reference': reference, 'precision': score.precision,
                      'recall': score.recall, 'fmeasure': score.fmeasure}

        scores = scores.append(score_dict, ignore_index=True)

    return scores


def bleu_score(predictions, references):

    scores = pd.DataFrame(columns=['prediction', 'reference', 'bleu_score'])
    for prediction, reference in zip(predictions, references):
        tok_prediction = word_tokenize(prediction)
        tok_reference = word_tokenize(reference)

        score = sentence_bleu(tok_reference, tok_prediction)
        score_dict = {"prediction": prediction, "reference": reference, "bleu_score": score}
        scores = scores.append(score_dict, ignore_index=True)

    return scores


def perform_baseline(test_data: pd.DataFrame, scorer: RougeScorer):
    summaries = []
    references = []
    for i, row in test_data.iterrows():
        summaries.append(summarize(row['article'].replace('.', '\n').replace('?', '\n').replace('!', '\n')))
        references.append(row['highlights'])

    print(summaries)

    rouge_scores = rogue_score(summaries, references, scorer)
    bleu_scores = bleu_score(summaries, references)

    scores = pd.merge(rouge_scores, bleu_scores, on=['prediction', 'reference'])

    return scores


def plot_scores(scores):
    plt.figure(1)
    plt.plot(range(len(scores)), scores['bleu_score'], 'b')
    plt.title("Bleu Scores")
    plt.ylabel("Score")
    plt.xlabel("Example #")
    plt.show()

    plt.figure(2)
    plt.plot(range(len(scores)), scores['fmeasure'], 'g', label='F1 Score')
    # plt.plot(range(len(scores)), scores['recall'], 'r--', label='Recall')
    # plt.plot(range(len(scores)), scores['precision'], 'y--', label='Precision')
    plt.title("ROUGE-2 Scores")
    plt.ylabel("Score")
    plt.xlabel("Example #")
    plt.legend()
    plt.show()


def print_results(scores, model_name):
    print(f"{model_name} Mean F1:  {scores['fmeasure'].mean()}")
    print(f"{model_name} Mean Bleu Score: {scores['bleu_score'].mean()}")
    print(f"{model_name} Mean Precision: {scores['precision'].mean()}")
    print(f"{model_name} Mean Recall: {scores['recall'].mean()}\n")


if __name__ == '__main__':
    train = pd.read_csv("data/train.csv")
    validation = pd.read_csv("data/validation.csv")
    test = pd.read_csv("data/test.csv")

    vocab = []
    vocab_counts = Counter()
    threshold = 5
    for i, row in train.iterrows():
        vocab_counts.update(row['article'].split())

    for word, occurence_count in vocab_counts.items():
        if occurence_count >= threshold:
            vocab.append(word)

    if os.path.exists(MODEL_NAME):
        vector_model = KeyedVectors.load(MODEL_NAME)
    else:
        vector_model = gensim.downloader.load('fasttext-wiki-news-subwords-300')
        vector_model.save(MODEL_NAME)

    # print(preprocess_data(train.head()["article"], vector_model))
    scorer = RougeScorer(["rouge2"], use_stemmer=True)
    baseline_scores = perform_baseline(test, scorer)
    print_results(baseline_scores, "Baseline Model")

    seq2seq_model = Seq2Seq(256, 100, vocab, .01)

    inputs = train['article'].values.tolist()
    references = train['highlights'].values.tolist()
    val_inputs = validation['article'].values.tolist()

    seq2seq_model.train_model(inputs, references, epochs=10)
    results = seq2seq_model.predict(val_inputs)
    rouge_scores = rogue_score(results, references, scorer)
    bleu_scores = bleu_score(results, references)

    seq2seq_scores = pd.merge(rouge_scores, bleu_scores, on=['prediction', 'reference'])
    print_results(seq2seq_scores, "Seq2Seq Model")

