import os

import gensim.downloader
import torch.cuda
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors, Word2Vec
from gensim.summarization.summarizer import summarize
from rouge_score.rouge_scorer import RougeScorer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, RegexpTokenizer
import matplotlib.pyplot as plt
from seq2seq import Seq2Seq
from collections import Counter
from typing import List
from transformers import pipeline

MODEL_NAME = "pretrained.model"


def preprocess_data(data: pd.Series) -> List[List[str]]:
    """
    Transforms a Series of data into a list of list of word vectors for each document

    :param data:
    :return: list of list of vectors
    """
    data = data.str.lower()
    clean_texts = []
    for i, text in data.items():
        remove_punct = RegexpTokenizer(r'\w+')
        clean_texts.append(remove_punct.tokenize(text))

    return clean_texts


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
        predicted = ' '.join(predicted)
        reference = ' '.join(reference)

        score = scorer.score(predicted, reference)['rouge2']
        score_dict = pd.DataFrame([{'prediction': predicted, 'reference': reference, 'precision': score.precision,
                                    'recall': score.recall, 'fmeasure': score.fmeasure}])

        scores = pd.concat([scores, score_dict], ignore_index=True)

    return scores


def bleu_score(predictions, references):

    scores = pd.DataFrame(columns=['prediction', 'reference', 'bleu_score'])
    smooth = SmoothingFunction()
    for prediction, reference in zip(predictions, references):
        score = sentence_bleu(reference, prediction, smoothing_function=smooth.method3)
        score_dict = pd.DataFrame([{"prediction": ' '.join(prediction), "reference": ' '.join(reference), "bleu_score": score}])
        scores = pd.concat([scores, score_dict], ignore_index=True)

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


def perform_transformer_baseline(inputs: List[List[str]], reference_summaries: List[List[str]], scorer) -> pd.DataFrame:
    summaries = []
    references = []
    summarizer = pipeline("summarization", model="google/pegasus-xsum")

    for (article, reference) in zip(inputs, reference_summaries):
        article = ' '.join(article)
        article = article[:500]
        summaries.append(word_tokenize(summarizer(article)[0]["summary_text"]))
        references.append(reference)

    rouge_scores = rogue_score(summaries, references, scorer)
    bleu_scores = bleu_score(summaries, references)

    scores = pd.merge(rouge_scores, bleu_scores, on='reference')

    return scores


if __name__ == '__main__':
    print("Beginning Encoder-Decoder Model Training")

    train = pd.read_csv("data/train.csv")
    validation = pd.read_csv("data/validation.csv")
    test = pd.read_csv("data/test.csv")

    training_inputs = preprocess_data(train["article"][:1000])
    training_references = preprocess_data(train["highlights"][:1000])
    validation_inputs = preprocess_data(validation["article"][:1000])
    validation_references = preprocess_data(validation["highlights"][:1000])

    scorer = RougeScorer(["rouge2"], use_stemmer=True)

    transformer_test_scores = perform_transformer_baseline(validation_inputs, validation_references, scorer)
    print_results(transformer_test_scores, "Transformer Baseline")
    transformer_test_scores.to_csv("transformer_test_scores.csv")

    vector_model = Word2Vec(sentences=training_inputs, size=300, window=5, min_count=1, workers=4, sg=1)
    vector_model.save("original_w2v.npy")
    # vector_model = Word2Vec.load("original_w2v.npy")

    print("Trained Word2Vec Model... Beginning Training")
    # baseline_scores = perform_baseline(test, scorer)
    # print_results(baseline_scores, "Baseline Model")

    seq2seq_model = Seq2Seq(300, 100, vector_model, .01)

    losses = seq2seq_model.train_model(training_inputs, training_references, epochs=10)
    seq2seq_model.save()

    print("Done Training model, Starting Evaluation:")

    results = seq2seq_model.predict(validation_references)
    rouge_scores = rogue_score(results, validation_references, scorer)
    bleu_scores = bleu_score(results, validation_references)

    seq2seq_scores = pd.merge(rouge_scores, bleu_scores, on='reference')
    print_results(seq2seq_scores, "Seq2Seq Model")
    seq2seq_scores.to_csv("validation_scores.csv")

    plt.plot(range(len(losses)), losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Encoder-Decoder Training Loss")
    plt.show()

    plt.savefig("Loss Chart")
