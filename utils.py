from rouge_score.rouge_scorer import RougeScorer
from nltk.tokenize import RegexpTokenizer
from gensim.summarization.summarizer import summarize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from models.seq2seq import CustomDataset


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


def rogue_score(predictions: List[List[str]], references: List[List[str]], scorer: RougeScorer):
    """
    Returns a DataFrame of scores for each prediction/reference pair

    :param predictions:
    :param references:
    :param scorer:
    :return: A DataFrame containing the prediction, reference and scores for each example
    """

    scores = pd.DataFrame(columns=['prediction', 'reference'])
    for predicted, reference in zip(predictions, references):
        predicted = ' '.join(predicted)
        reference = ' '.join(reference)

        score = scorer.score(predicted, reference)
        score1 = score['rouge1']
        score2 = score['rouge2']
        score_dict = pd.DataFrame([{'prediction': predicted, 'reference': reference, 'unigram_precision': score1.precision,
                                    'unigram_recall': score1.recall, 'unigram_fmeasure': score1.fmeasure, 'bigram_precision': score2.precision,
                                    'bigram_recall': score2.recall, 'bigram_fmeasure': score2.fmeasure}])

        scores = pd.concat([scores, score_dict], ignore_index=True)

    return scores


def bleu_score(predictions: List[List[str]], references: List[List[str]]):

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

    rouge_scores = rogue_score(summaries, references, scorer)
    bleu_scores = bleu_score(summaries, references)

    scores = pd.merge(rouge_scores, bleu_scores, on='reference')

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
    plt.title("ROUGE-2 Scores")
    plt.ylabel("Score")
    plt.xlabel("Example #")
    plt.legend()
    plt.show()


def print_results(scores, model_name):
    print(f"{model_name} Mean Bleu Score: {scores['bleu_score'].mean()}\n")

    print(f"{model_name} Mean Unigram Recall: {scores['unigram_recall'].mean()}")
    print(f"{model_name} Mean Unigram Precision: {scores['unigram_precision'].mean()}")
    print(f"{model_name} Mean Unigram F-Measure: {scores['unigram_fmeasure'].mean()}\n")

    print(f"{model_name} Mean Bigram Recall: {scores['bigram_recall'].mean()}")
    print(f"{model_name} Mean Bigram Precision: {scores['bigram_precision'].mean()}")
    print(f"{model_name} Mean Bigram F-Measure: {scores['bigram_fmeasure'].mean()}")


def preprocess_for_transformer(data: pd.DataFrame, tokenizer: AutoTokenizer):
    inputs = ["summarize: " + row for row in data["article"]]
    highlights = [row for row in data["highlights"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    references = tokenizer(highlights, max_length=128, truncation=True)

    return CustomDataset(model_inputs, references["input_ids"])