from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
from gensim.summarization.summarizer import summarize
from rouge_score.rouge_scorer import RougeScorer
from nltk.corpus import stopwords
from nltk.tokenize import  RegexpTokenizer
from typing import List
from lda import LDA 


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


def print_results(scores, model_name):
    print(f"{model_name} Mean Bleu Score: {scores['bleu_score'].mean()}\n")

    print(f"{model_name} Mean Unigram Recall: {scores['unigram_recall'].mean()}")
    print(f"{model_name} Mean Unigram Precision: {scores['unigram_precision'].mean()}")
    print(f"{model_name} Mean Unigram F-Measure: {scores['unigram_fmeasure'].mean()}\n")

    print(f"{model_name} Mean Bigram Recall: {scores['bigram_recall'].mean()}")
    print(f"{model_name} Mean Bigram Precision: {scores['bigram_precision'].mean()}")
    print(f"{model_name} Mean Bigram F-Measure: {scores['bigram_fmeasure'].mean()}")


stop_words = set(stopwords.words('english'))

test = pd.read_csv("data/test.csv")

x_test = test['article'][0:1000]
y_test = preprocess_data(test['highlights'][0:1000])

lda = LDA(3, 3)
summaries = lda.predict(x_test)

summaries = [s.split(" ") for s in summaries]

scorer = RougeScorer(["rouge1", "rouge2"], use_stemmer=True)

rouge_scores = rogue_score(summaries, y_test, scorer)
bleu_scores = bleu_score(summaries,  y_test)

lda_scores = pd.merge(rouge_scores, bleu_scores, on='reference')

print(bleu_scores)
print(rouge_scores)
print_results(lda_scores, "LDA Model")