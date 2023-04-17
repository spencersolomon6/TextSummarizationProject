from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
from gensim.summarization.summarizer import summarize
from rouge_score.rouge_scorer import RougeScorer
from nltk.corpus import stopwords
from nltk.tokenize import  RegexpTokenizer
from typing import List
from models.lda import LDA 
from utils import preprocess_data, rogue_score, bleu_score, print_results


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