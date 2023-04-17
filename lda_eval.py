import pandas as pd
from rouge_score.rouge_scorer import RougeScorer
from models.lda import LDA 
from utils import preprocess_data, rogue_score, bleu_score, print_results


"""
    Evaluation of the LDA model.

    Before running this file, make sure you have extracted the 3 data files
    from the CNN-Dailymail dataset from Kaggle.

    Make sure there is a "data" folder in the same place that this file is run.
    That folder should have the following three files: "train.csv", "validation.csv",
    and "test.csv"
"""


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