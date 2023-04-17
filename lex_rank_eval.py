from models.lex_rank import LexRank

import pandas as pd
import zipfile
from rouge_score.rouge_scorer import RougeScorer
from utils import bleu_score, print_results, rogue_score

"""
Evaluating the LexRank model
"""

if __name__ == "__main__":
    model = LexRank()
    k = 2
    n_samples = 1000
    data = None

    print(f"Testing with {k} sentences per summary and {n_samples} samples\n")
    print("Loading data...\n")
    with zipfile.ZipFile("data.zip") as archive:
        with archive.open("cnn_dailymail/test.csv") as f:
            data = pd.read_csv(f)            

    articles = data["article"].tolist()[0:n_samples]
    references = [summary.split() for summary in data["highlights"].tolist()[0:n_samples]]

    print("Data has been loaded. Generating summaries...\n")
    predictions = [model.predict(article, k) for article in articles]

    scorer = RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    rouge_scores = rogue_score(predictions, references, scorer)
    bleu_scores = bleu_score(predictions, references)

    lexrank_scores = pd.merge(rouge_scores, bleu_scores, on='reference')
    print_results(lexrank_scores, "LexRank")