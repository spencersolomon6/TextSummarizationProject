from typing import List
import copy
from collections import defaultdict
from base_model import BaseModel

import numpy as np
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import zipfile
from rouge_score.rouge_scorer import RougeScorer
from main import bleu_score, print_results, rogue_score, preprocess_data


"""
    This is an implementation of the LexRank text summarization
    method.

    Implemented as according to: https://arxiv.org/abs/1109.2128
"""

class LexRank(BaseModel):
    def __init__(self, threshold: float = 0.01):
        """            
        Parameters:
        -----------
            threshold : float
                The similarity threshold. Sentence pairs with a similarity about this
                value will have their edge removed in order to prevent the summary
                from being overpopulated by similar sentences.
            k : int
                How many sentences to return for the summary
        """
        super().__init__()
        self.threshold = threshold


    def train(self, train_data: List[str], references: List[str]) -> None:
        """
        Unused since LexRank is not a "trainable" algorithm.
        """
        pass

    def __process_cos_sim_mat(self, cosine_mat: np.ndarray):
        """
        Processes the given cosine similarity matrix by removing values that are above
        the similarity threshold and then normalizing each value by the degree of its row.

        Parameters:
        -----------
            cosine_mat : ndarray
                Cosine similarity matrix to be processed
        
        Returns:
            ndarray : the processed similarity matrix
            defaultdict : a dictionary with the degree of each row
        """
        mat = copy.deepcopy(cosine_mat)
        degrees = defaultdict(int)

        n = cosine_mat.shape[0]

        for i in range(n):
            for j in range(n):
                if mat[i][j] >= self.threshold:
                    degrees[i] += 1
                else:
                    mat[i][j] = 0

        for i in range(n):
            for j in range(n):
                mat[i][j] = mat[i][j] / degrees[i]

        return mat, degrees

    def __power_method(self, sim_mat: np.ndarray, degrees: defaultdict):
        """
        Power iteration method for performing eigenvalue approximation.
        Implemented according to the LexRank paper.

        Parameters:
        -----------
            sim_mat : np.ndarray
                The similarity matrix which should have been already processed.
            degrees : defaultdict
                Dictionary with the degree of each row of the sim_mat
        
        Returns:
            np.ndarray: 1 dimensional row of eigenvalues
        """

        p_t_min_1 = np.ones(shape=len(degrees))/len(degrees)
        p_t = None
        for _ in range(100):
            p_t = np.matmul(sim_mat.T, p_t_min_1)

        return p_t

    def __preprocess_data(self, data: str) -> List[str]:
        """
        Performs pre-processing on the given string corpus by lowering all
        words and removing punctuation.

        

        """
        sentences = sent_tokenize(data)
        remove_punct = RegexpTokenizer(r'\w+')

        cleaned = []
        for sent in sentences:
            lowered = sent.lower()
            tokens = remove_punct.tokenize(lowered)
            clean_sent = " ".join(tokens)
            cleaned.append(clean_sent)
        
        return cleaned



    def predict(self, data: str, k: int) -> str:
        """
        Summarizes the given data by extracting the top k sentences
        in the data.
        
        Parameters:
        -----------
            data : str
                The corpus of data from which to extract the summary from.
            k : int
                The number of sentences to be included in the summary.
        
        Returns:
            List[str] : an extractice summary of the given corpus
        """
        vectorizer = TfidfVectorizer()
        processed_data = self.__preprocess_data(data)
        tfidf = vectorizer.fit_transform(processed_data)
        cosine_mat = cosine_similarity(tfidf, tfidf)

        sim_mat, degrees = self.__process_cos_sim_mat(cosine_mat)
        scores = self.__power_method(sim_mat, degrees)

        indices_sorted_by_score = np.argsort(scores)
        indices_sorted_by_score = indices_sorted_by_score[::-1] # reverse to get high to low order
        summary = [processed_data[i] for i in indices_sorted_by_score][: k]

        tokenized_summary = []
        for sent in summary:
            tokenized_summary += sent.split()

        return tokenized_summary


if __name__ == "__main__":
    model = LexRank()
    k = 2
    data = None

    with zipfile.ZipFile("data.zip") as archive:
        with archive.open("cnn_dailymail/test.csv") as f:
            data = pd.read_csv(f)            
            print(data.head())

    articles = data["article"].tolist()[0:1000]
    references = preprocess_data(data["highlights"])[0:1000]
    predictions = [model.predict(article, k) for article in articles]

    scorer = RougeScorer(["rouge1", "rouge2"], use_stemmer=True)
    rouge_scores = rogue_score(predictions, references, scorer)
    bleu_scores = bleu_score(predictions, references)

    lexrank_scores = pd.merge(rouge_scores, bleu_scores, on='reference')
    print_results(lexrank_scores, "LexRank")
