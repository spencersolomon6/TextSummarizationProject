from typing import List
import copy
from collections import defaultdict
from base_model import BaseModel

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



"""
    This is an implementation of the LexRank text summarization
    method.

    Implemented as according to: https://arxiv.org/abs/1109.2128
"""

class LexRank(BaseModel):
    def __init__(self, threshold: float = 0.08):
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

    def __process_cos_sim_mat(self, cosine_mat: np.ndarray, n: int):
        """
        Processes the given cosine similarity matrix by removing values that are above
        the similarity threshold and then normalizing each value by the degree of its row.

        Parameters:
        -----------
            cosine_mat : ndarray
                Cosine similarity matrix to be processed
            n : int
                Dimension of cosine_mat, which is a (n x n) matrix
        
        Returns:
            ndarray : the processed similarity matrix
            defaultdict : a dictionary with the degree of each row
        """

        mat = copy.deepcopy(cosine_mat)
        degrees = defaultdict(int)

        for i in range(n):
            for j in range(n):
                if mat[i][j] > self.threshold:
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
        for _ in range(10):
            p_t = np.matmul(sim_mat.T, p_t_min_1)

        return p_t


    def predict(self, data: List[str], k: int) -> List[str]:
        """
        Summarizes the given data by extracting the top k sentences
        in the data.
        
        Parameters:
        -----------
            data : List[str]
                The corpus of data from which to extract the summary from.
            k : int
                The number of sentences to be included in the summary.
        
        Returns:
            List[str] : an extractice summary of the given corpus
        """
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(data)
        cosine_mat = cosine_similarity(tfidf, tfidf)

        n = len(data)

        sim_mat, degrees = self.__process_cos_sim_mat(cosine_mat, n)
        scores = self.__power_method(sim_mat, degrees)

        indices_sorted_by_score = np.argsort(scores)

        summary = [data[i] for i in indices_sorted_by_score][: k]

        return summary


        


