import numpy as np
import tqdm
from typing import List
from base_model import BaseModel
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity


class LDA(BaseModel):
    def __init__(self, n_components, no_sentences):
        super().__init__()
        self.n_components = n_components
        self.no_sentences = no_sentences

    def train(self, train_data: List[str]):
        lda = LatentDirichletAllocation(n_components = self.n_components , max_iter = 5, learning_method = "online",
                                    learning_offset = 50, random_state = 0)
        distribution = lda.fit_transform(train_data)
        return lda, distribution

    def predict(self, data: List[str]) -> List[str]:
        summaries = []
        vectorizer = TfidfVectorizer(stop_words = "english")
        for doc in tqdm.tqdm(data):
            
            sents = [sent for sent in sent_tokenize(doc)  if len(sent.split()) > 6]
            try:
                document_model = vectorizer.fit_transform([" ".join(sents)])
                model, distribution = self.train(document_model)
            except:
                print("Error", doc)
                continue

            total_sentences = []
            picked = set()
    
            
            for _ in range(self.no_sentences):
                best_sentence = ""
                best_score = float("-inf")
                order = 0
                for sentence in sents:
                    if(order in picked):
                        continue
                    new_sentences = list(map(lambda x: x[0], total_sentences))
                    new_sentences.append(sentence)
                    
                    distribution_sentence = model.transform(vectorizer.transform([" ".join(new_sentences)]))
                    
                    current_score = cosine_similarity(distribution[0].reshape(1, -1), distribution_sentence[0].reshape(1, -1))
                    if(current_score > best_score):
                        best_score = current_score
                        best_sentence = [sentence, order]
                    order += 1
                if(best_sentence != ""):
                    total_sentences.append(best_sentence)
                    picked.add(best_sentence[1])
            summaries.append(" ".join(list(map(lambda x: x[0], sorted(total_sentences, key = lambda x: x[1])))))
        return summaries
