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
                lda, distribution = self.train(document_model)
            except:
                print("Error", doc)
                continue
            distribution_sentences = []
            ranked_list = {}
            for i, sent in enumerate(sents):
                distribution_sentence = lda.transform(vectorizer.transform([sent]))
                distribution_sentences.append(distribution_sentence)
                #print(np.array(cosine_similarity(distribution, distribution_sentence)).flatten())   
                ranked_list[i] = np.array(cosine_similarity(distribution, distribution_sentence)).flatten()[0]


            ranked_list = sorted(ranked_list.items(), key=lambda x:x[1],reverse=True)
            #print(ranked_list)

            summary = []
            for i, rank in ranked_list:
                if(len(summary) < self.no_sentences):
                    if(summary == []):
                        summary.append(sents[i])
                    else:
                        distribution_summary = lda.transform(vectorizer.transform([" ".join(summary)]))
                        cur_sent = distribution_sentences[i]
                        if(cosine_similarity(cur_sent, distribution_summary) < 0.66):
                            summary.append(sents[i])

            summaries.append([" ".join(summary)])
        return summaries
