import numpy as np
import tqdm
from typing import List
from models.base_model import BaseModel
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity


class LDA(BaseModel):
    def __init__(self, n_components, no_sentences):
        super().__init__()
        """
        Initializes the parameters used to create the LDA model

        :param n_components: The number of topics used in the lda model
        :param no_sentences: Number of sentences to include in the summary
        """
        self.n_components = n_components
        self.no_sentences = no_sentences

    def train(self, train_data: List[str]):
        """
        Trains the lda model based on the initialized parameters and dataset passed

        :param train_data: The data to be trained on the LDA model

        returns the trained model and document transformed to fit the topic distribution
        """
        lda = LatentDirichletAllocation(n_components = self.n_components , max_iter = 5, learning_method = "online",
                                    learning_offset = 50, random_state = 0)
        distribution = lda.fit_transform(train_data)
        return lda, distribution

    def predict(self, data: List[str]) -> List[str]:
        """
        Generates the summary based on the given dataset passed

        :param data: The dataset used to generate summaries will be a list of documents

        returns the summaries generate from the dataset
        """
        summaries = []
        vectorizer = TfidfVectorizer(stop_words = "english")
        for doc in tqdm.tqdm(data):
            #Remove some of the header sentences
            sents = [sent for sent in sent_tokenize(doc)  if len(sent.split()) > 6]
            try:
                document_model = vectorizer.fit_transform([" ".join(sents)])    #Fit document to TFIDF
                model, distribution = self.train(document_model)    #Train the lda model using the generated vectors
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
                    if(order in picked): #If the current sentence was already picked
                        continue
                    new_sentences = list(map(lambda x: x[0], total_sentences)) #Add current sentence to the total sentence
                    new_sentences.append(sentence)
                    
                    #Calculate the topic distribution for the new sentence
                    distribution_sentence = model.transform(vectorizer.transform([" ".join(new_sentences)])) 

                    #Use cosine similarity to score the current sentence with the lda distribution
                    current_score = cosine_similarity(distribution[0].reshape(1, -1), distribution_sentence[0].reshape(1, -1))
                    if(current_score > best_score):
                        best_score = current_score
                        best_sentence = [sentence, order]
                    order += 1
                if(best_sentence != ""):
                    total_sentences.append(best_sentence)
                    picked.add(best_sentence[1])
            
            #Add sentence and order the best picked sentences in the order they occur in the document
            summaries.append(" ".join(list(map(lambda x: x[0], sorted(total_sentences, key = lambda x: x[1])))))
        return summaries
