import sys
from dataclasses import dataclass

from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

from src_brand.exception import CustomException
from src_brand.logger import logging
from src_brand.components.data_transformation import DataTransformation


@dataclass
class ModelTrainer:

    vectorizer_model = CountVectorizer(stop_words="english")
    generic_model = BERTopic(
        vectorizer_model=vectorizer_model, 
        language='english', 
        calculate_probabilities=True,
        verbose=True)
    
    def train_BERTopic(self, bert_model, text_list):
        try:

            topics, probs = bert_model.fit_transform(text_list)

            return bert_model, topics, probs           
        
        except Exception as e:
            raise CustomException(sys, e)
