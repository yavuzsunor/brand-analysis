import sys
from dataclasses import dataclass

import os
import glob
import pandas as pd
import pyarrow

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
    
    def train_custom_BERTopic(self, bert_model, text_list):
        try:

            topics, probs = bert_model.fit_transform(text_list)

            return bert_model, topics, probs           
        
        except Exception as e:
            raise CustomException(sys, e)

if __name__ == "__main__":

    data_transformation = DataTransformation()
    df = data_transformation.read_data("social")
    df = data_transformation.clean_transform(df, "social")
    text_to_model = data_transformation.select_airline(df, "united airlines")

    model_trainer = ModelTrainer()
    model, topics, prob = model_trainer.train_custom_BERTopic(model_trainer.generic_model, text_to_model)
    
    freq = model.get_topic_info()
    print(freq.head())
    
