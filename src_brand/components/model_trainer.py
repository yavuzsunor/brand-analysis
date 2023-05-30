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
    
    # def train_custom_BERTopic(self, airline_name):
    #     try:
            
    #         social_united_list = df_social_UNITED['text_clean'].tolist()
    #         topics, probs = social_united_model.fit_transform(social_united_list)            
        
    #     except Exception as e:
    #         raise CustomException(sys, e)

if __name__ == "__main__":

    data_transformation = DataTransformation()
    df = data_transformation.read_data("social")
    df = data_transformation.clean_transform(df, "social")
    data_transformation.select_airline(df, "united airlines")
