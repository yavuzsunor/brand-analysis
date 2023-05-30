import sys
from dataclasses import dataclass

from src_brand.exception import CustomException
from src_brand.logger import logging
from src_brand.components.data_transformation import DataTransformation
from src_brand.components.model_trainer import ModelTrainer

@dataclass
class Inference:

    def train_custom_model(self, media, key_word):
        
        data_transformation = DataTransformation()
        model_trainer = ModelTrainer()

        try:
            df = data_transformation.read_data(media)
            df = data_transformation.clean_transform(df, media)
            text_to_model = data_transformation.select_airline(df, key_word)
            
            return model_trainer.train_BERTopic(model_trainer.generic_model, text_to_model)
        
        except Exception as e:
            raise CustomException(sys, e)  
    
    def main_topics(self, model):

        try:
            freq = model.get_topic_info()
            print(freq.head()) 

        except Exception as e:
            raise CustomException(sys, e)

if __name__ == "__main__":

    model_inference = Inference()
    model, topics, prob = model_inference.train_custom_model("blog", "united airlines")
    model_inference.main_topics(model)