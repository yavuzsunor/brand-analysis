import sys
from dataclasses import dataclass

from src_brand.exception import CustomException
from src_brand.logger import logging
from src_brand.components.data_transformation import DataTransformation
from src_brand.components.model_trainer import ModelTrainerConfig, ModelTrainer
from src_brand.utils import make_model_name, save_model, load_model

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
            raise CustomException(e, sys)  
    
    def main_topics(self, model):

        try:
            freq = model.get_topic_info()
            print(freq.head()) 

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":

    model_inference = Inference()
    model_trainer_config = ModelTrainerConfig()

    media = "blog"
    keyword = "united airlines"
    model_name = make_model_name(media, keyword)
    
    # train models and save them
    # trained_model, topics, prob = model_inference.train_custom_model(media, keyword)
    # save_model(model_trainer_config.model_data_path, model_name, trained_model)
    # logging.info("Training and saving BERTopic model")

    # load models and show their results
    loaded_model = load_model(model_trainer_config.model_data_path, model_name)
    model_inference.main_topics(loaded_model)