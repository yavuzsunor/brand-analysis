import os
import sys
import re

from bertopic import BERTopic
from src_brand.exception import CustomException


def save_model(model_dir_path, model_name, model_obj):
    try:

        model_obj.save(model_dir_path+model_name)

    except Exception as e:
        raise CustomException(e, sys)

def load_model(model_obj_path, model_name):
    try:
        
        return BERTopic.load(model_obj_path + model_name)

    except Exception as e:
        raise CustomException(e, sys)

def clean_tweet(tweet):
    try:
        temp = tweet.lower()
        temp = re.sub("'", "", temp) # to avoid removing contractions in english
        temp = re.sub("@[A-Za-z0-9_]+","", temp)
        temp = re.sub("#[A-Za-z0-9_]+","", temp)
        temp = re.sub(r'http\S+', '', temp)
        temp = re.sub('[()!?]', ' ', temp)
        temp = re.sub('\[.*?\]',' ', temp)
        temp = re.sub("[^a-z0-9]"," ", temp)
        temp = temp.split()
        temp = " ".join(word for word in temp)
        return temp
    
    except Exception as e:
        raise CustomException(e, sys)

def make_model_name(media, key_word):
    try:
        temp = media
        word_list = key_word.split(" ")
        
        for word in word_list:
            temp = temp +'_'+ word
        
        return temp

    except Exception as e:
        raise CustomException(e, sys)