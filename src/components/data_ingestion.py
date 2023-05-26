import os 
import sys
from dataclasses import dataclass

import boto3
import pyarrow
import pandas as pd
# from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging
# from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    blog_data_path: str=os.path.join('artifacts/data/blog/') 
    news_data_path: str=os.path.join('artifacts/data/news/')
    social_data_path: str=os.path.join('artifacts/data/social/')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        # session = boto3.Session( aws_access_key_id='AKIARB5CFGGOCDATTVVR', 
        #                          aws_secret_access_key='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') 
        # s3 = session.resource('s3')
        self.s3 = s3 = boto3.resource('s3')
     
    def download_files(self):
        logging.info("Connecting to s3 to download parquet files")
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.blog_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.news_data_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.ingestion_config.social_data_path), exist_ok=True)

            my_bucket = self.s3.Bucket("peakmetrics-challenges")
            for my_bucket_object in my_bucket.objects.all():
                    print("downloading:", my_bucket_object.key)
                    
                    file_name = my_bucket_object.key.split('/')[2]
                    
                    if my_bucket_object.key.split('/')[1] == 'blog':
                        path = self.ingestion_config.blog_data_path
                    elif my_bucket_object.key.split('/')[1] == 'news':
                        path = self.ingestion_config.news_data_path
                    else:
                        path = self.ingestion_config.social_data_path

                    self.s3.meta.client.download_file('peakmetrics-challenges', my_bucket_object.key, path+file_name)
        except Exception as e:
            raise CustomException(sys, e)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.download_files()
