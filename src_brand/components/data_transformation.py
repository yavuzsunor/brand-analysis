import sys
from dataclasses import dataclass

import os
import glob
import pandas as pd
import pyarrow
import importlib 

from src_brand.exception import CustomException
from src_brand.logger import logging
from src_brand.utils import clean_tweet


@dataclass
class DataTransformation:

    path = 'brand_artifacts/data/'

    def read_data(self, media_type):
        try:            
            parquet_files = glob.glob(os.path.join(self.path+media_type+'/', "*.parquet"))

            # loop over the list of csv files 
            for i, f in enumerate(parquet_files):
                # read the parquet file
                df_temp = pd.read_parquet(f)
                # print the location and filename
                print('File Name:', f.split("\\")[-1])
                if i == 0:
                    df = df_temp
                else:
                    df = pd.concat([df, df_temp])  

            df.reset_index(drop=True, inplace=True)
            print(df.info())
            return df

        except Exception as e:
            raise CustomException(sys, e) 


    def clean_transform(self, df, media_type):
        try:
            if media_type == 'social':
                df = df[(df.language == 'en') & (df.text.notna())]
                df['text_clean'] = df['text'].apply(lambda x: clean_tweet(x))
            else:
                df = df[(df['body'].str.len() > 30) & 
                        (df.language == 'en') &
                        (df.body.notna())].reset_index()
                df['text_clean'] = df['body'].apply(lambda x: x.lower())

            return df 

        except Exception as e:
            raise CustomException(sys, e) 


    def select_airline(self, df, airline_name):
        try:
        
            df = df[df['text_clean'].str.contains(airline_name)]
            text_list = df['text_clean'].tolist()   

            return text_list
        
        except Exception as e:
            raise CustomException(sys, e)
