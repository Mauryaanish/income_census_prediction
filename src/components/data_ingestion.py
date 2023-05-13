import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.utils import whitespace_remover
from src.components.data_transformation import DataTransformation
from src.utils import Special_cha_remover

@dataclass

class DataIngestionconfig:
    train_data_path:str=os.path.join('artifacts' , 'train.csv')
    test_data_path:str = os.path.join('artifacts' , 'test.csv')
    raw_data_path:str = os.path.join('artifacts' , 'raw.csv')

## create a class for Data Ingestion

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Method Starts")
        try:
            df = pd.read_csv(os.path.join('notebooks/data' , 'adult.data') , header=None , names=['Age' , 'Work_class' , 'Final_weight' , 'Education' ,'Education_num' ,'Marital_status' , 'Occupation' , 'Relationship' , 'Race' , 'Sex' , 'Capital_gain' , 'Capital_loss' , 'Hours_per_week' , 'Native_county' , 'Income' ])
            # Data cleaing part 

            logging.info('Dateset read as pandas Dataframe')
            df = whitespace_remover(df)

            logging.info('Remove White Space in Data set')
            df = Special_cha_remover(df)
            logging.info("Remove Special Cherater In Dataset")

            ## Capital Loss and final weight is not co-releted for  target feature.
            df.drop(labels = ["Capital_loss" , "Final_weight" , "Native_county"] , axis = 1  , inplace = True)
            # occupation "? " replace  occupation mode
            df["Occupation"].mode()
            df["Occupation"] = df["Occupation"].replace("?" , "Prof specialty")
           
            # remove this index because this is a single group 
            df.drop(index = 19609 , axis = 0 , inplace = True)

            # maping income <=50K = 0 , >50K = 1
            map_income = {"<=50K" : 0 , ">50K" : 1}
            df["Income"] = df["Income"].map(map_income)
            logging.info("Data Cleaing part is complete")

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path) , exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path , index = False)

            logging.info('Train Test Split')
            train_set , test_set = train_test_split(df , test_size=0.3 , random_state= 42)
            train_set.to_csv(self.ingestion_config.train_data_path , index = False , header = True)
            test_set.to_csv(self.ingestion_config.test_data_path , index = False , header = True)

            logging.info('Ingetion of data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path )




            
        except Exception as e:
            logging.info("Exception occured at data Ingestion stage")
            raise CustomException(e , sys)
        

