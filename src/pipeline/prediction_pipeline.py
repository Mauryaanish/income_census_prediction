import sys
import os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 Age:int,
                 Work_class:str,
                 Final_weight:int,
                 Education:str,
                 Education_num:int,
                 Marital_status:str,
                 Occupation:str,
                 Relationship:str,
                 Race:str,
                 Sex:str,
                 Capital_gain:int,
                 Capital_loss:int,
                 Hours_per_week:int,
                 Native_county:str):
    
    
        
        self.Age=Age
        self.Work_class=Work_class
        self.Final_weight=Final_weight
        self.Education=Education
        self.Education_num=Education_num
        self.Marital_status=Marital_status
        self.Occupation = Occupation
        self.Relationship =  Relationship
        self.Race = Race
        self.sex = Sex
        self.Capital_gain = Capital_gain
        self.Capital_loss = Capital_loss
        self.Hours_per_week = Hours_per_week
        self.Native_county = Native_county
    

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'Age':[self.Age],
                'Work_class':[self.Work_class],
                'Final-weight':[self.Final_weight],
                'Education':[self.Education],
                'Education-num':[self.Education_num],
                'Marital-status':[self.Marital_status],
                'Occupation':[self.Occupation],
                'Relationship':[self.Relationship],
                'Race':[self.Race],
                'sex' :[self.sex],
                'Capital_gain' : [self.Capital_gain],
                'Capital_loss' : [self.Capital_loss],
                'Hours_per_week' : [self.Hours_per_week],
                'Native_county' : [self.Native_county]
             }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)