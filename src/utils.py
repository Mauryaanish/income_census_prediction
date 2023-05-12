import os
import sys
import pickle

import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException

def whitespace_remover(dataframe):
    for col in dataframe.columns:
        # chaeking data type echa columns
        if dataframe[col].dtypes == "object":
                dataframe[col] = dataframe[col].map(str.strip)
        else:
            pass
    
    return dataframe



def Special_cha_remover(dataframe):
     
     for  col in dataframe.columns:
               # check every columns

          
          if dataframe[col].dtypes =="object":
              #removing special character remove in dataset
             
             dataframe[col] = dataframe[col].str.replace("-" , " ")

     return dataframe


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)

