import sys
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from dataclasses import dataclass


@dataclass 
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self,train_array,test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            classifier_regressor = LogisticRegression(penalty= 'l2',C= 20 ,max_iter=100)
            classifier_regressor.fit(X_train,y_train)
            ## prediction 
            y_pred = classifier_regressor.predict(X_test)
            score = accuracy_score(y_pred , y_test)

            confusion_mat = confusion_matrix(y_pred , y_test)
            report = classification_report (y_test , y_pred)

            print(f'Model Name: Logistic Regression, Accuracy Score: {score}')
            logging.info(f'Model Name: Logistic Regression, Accuracy Score: {score}')

            print('\n====================================================================================\n')
            print(f'classification Report: {report}')
            logging.info(f'classification Report: {report}')

            print("\n====================================================================================\n")
            print(f"Confusion Metrix : {confusion_mat}")

            logging.info(f'classification Confusion Metrix: {confusion_mat}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=classifier_regressor
            )

        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise CustomException(e,sys)