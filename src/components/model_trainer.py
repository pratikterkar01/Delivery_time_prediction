import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
from src.utils import evaluate_model

from dataclasses import dataclass
import sys
import os

@dataclass
class ModeltrainerConfig:
    trained_model_file_path=os.path.join('artifacts','model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModeltrainerConfig()

    def initiate_model_traning(self,train_array,test_array):
        try:
            logging.info("Spliting Dependent and Independent variables from train and test data")

            x_train,y_train,x_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                'LinearRegression':LinearRegression(),
                'Lasso':Lasso(),
                'Ridge':Ridge(),
                'Elasticnet':ElasticNet()
            }

            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
            print(model_report)
            print('\n =================================================================================\n')
            logging.info(f'Model Report : {model_report}')

            #To get best score from dictonery
            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model=models[best_model_name]

            print(f'Best Model Found ,Model name : {best_model_name},R2 Score :{best_model_score}')
            print('/n================================================================\n')
            logging.info(f"best model Found , model Name : {best_model_name} , R2 Score : {best_model_score}")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            raise CustomException(e,sys)
            