import os 
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.Data_transformation import DataTransformation




##initiating the data ingection
@dataclass
class DataIngectionConfig:
    #this code wr
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

class DataIngection:
    def __init__(self):
        self.ingection_config=DataIngectionConfig()

    def initiate_data_ingection(self):
        logging.info('Data Ingection Method Starts')
        try:
            df=pd.read_csv(os.path.join('notebook/data','finalTrain.csv'))
            logging.info("Dataset read by pandas Dataframe")
           
            os.makedirs(os.path.dirname(self.ingection_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingection_config.raw_data_path,index=False)

            logging.info(f"Train Dataframe Head : \n{df.info()}")

            logging.info('Train Test Split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=40)

            
            logging.info("ingection of Data completed")
            

            
            
            train_set.to_csv(self.ingection_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingection_config.test_data_path,index=False,header=True)
            
            logging.info(f'train data set :\n{train_set.head().to_string()}')
            logging.info(f'test data set  :\n{test_set.head().to_string()}')
            logging.info("ingection of Data completed")

            return(
                self.ingection_config.train_data_path,
                self.ingection_config.test_data_path
            )

        except Exception as e:
            logging.info("error occur during initiating the data ingection")
            raise CustomException(e,sys)


if __name__=="__main__":
        obj=DataIngection()
        train_data_path,test_data_path=obj.initiate_data_ingection()
        data_transformation=DataTransformation()
        train_arr,test_arr = data_transformation.initiate_data_transformation(train_data_path,test_data_path)
    