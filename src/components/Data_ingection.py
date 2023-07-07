import os
import sys 
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.Data_transformation import DataTransformation

#initialising the data ingestion
@dataclass
class DataIngectionconfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')

#create a class for data ingection
class Dataingection:
    def __init__(self):
        self.ingection_config=DataIngectionconfig()

    def initiate_data_ingection(self):
        logging.info('Data Ingection method starts')
        try:
            #reading the csv
            df=pd.read_csv(os.path.join('notebook/data','finalTrain.csv'))
            logging.info('DataSet Read as Dataframe')

            #making copyof that file as raw data
            os.makedirs(os.path.dirname(self.ingection_config.raw_data_path),exist_ok=True)
            df.to_csv(self.ingection_config.raw_data_path,index=False)

            #train test split
            logging.info('train test split')
            train_set,test_set=train_test_split(df,test_size=0.30,random_state=40)
            
            #spliting tha data to csv
            train_set.to_csv(self.ingection_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingection_config.test_data_path,index=False,header=True)
            
            logging.info('Ingetion of Data is completed')

            return(
                self.ingection_config.train_data_path,
                self.ingection_config.test_data_path
            )
        except Exception as e:
            logging.info('Exception occured at Data Ingection Stage')
            raise CustomException(e,sys)
        

##runing the file
if __name__=="__main__":
    obj=Dataingection()
    train_data_path,test_data_path=obj.initiate_data_ingection()
    data_transformation=DataTransformation()
    train_arr,test_arr,_= data_transformation.initiate_data_transformation(train_data_path,test_data_path)


