import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,OneHotEncoder

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformatonConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformatonConfig()

    
    
    def get_data_transformation_object(self):
        try:
            logging.info('Data tranformation initiated')
            #Define which column should be ordinal-encoded
            numerical_cols=['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
                                'Restaurant_longitude', 'Delivery_location_latitude',
                                'Delivery_location_longitude', 'Time_Orderd', 'Time_Order_picked',
                                'Vehicle_condition', 'multiple_deliveries']
            one_columns=['Road_traffic_density','Festival']
            ohe_columns=['Type_of_order','Type_of_vehicle','City','Weather_conditions']
            
            ##Define the custom ranking for each ordinal variable
            traffic_cat=['Jam', 'High', 'Medium', 'Low']
            festivel_cat=['Yes','No']

            logging.info('Pipeline Initiated')
            
            
            #numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )

            #catagorical ordinal encoding pipeline
            cat_one_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder',OrdinalEncoder(categories=[traffic_cat,festivel_cat])),
                    ('scaler',StandardScaler()),

                ]
            )

            cat_ohe_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('Onehotencoder',OneHotEncoder()),
                    ('scaler',StandardScaler())
                ]
            )
            
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_one_pipeline',cat_one_pipeline,one_columns),
                ('cat_ohe_pipeline',cat_ohe_pipeline,ohe_columns)
            ]
            )
            
            return preprocessor
            logging.info('Pipeline Completed')

        except Exception as e:
            logging.info("error in Data transformation")
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            #reading train and test data
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test dataset completed")
            logging.info(f'train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head : \n{test_df.head().to_string()}')

            logging.info('obtaning preprocessing object')

            ## Feature engg 
              

            target_column_name='Time_taken (min)'
            drop_columns=[target_column_name,'ID','Delivery_person_ID']  

            train_df['Time_Orderd']=train_df["Time_Orderd"].str.replace(":",".")
            test_df['Time_Orderd']=train_df["Time_Orderd"].str.replace(":",".")
            
            train_df['Time_Orderd']=train_df['Time_Orderd'].astype('float')
            test_df['Time_Orderd']=test_df['Time_Orderd'].astype('float')

            #for replacing the ':' by "."
            train_df['Time_Order_picked']=train_df['Time_Order_picked'].str.replace(':',".")
            test_df['Time_Order_picked']=test_df['Time_Order_picked'].str.replace(':',".")

            #then also the data containtwo point that we need to remove
            for i in train_df['Time_Order_picked']:
                string = i
                second_dot_index = string.find('.', string.find('.') + 1)
 
                if second_dot_index != -1:
                    modified_string = string[:second_dot_index] + string[second_dot_index + 1:]
                    train_df['Time_Order_picked']=modified_string
                else:
                    pass

            for i in test_df['Time_Order_picked']:
                string = i
                second_dot_index = string.find('.', string.find('.') + 1)
 
                if second_dot_index != -1:
                    modified_string = string[:second_dot_index] + string[second_dot_index + 1:]
                    test_df['Time_Order_picked']=modified_string
                else:
                    pass

            #change their data type
            
            train_df['Time_Order_picked']=train_df["Time_Order_picked"].astype('float')
            test_df['Time_Order_picked']=test_df["Time_Order_picked"].astype('float')
            
            input_feature_train_df=train_df.drop(columns=drop_columns,axis=1)
            target_feature_train_df=train_df[target_column_name] 

            input_feature_test_df=test_df.drop(columns=drop_columns,axis=1)
            target_feature_test_df=test_df[target_column_name]

            preprocessing_obj=self.get_data_transformation_object()

            ## Trnasformating using preprocessor obj
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df) 
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df) 
        

            logging.info('Applying preprocessing object on traning and testing datasets')
            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            logging.info("Exception occure in the inititate_datatransformation")
            raise CustomException(e,sys)
        

