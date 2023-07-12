import sys 
from dataclasses import dataclass

import pandas as pd 
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler

from src.logger import logging
from src.exception import CustomException
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation initiated')
            numerical_cols=['Delivery_person_Age', 'Delivery_person_Ratings', 'Restaurant_latitude',
                            'Restaurant_longitude', 'Delivery_location_latitude',
                            'Delivery_location_longitude', 'Time_Orderd', 'Time_Order_picked',
                            'Vehicle_condition', 'multiple_deliveries', 'day', 'month', 'year']
            
            one_columns=['Road_traffic_density','Festival','City']
            ohe_columns=['Type_of_order','Type_of_vehicle','Weather_conditions']

            
            

            traffic_cat=['Jam', 'High', 'Medium', 'Low']
            weather_cat=['Fog', 'Stormy', 'Sandstorms', 'Windy', 'Cloudy', 'Sunny']
            festivel_cat=['Yes','No']
            city_cat=['Metropolitian', 'Urban', 'Semi-Urban']
            order_cat=['Snack', 'Meal', 'Drinks', 'Buffet']
            vehicle_cat=['motorcycle', 'scooter', 'electric_scooter', 'bicycle']
            
            logging.info('Pipeline initiated')
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                    ]
                )
            cat_pipeline_one=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('ordinalencoder',OrdinalEncoder(categories=[traffic_cat,festivel_cat,city_cat])),
                    ('scaler',StandardScaler())
                    ]
                )
            cat_pipeline_ohe=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    
                    ('onehotencoder',OneHotEncoder(sparse=False,drop='first')),
                    ('scaler',StandardScaler())
                    
                    ]
                )
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,numerical_cols),
                ('cat_pipeline_one',cat_pipeline_one,one_columns)
                
                ])
            
            
            return(preprocessor)
            logging.info("pipeline completed")
        except Exception as e:
            logging.info("error occeured in Date Transformation (pipeline)")
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            
            train_df['Time_Orderd']=train_df["Time_Orderd"].str.replace(":",".")
            test_df['Time_Orderd']=test_df["Time_Orderd"].str.replace(":",".")

            train_df['Time_Orderd']=train_df['Time_Orderd'].astype('float')
            test_df['Time_Orderd']=test_df['Time_Orderd'].astype('float')

            train_df['Time_Order_picked']=train_df['Time_Order_picked'].str.replace(':',".")
            test_df['Time_Order_picked']=test_df['Time_Order_picked'].str.replace(':',".")

            train_df['day']=train_df['Order_Date'].str.split('-').str[0]
            test_df['day']=test_df['Order_Date'].str.split('-').str[0]
            train_df['day']=train_df['day'].astype('int')
            test_df['day']=test_df['day'].astype('int')

            train_df['month']=train_df['Order_Date'].str.split('-').str[1]
            test_df['month']=test_df['Order_Date'].str.split('-').str[1]
            train_df['month']=train_df['month'].astype('int')
            test_df['month']=test_df['month'].astype('int')

            train_df['year']=train_df['Order_Date'].str.split('-').str[2]
            test_df['year']=test_df['Order_Date'].str.split('-').str[2]
            train_df['year']=train_df['year'].astype('int')
            test_df['year']=test_df['year'].astype('int')

          
            
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
            
            
            logging.info('Read Train Test data complected')
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Train Dataframe Head : \n{test_df.head().to_string()}")

            logging.info('obtaning preprocessing object')

            target_column='Time_taken (min)'
            drop_column=['ID','Delivery_person_ID','Order_Date']

            input_feature_train_df=train_df.drop(columns=drop_column,axis=1)
            target_feature_train_df=train_df[target_column]

            input_feature_test_df=test_df.drop(columns=drop_column,axis=1)
            target_feature_test_df=test_df[target_column]



            
            logging.info('Applying preprocessing object on traning and testing datasets')
            

            preprocessing_obj=self.get_data_transformation_object()
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df,with_mean=False)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df,with_mean=False)

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info("Preprocessor pickel file saved")

            return(train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)

            
        except Exception as e:
            logging.info('error occure in initiate data transformation')
            raise CustomException(e,sys)

