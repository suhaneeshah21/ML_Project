import sys
from dataclasses import dataclass

import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_transformer_object(self):
        try:
            num_features = ['writing_score', 'reading_score']
            cat_features = ['gender', 'race_ethnicity', 'parental_level_of_education',
                            'lunch', 'test_preparation_course']
            
            num_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy="median")),
                ('scaler',StandardScaler())
            ])

            cat_pipeline=Pipeline(steps=[
                ('imputer',SimpleImputer(strategy="most_frequent")),
                ('one_hot_encoder',OneHotEncoder())
            ])


            logging.info("Numerical and categorical pipelines created")

            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,num_features),
                ("cat_pipeline",cat_pipeline,cat_features)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessor object")
            preprocessor_obj=self.get_transformer_object()

            target_column_name="math_score"
            num_features = ['writing_score', 'reading_score']

            input_feat_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feat_train_df=train_df[target_column_name]

            input_feat_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feat_test_df=test_df[target_column_name]

            logging.info("Applying preprocessing on train and teat data")

            input_feat_train_arr=preprocessor_obj.fit_transform(input_feat_train_df)
            input_feat_test_arr=preprocessor_obj.transform(input_feat_test_df)

            train_arr=np.c_[
                input_feat_train_arr,np.array(target_feat_train_df)
            ]

            test_arr=np.c_[
                input_feat_test_arr,np.array(target_feat_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

            
        except Exception as e:
            raise CustomException(e,sys)
            
            