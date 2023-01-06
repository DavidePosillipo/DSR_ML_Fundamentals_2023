import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt
from pickle import dump
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from hyperopt import fmin, tpe, STATUS_OK, Trials, space_eval
from hyperopt.pyll import scope
from functools import partial

import sklearn

#import hyperparameter configs
from scripts.config import *


class Preprocessing:


    def __init__(self, input_data_path, task_type='regression'):
        self.input_data_path = input_data_path
        self.task_type = task_type
        if task_type=='regression':
            self.target = target_regression
        elif task_type=='classification':
            self.target = target_classification
        else:
            raise TypeError("Target not available for the requested task")


    def read_dataframe(self, request_tgt):

        df = pd.read_parquet(self.input_data_path)

        if self.task_type=='regression':
            df[self.target] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
            df[self.target]  = df[self.target].apply(lambda td: td.total_seconds() / 60)
            df = df[(df[self.target]  >= 1) & (df[self.target]  <= 60)].reset_index(drop=True) #suggested online  
        elif self.task_type=='classification':
            df = df[~pd.isna(df['trip_type'])]
            df[self.target] = df.apply(lambda x: 'A' if x['trip_type']==1 else 'B', axis=1)
            df = df.drop(columns=['trip_type'], axis=1)
    
        try:
            Y = df[self.target]
        except:
            if request_tgt == True:
                raise TypeError("Target variable not found")
            else:
                Y=None
        X = df.drop(columns=[self.target] , axis=1, errors='ignore')
          
        return X,Y


    def preprocess_for_regression(self, df: pd.DataFrame, enable_categorical: bool = False,  fit_ohe: bool = False, drop_first_column: bool = False, ohe: OneHotEncoder = None):

        '''
        - enable_categorical: 
            False: categorical variables will be one hot encoded; 
            True: they will be converted as 'category' type (suitable for boosting algs that manage cat variales (es catboost or 
            xgboost with enable_categorical=True ))
        - fit_ohe: wether to fit a ohe. OSS: This parameter has effect only if enable_categorical=False
        - drop_first_column: remove the first column of the OHE representation (to avoid multicollinearity for unregularized linear models). OSS: This parameter has effect only if enable_categorical=False
        - ohe: 
            pretrained one hot encoder, used when fit_ohe=False (use ohe=True for preporcessing in a scoring pipeline)
            OSS: This parameter has effect only if enable_categorical=False
        '''

        # Pickup dates preprocessing
        df['lpep_pickup_datetime_week'] = df['lpep_pickup_datetime'].dt.week
        df['lpep_pickup_datetime_day'] = df['lpep_pickup_datetime'].dt.day
        df['lpep_pickup_datetime_hour'] = df['lpep_pickup_datetime'].dt.hour
        df['lpep_pickup_datetime_minute'] = df['lpep_pickup_datetime'].dt.minute
        df['lpep_pickup_datetime_dayofweek'] = df['lpep_pickup_datetime'].dt.dayofweek

        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

        categorical_cols = [
                'PU_DO',
                'store_and_fwd_flag']

        numerical_cols = ['passenger_count', 
                'lpep_pickup_datetime_week',
                'lpep_pickup_datetime_day',
                'lpep_pickup_datetime_hour',
                'lpep_pickup_datetime_minute',
                'lpep_pickup_datetime_dayofweek']


        #missing imputation
        numeric_means = df[numerical_cols].mean()
        categ_modes = df[categorical_cols].mode().iloc[0]

        df = df.fillna(numeric_means).fillna(categ_modes)

        if enable_categorical:
            df[categorical_cols] = df[categorical_cols].astype('category')
            X = df[numerical_cols+categorical_cols]

        else:
            # one hot encoding 
            if fit_ohe:
                if drop_first_column:
                    ohe = OneHotEncoder(
                            drop='first',
                            handle_unknown='ignore', 
                            sparse = False)
                else:
                    ohe = OneHotEncoder(
                            handle_unknown='ignore',
                            sparse=False)

                ohe.fit(df[categorical_cols])
                cat_ohe = ohe.transform(df[categorical_cols]) # it is an array, convert it in df with column names
                ohe_df = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = categorical_cols), index = df.index)
                X = pd.concat([df[numerical_cols], ohe_df], axis=1)

            else:
                cat_ohe = ohe.transform(df[categorical_cols]) # it is an array, convert it in df with column names
                ohe_df = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = categorical_cols), index = df.index)
                X = pd.concat([df[numerical_cols], ohe_df], axis=1)

        return X, ohe


    def preprocess_for_classification(self, df: pd.DataFrame, enable_categorical: bool = False,  fit_ohe: bool = False, drop_first_column: bool = False, ohe: OneHotEncoder = None):

        '''
        - enable_categorical: 
            False: categorical variables will be one hot encoded; 
            True: they will be converted as 'category' type (suitable for boosting algs that manage cat variales (es catboost or 
            xgboost with enable_categorical=True ))
        - fit_ohe: wether to fit a ohe. OSS: This parameter has effect only if enable_categorical=False
        - drop_first_column: remove the first column of the OHE representation (to avoid multicollinearity for unregularized linear models). OSS: This parameter has effect only if enable_categorical=False
        - ohe: 
            pretrained one hot encoder, used when fit_ohe=False (use ohe=True for preporcessing in a scoring pipeline)
            OSS: This parameter has effect only if enable_categorical=False
        '''

        # Pickup dates preprocessing
        df['lpep_pickup_datetime_week'] = df['lpep_pickup_datetime'].dt.week
        df['lpep_pickup_datetime_day'] = df['lpep_pickup_datetime'].dt.day
        df['lpep_pickup_datetime_hour'] = df['lpep_pickup_datetime'].dt.hour
        df['lpep_pickup_datetime_minute'] = df['lpep_pickup_datetime'].dt.minute
        df['lpep_pickup_datetime_dayofweek'] = df['lpep_pickup_datetime'].dt.dayofweek

        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

        categorical_cols = [
                'PU_DO',
                'store_and_fwd_flag',
                'RatecodeID',
                'payment_type']

        numerical_cols = ['passenger_count', 
                'trip_distance',
                'fare_amount',
                'extra',
                'mta_tax',
                'improvement_surcharge',
                'tip_amount',
                'tolls_amount',
                'total_amount',
                'lpep_pickup_datetime_week',
                'lpep_pickup_datetime_day',
                'lpep_pickup_datetime_hour',
                'lpep_pickup_datetime_minute',
                'lpep_pickup_datetime_dayofweek']


        #missing imputation
        numeric_means = df[numerical_cols].mean()
        categ_modes = df[categorical_cols].mode().iloc[0]

        df = df.fillna(numeric_means).fillna(categ_modes)

        if enable_categorical:
            df[categorical_cols] = df[categorical_cols].astype('category')
            X = df[numerical_cols+categorical_cols]

        else:
            # one hot encoding 
            if fit_ohe:
                if drop_first_column:
                    ohe = OneHotEncoder(
                            drop='first',
                            handle_unknown='ignore', 
                            sparse = False)
                else:
                    ohe = OneHotEncoder(
                            handle_unknown='ignore',
                            sparse=False)

                ohe.fit(df[categorical_cols])
                cat_ohe = ohe.transform(df[categorical_cols]) # it is an array, convert it in df with column names
                ohe_df = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = categorical_cols), index = df.index)
                X = pd.concat([df[numerical_cols], ohe_df], axis=1)

            else:
                cat_ohe = ohe.transform(df[categorical_cols]) # it is an array, convert it in df with column names
                ohe_df = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = categorical_cols), index = df.index)
                X = pd.concat([df[numerical_cols], ohe_df], axis=1)

        return X, ohe





