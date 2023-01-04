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
from scripts.config_regression import *


class Preprocess:


    def __init__(self, input_data_path):
        self.input_data_path = input_data_path
        self.target = target


    def read_dataframe(self, request_tgt):

        df = pd.read_parquet(self.input_data_path)

        df[self.target] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
        df[self.target]  = df[self.target].apply(lambda td: td.total_seconds() / 60)
        df = df[(df[self.target]  >= 1) & (df[self.target]  <= 60)].reset_index(drop=True) #suggested online  

        #categorical = ['PULocationID', 'DOLocationID']
        #df[categorical] = df[categorical].astype(str)

        try:
            Y = df[self.target]
        except:
            if request_tgt == True:
                raise TypeError("Target variable not found")
            else:
                Y=None
        X = df.drop(columns=[self.target] , axis=1, errors='ignore')

        return X,Y


    def preprocess(self, df: pd.DataFrame, enable_categorical: bool = False,  fit_ohe: bool = False, drop_first_column: bool = False, ohe: OneHotEncoder = None):

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

        df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
        categorical_cols = ['PU_DO']
        numerical_cols = ['trip_distance']

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
