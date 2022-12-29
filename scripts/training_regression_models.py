# replace grid search whit hyper opt (bayesan approach --> http://hyperopt.github.io/hyperopt/  https://www.phdata.io/blog/bayesian-hyperparameter-optimization-with-mlflow/)

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
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
        print('df shape',df.shape)
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


    def preprocess(self, df: pd.DataFrame, enable_categorical: bool = False,  fit_ohe: bool = False, ohe: OneHotEncoder = None):

        '''
        - enable_categorical: 
            False: categorical variables will be one hot encoded; 
            True: they will be converted as 'category' type (suitable for boosting algs that manage cat variales (es catboost or 
            xgboost with enable_categorical=True ))
        - fit_ohe: wether to fit a ohe. OSS: This parameter has effect only if enable_categorical=False
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
                ohe = OneHotEncoder(handle_unknown='ignore', sparse = False)
                ohe.fit(df[categorical_cols])
                cat_ohe = ohe.transform(df[categorical_cols]) # it is an array, convert it in df with column names
                ohe_df = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = categorical_cols), index = df.index)
                X = pd.concat([df[numerical_cols], ohe_df], axis=1)

            else:
                cat_ohe = ohe.transform(df[categorical_cols]) # it is an array, convert it in df with column names
                ohe_df = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = categorical_cols), index = df.index)
                X = pd.concat([df[numerical_cols], ohe_df], axis=1)

        return X, ohe

class Training:
    '''
    Class for importing, preprocessing and training data, keeping track of preprocessing artifacts, training parameters, performance and artifacts with mlflow
    '''

    def __init__(self, input_data_path, local_path_save, year_month):
        
        self.input_data_path = input_data_path
        self.local_path_save = local_path_save
        self.year_month = year_month


    def gb_evaluation(self, gb, params, X_train, X_test, Y_train, Y_test, Y_pred_train, Y_pred_test, draw = False):

        rmse_train = mean_squared_error(Y_train,Y_pred_train)**0.5
        rmse_test = mean_squared_error(Y_test,Y_pred_test)**0.5
        
        #track history
        history_rmse_train=[None for i in range(params["n_estimators"])]
        history_rmse_test=[None for i in range(params["n_estimators"])]
        for i,Y_pred_train in enumerate(gb.staged_predict(X_train)):
            history_rmse_train[i] = mean_squared_error(Y_train, Y_pred_train)**0.5
            mlflow.log_metric('history_rmse_train', history_rmse_train[i], step=i) # with autolog this is done automatically
        for i,Y_pred_test in enumerate(gb.staged_predict(X_test)):
            history_rmse_test[i] = mean_squared_error(Y_test, Y_pred_test)**0.5
            mlflow.log_metric('history_rmse_test', history_rmse_test[i], step=i)

        # example of saving a plot as artifact (oss: it is under experimentation the possibility to save custom metric (graphs, df ertc., see mlflow.evaluate))
        x=range(params["n_estimators"])
        plt.plot(x,list(history_rmse_train),label='train_rmse')
        plt.plot(x,list(history_rmse_test),label='test_rmse')
        plt.legend()

        plt.savefig(self.local_path_save+'iteration_plot.png')
        mlflow.log_artifact(local_path = self.local_path_save+"iteration_plot.png", artifact_path='training_info')

        if draw:
            plt.show()
        plt.close()

        return rmse_train, rmse_test

    def rf_evaluation(self, Y_train, Y_test, Y_pred_train, Y_pred_test):

        rmse_train = mean_squared_error(Y_train,Y_pred_train)**0.5
        rmse_test = mean_squared_error(Y_test,Y_pred_test)**0.5
        
        return rmse_train, rmse_test
    

    def lr_evaluation(self, Y_train, Y_test, Y_pred_train, Y_pred_test):
        '''
        Function for evaluation of Linear Regression goodness
        '''
        rmse_train = mean_squared_error(Y_train,Y_pred_train)**0.5
        rmse_test = mean_squared_error(Y_test,Y_pred_test)**0.5
        
        return rmse_train, rmse_test


    def xgb_evaluation(self, gb, Y_train, Y_test, Y_pred_train, Y_pred_test, draw = False):
 
        rmse_train = mean_squared_error(Y_train,Y_pred_train)**0.5
        rmse_test = mean_squared_error(Y_test,Y_pred_test)**0.5

        results = gb.evals_result()

        history_rmse_train =  np.array(results['validation_0']['rmse'])
        history_rmse_test = np.array(results['validation_1']['rmse'])

        for i, val in enumerate(history_rmse_train):
            mlflow.log_metric("history_rmse_train", val, step=i) 
        for i, val in enumerate(history_rmse_test):
            mlflow.log_metric("history_rmse_test", val, step=i) 
        #[metrics = (mlflow.entities.Metric('history_rmse_train',val,i) for i,val in enumerate(history_rmse_train))
        #MlflowClient().log_batch(mlflow.active_run().info.run_id, metrics=mlflow.entities.Metric('history_rmse_train',10,1))

        x = range(len(history_rmse_train))
        plt.plot(x,list(history_rmse_train),label='train_rmse')
        plt.plot(x,list(history_rmse_test),label='test_rmse')
        plt.legend()

        plt.savefig(self.local_path_save+'iteration_plot.png')
        mlflow.log_artifact(local_path = self.local_path_save+"iteration_plot.png", artifact_path='training_info')

        if draw:
            plt.show()
        plt.close()

        return rmse_train, rmse_test


    def objective_gb(self, params, X_train, X_test, Y_train, Y_test):
        # oss: params is not something that I can call explictly: when calling fmin, it automatically pass the params for the search. IF,as in this case, I want to pass other parameters to the
        # objective function other than params, I have to use partial call of params.
        #you could call single elements of params as params['max_depth'] etc.

        with mlflow.start_run():
            mlflow.set_tag('model_type','GradientBoostingRegressor')
            mlflow.set_tag('year_month',self.year_month)
            mlflow.log_param('model_type','GradientBoostingRegressor')
            mlflow.log_param('data',self.input_data_path)
            mlflow.log_params(params)

            gb = GradientBoostingRegressor(**params)

            gb.fit(X_train, Y_train)

            Y_pred_train=gb.predict(X_train)
            Y_pred_test=gb.predict(X_test)
            rmse_train, rmse_test = self.gb_evaluation(gb, params, X_train, X_test, Y_train, Y_test, Y_pred_train, Y_pred_test)

            mlflow.log_metrics({'rmse_train':rmse_train,'rmse_test':rmse_test})
            print('\n rmse_train = ', rmse_train, 'rmse_test', rmse_test)

            mlflow.log_artifact(local_path = self.local_path_save+"ohe.pkl", artifact_path="preprocessing") 
            mlflow.sklearn.log_model(gb,  artifact_path='model')
        
        #I'm minimizing test score. I could also minimize cross val score (above I would need to do cross_val_score(gb, X_train, Y_train, cv=10, scoring='neg_root_mean_squared_error', njobs=-1).mean() e loss = -neg_root_mean_squared_error )
        return {'loss': rmse_test, 'status': STATUS_OK}

    def objective_rf(self, params, X_train, X_test, Y_train, Y_test):
        # oss: params is not something that I can call explictly: when calling fmin, it automatically pass the params for the search. IF,as in this case, I want to pass other parameters to the
        # objective function other than params, I have to use partial call of params.
        #you could call single elements of params as params['max_depth'] etc.

        with mlflow.start_run():
            mlflow.set_tag('model_type','RandomForestRegressor')
            mlflow.set_tag('year_month',self.year_month)
            mlflow.log_param('model_type','RandomForestRegressor')
            mlflow.log_param('data',self.input_data_path)
            mlflow.log_params(params)

            rf = RandomForestRegressor(**params)

            rf.fit(X_train, Y_train)

            Y_pred_train=rf.predict(X_train)
            Y_pred_test=rf.predict(X_test)
            rmse_train, rmse_test = self.rf_evaluation(Y_train, Y_test, Y_pred_train, Y_pred_test)

            mlflow.log_metrics({'rmse_train':rmse_train,'rmse_test':rmse_test})
            print('rmse_train = ', rmse_train, '\n rmse_test', rmse_test)

            mlflow.log_artifact(local_path = self.local_path_save+"ohe.pkl", artifact_path='preprocessing') 
            mlflow.sklearn.log_model(rf,  artifact_path='model')
        
        #I'm minimizing test score. I could also minimize cross val score (above I would need to do cross_val_score(gb, X_train, Y_train, cv=10, scoring='neg_root_mean_squared_error', njobs=-1).mean() e loss = -neg_root_mean_squared_error )
        return {'loss': rmse_test, 'status': STATUS_OK}


    def objective_lr(self, X_train, X_test, Y_train, Y_test):
        '''
        Fitting function for Linear Regression
        '''
        with mlflow.start_run():
            mlflow.set_tag('model_type','LinearRegression')
            mlflow.set_tag('year_month',self.year_month)
            mlflow.log_param('model_type','LinearRegression')
            mlflow.log_param('data',self.input_data_path)
            #mlflow.log_params(params)

            lr = LinearRegression()

            lr.fit(X_train, Y_train)

            Y_pred_train = lr.predict(X_train)
            Y_pred_test = lr.predict(X_test)
            rmse_train, rmse_test = self.lr_evaluation(Y_train, Y_test, Y_pred_train, Y_pred_test)

            mlflow.log_metrics({'rmse_train':rmse_train, 'rmse_test':rmse_test})
            print('rmse_train = ', rmse_train, '\n rmse_test', rmse_test)

            mlflow.log_artifact(local_path = self.local_path_save+"ohe.pkl", artifact_path='preprocessing') 
            mlflow.sklearn.log_model(lr, artifact_path='model')
        
        #I'm minimizing test score. I could also minimize cross val score (above I would need to do cross_val_score(gb, X_train, Y_train, cv=10, scoring='neg_root_mean_squared_error', njobs=-1).mean() e loss = -neg_root_mean_squared_error )
        return {'loss': rmse_test, 'status': STATUS_OK}


    def objective_xgb(self, params, X_train, X_test, Y_train, Y_test):
        # oss: params is not something that I can call explictly: when calling fmin, it automatically pass the params for the search. IF,as in this case, I want to pass other parameters to the
        # objective function other than params, I have to use partial call of params.
        #you could call single elements of params as params['max_depth'] etc.

        with mlflow.start_run():
            mlflow.set_tag('model_type','XGboostRegressor')
            mlflow.set_tag('year_month',self.year_month)
            mlflow.log_param('model_type','XGboostRegressor')
            mlflow.log_param('data',self.input_data_path)
            mlflow.log_params(params)
            mlflow.log_params(xgb_params_fit)

            gb = xgb.XGBRegressor(**params)
            gb.fit(X_train, Y_train, **xgb_params_fit, eval_set=[(X_train, Y_train), (X_test, Y_test)])

            Y_pred_train=gb.predict(X_train)
            Y_pred_test=gb.predict(X_test)
            rmse_train, rmse_test = self.xgb_evaluation(gb, Y_train, Y_test, Y_pred_train, Y_pred_test)

            mlflow.log_metrics({'rmse_train':rmse_train,'rmse_test':rmse_test})
            print('\n rmse_train = ', rmse_train, 'rmse_test', rmse_test)

            mlflow.log_artifact(local_path = self.local_path_save+"ohe.pkl", artifact_path="preprocessing") 
            mlflow.sklearn.log_model(gb,  artifact_path='model')#I could use mlflow.sklearn.log_model, but it gives problems if I'm using categorical data(enable_categorical=True)
        
        #I'm minimizing test score. I could also minimize cross val score (above I would need to do cross_val_score(gb, X_train, Y_train, cv=10, scoring='neg_root_mean_squared_error', njobs=-1).mean() e loss = -neg_root_mean_squared_error )
        return {'loss': rmse_test, 'status': STATUS_OK}


    def preprocess_and_train(self, models):

        prepr = Preprocess(self.input_data_path)
        X, Y = prepr.read_dataframe(request_tgt=True)

        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=seed)

        #preprocessing ohe
        shapes_pre = (X_train.shape[0], X_test.shape[0])
        X_train_ohe, ohe = prepr.preprocess(df=X_train, fit_ohe=True)
        X_test_ohe, _ = prepr.preprocess(df=X_test, fit_ohe=False, ohe=ohe)
        assert shapes_pre == (X_train.shape[0], X_test.shape[0])
        dump(ohe, open(self.local_path_save+'ohe.pkl', 'wb'))

        #preprocessing enable_categorical
        shapes_pre = (X_train.shape[0], X_test.shape[0])
        X_train_cat, _ = prepr.preprocess(df=X_train, enable_categorical = True)
        X_test_cat, _ = prepr.preprocess(df=X_test, enable_categorical = True)
        assert shapes_pre == (X_train.shape[0], X_test.shape[0])

        for model in models:

            trials = Trials() # to track the iterations. I don't really need it since i'm using MLFlow for tracking


            if model == 'gb':
                print("###################### training ", model, 'model #########################')
                best_result = fmin(
                    fn=partial(self.objective_gb, X_train = X_train_ohe, X_test = X_test_ohe, Y_train = Y_train, Y_test= Y_test), # or only objective if the obfective funion doesn't have othe parameters but params
                    space=gb_parameters_search,
                    algo=tpe.suggest,
                    max_evals=gb_max_evals,
                    trials=trials,
                    rstate = np.random.default_rng(seed)
                )
            
            if model == 'rf':
                print("###################### training ", model, 'model #########################')
                best_result = fmin(
                    fn=partial(self.objective_rf, X_train = X_train_ohe, X_test = X_test_ohe, Y_train = Y_train, Y_test= Y_test), # or only objective if the obfective funion doesn't have othe parameters but params
                    space=rf_parameters_search,
                    algo=tpe.suggest,
                    max_evals=rf_max_evals,
                    trials=trials,
                    rstate = np.random.default_rng(seed)
                )

            if model == 'lr':
                print("###################### training ", model, 'model #########################')
                best_result = self.objective_lr(X_train = X_train_ohe,
                        X_test = X_test_ohe,
                        Y_train = Y_train,
                        Y_test= Y_test)    

            if model == 'xgb':
                print("###################### training ", model, 'model #########################')

                if xgboost_parameters_search['enable_categorical'] == True:
                    X_train_xgb = X_train_cat
                    X_test_xgb = X_test_cat
                else:
                    X_train_xgb = X_train_ohe
                    X_test_xgb = X_test_ohe

                best_result = fmin(
                    fn=partial(self.objective_xgb, X_train = X_train_xgb, X_test = X_test_xgb, Y_train = Y_train, Y_test= Y_test), # or only objective if the obfective funion doesn't have othe parameters but params
                    space=xgboost_parameters_search,
                    algo=tpe.suggest,
                    max_evals=xgb_max_evals,
                    trials=trials,
                    rstate = np.random.default_rng(seed)
                )

        return best_result



#TODO: ADD VARIABLE IMPORTANCE IN GB and RF
