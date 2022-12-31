#maybe define here hp
from hyperopt import hp
import numpy as np
seed = 123


## Set monitor, train and prediction snapshots
year_month_pred = '2022-02'
year_month_train = '2022-01'
year_month_monitor = '2021-12'

## Data paths(don't edit)
#data to train the model with
input_data_path_train = './data/input/green_tripdata_'+year_month_train+'.parquet'
#data to score
input_data_path_to_score = './data/input/green_tripdata_'+year_month_pred+'.parquet'
#where to save the scored predictions
scored_data_path = './data/prediction/green_tripdata_'+year_month_pred+'_pred.pkl'


## MLflow variables
#exp_name = 'NY taxi - level 0'
model_name_pref = 'NY'
#local path where to save and download objects and artifacts
local_path_save='./local_artifacts_tmp/' 


#target variable name
target = 'duration'


models=['lr', 'gb', 'rf', 'xgb']
#models=[ 'xgb']

#### Linear Regression with no regularization ####
lr_parameters_search = {'fit_intercept':True}
lr_max_evals = 1

##########################################################
######## sklearn boosting  parameters ####################
##########################################################

gb_max_evals = 5 #max number of experiments in hyperopt

gb_parameters_search = {
    'n_estimators' : 5,
    'validation_fraction':0.1, 
    'n_iter_no_change':5, #early stopping
    'random_state' : seed,
    'max_depth': hp.quniform('max_depth', 3, 5, 1),
    'learning_rate': hp.uniform('learning_rate', 0.05, 0.1)
}

'''


gb_max_evals = 1
gb_parameters_search = {
    'n_estimators' : 5,
    'validation_fraction':0.1, 
    'n_iter_no_change':5,  
    'random_state' : seed,
    'max_depth': 3,
    'learning_rate': 0.05
}
'''


##########################################################
######## sklearn rf  parameters ##########################
##########################################################
rf_max_evals = 5 #max number of experiments in hyperopt

rf_parameters_search = {
    'n_estimators' : 5,
    'min_samples_leaf' : 0.05,
    "random_state" : seed,
    'n_jobs' : -1,
    "max_features" : hp.uniform('max_features', 0.2, 0.6), 
    'max_depth' : hp.quniform('max_depth', 3, 5, 1)
}

'''
rf_parameters_search = {
    'n_estimators' : 5,
    'min_samples_leaf' : 0.05,
    "random_state" : seed,
    'n_jobs' : -1,
    "max_features" : 0.1, 
    'max_depth' : 3
}
'''

##########################################################
######## xgboost  parameters #############################
##########################################################

xgb_max_evals = 5

#early stopping on eval metric calculated in last item of 'eval_set' (inside xgb.fit)
xgb_params_fit = {"verbose" : 1}  #0 --> don't print training steps metrics

xgboost_parameters_search = {
    "objective":'reg:squarederror', # necessario se non uso api sklearn, altrimenti con xgb.XGBClassifier è già il suo defaulrt
    "tree_method": "hist",#"gpu_hist", # con altri metodi, categorical_model non funziona 
    "n_estimators" : 5,
    "eval_metric":"rmse", 
    #"early_stopping_rounds " : 5, #. Cannot be inserted here, but separately in the fit()
    "max_depth" : hp.choice('max_depth', np.arange(3, 5, dtype=int)), #xgb alllows only int max_depth (max_depth=1.0 will raise an error)
    'eta': hp.uniform('eta', 0.05, 0.5),
    #"colsample_bylevel": 0.7,# stocastic gb (at each split(level) randomly choose between 70% of features )
    'enable_categorical' : True, #if categorical data are present (i.e. fetures with pandas type='Category') they are managed, If they are not present and 'enable_categorical=True --> no problem
    "early_stopping_rounds" : 5,
    "seed":seed
}

