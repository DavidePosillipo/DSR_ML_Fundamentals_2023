import mlflow
import os
from scripts.training import Training
from scripts.model_registry import ModelRegistry
from scripts.scoring import Scoring
from scripts.monitoring import Monitoring
from scripts.config import *



if not os.path.exists(local_path_save):
    os.makedirs(local_path_save)
#save all metadata in a sqlite db. Artifacts will be saved on local folder ./mlflow    
mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment_id = mlflow.set_experiment(exp_name)



## Model training
#Train one or more ml models with fixed or grid-like parameters (define desired parameters in scripts/config.py file. Currently only random forest 'rf' and gradient boosting 'gb' are supported)
#All results will be tracked in the MLFlow UI (from a terminal positioned on the root run the command 'mlflow ui --backend-store-uri sqlite:///mlflow.db', then browse to http://127.0.0.1:5000/)
models = models
train = Training(input_data_path_train, local_path_save, year_month_train)
train.preprocess_and_train(models)


#comment part below if you only want to do experiment tracking

## Model registry management
#Identify the best(*) run and store the relative model as 'Production' model. 
#Archive former model from 'Production' to 'Archived'
#(*) model with lowest test error that doesn't overfit the data: (train -test)/train < of_treshold
model_reg = ModelRegistry(exp_name, year_month_train, model_name_pref)
model_reg.register_best_run(of_treshold=0.1)
#Archive predious model (from Production to Archived)
model_reg.archive_models(year_month_monitor)



## Scoring 
#Score latest available data using the Production model identified above, save scored data
scoring = Scoring(year_month_train, model_name_pref, local_path_save)
scoring.preprocess_and_predict(input_data_path_to_score, scored_data_path)



## Monitoring
#Evaluate the performance of the previous month model (the archived one), comparing its prediction with the now available observed target
if year_month_train != '2022-01': #if it is the first passage to production we don't have anything to monitor
    monitor = Monitoring(input_data_path_to_score, scored_data_path, model_name_pref, year_month_monitor, local_path_save)
    monitor.monitor()
