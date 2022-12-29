# https://www.mlflow.org/docs/latest/python_api/mlflow.tracking.html
import mlflow
import os

from sklearn.datasets import load_iris
from pickle import load
import pandas as pd

mlflow.set_tracking_uri("sqlite:///mlflow.db") # comment if in 01 you commented it

local_dir = './local_models_download/' #dir where to download artifacts/models from mlflow
if not os.path.exists(local_dir):
    os.mkdir(local_dir)

################################ access closed mlflow runs data with the tracking api ##################################################
print('\n','#'*100,'\n retrieve experiment \n', '#'*100, '\n')  
from mlflow.tracking import MlflowClient

client = MlflowClient()

experiment = client.get_experiment_by_name("Iris-sklearn") # potevo usare anche mlflow.get_experiment_by_name
print('experiment_id',experiment.experiment_id)
runs = client.search_runs(
    experiment_ids = experiment.experiment_id,
    filter_string = "metrics.accuracy_test > 0.9", #filtro di esempio
    run_view_type = mlflow.entities.ViewType.ACTIVE_ONLY,
    max_results = 3,
    order_by = ["metrics.accuracy_test DESC"]
)

print('\n extracted runs \n\n', runs)

print('\n best run metric \n',runs[0].data.metrics)

#we can download artifacts of the  best run
best_run_id = runs[0].info.run_id
client.download_artifacts(best_run_id,"scaler.pkl",local_dir) 
client.download_artifacts(best_run_id,"model/model.pkl",local_dir) #remote path INSIDE artifact_uri. It can be seen in UI. OSS: folder structure is mantained also in local downloads

#i don't need to download models, but I can directly load them from MLFlow model section
#model_uri can be clled in different ways: artifact_uri/model or runs:/run_id/model. The rwo calls below give the same result
#model = mlflow.sklearn.load_model(runs[0].info.artifact_uri+'/model')
model = mlflow.pyfunc.load_model(f"runs:/{best_run_id}"+'/model') # o mlflow.sklearn.load_model ...

#check if downloaded models and models read from MLFlow are the same: YES
X, Y = load_iris(return_X_y=True, as_frame=True)
scaler = load(open(local_dir+'scaler.pkl', 'rb'))
downloaded_model = load(open(local_dir+'model/model.pkl', 'rb'))
X=pd.DataFrame(scaler.transform(X),columns=X.columns)
Y_pred = model.predict(X)
Y_pred_local = downloaded_model.predict(X)
assert (Y_pred==Y_pred_local).all()

############################################################################################################
# add metrics / params / artifacts to existing runs. 
# useful if, e.g., we want to add a monitoring metric or file to a run closed last month and put in production
##########################################################################################################
with mlflow.start_run(run_id=best_run_id) as run: #run id is unique among experiments, you don't need to insert experiment_id
    mlflow.log_param("new_param",1000)
    mlflow.log_metric("new_metric",42.0) # your corrected metrics
# as an alternative you can use tracking api client.log_metric(run_id=best_run_id, "new_metric",42.0)

#N.B: If you try to log an existing metric/param/artifact you will get an error, unless you use nested runs --> in the UI you see them with the + button. (it is a child run: https://gist.github.com/smurching/366781ae6a3e5d597d716ef30cf26ba8)
# personally I would avoid nested runs if not strictly necessary
with mlflow.start_run(run_id=best_run_id) as run:
    with mlflow.start_run(nested=True, experiment_id = experiment.experiment_id): # without experiment id it would go in default experiment!!
        mlflow.log_param("new_param",2000)
        mlflow.log_metric("new_metric",84.0) # your corrected metrics

    #to access run_id of child runs, knowing the run_id of the parent, you can do
    #filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run_id)
    #runs = mlflow.search_runs(filter_string=filter_child_runs)

# TODO: programmatically delete all failed runs