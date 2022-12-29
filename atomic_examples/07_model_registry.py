# https://docs.databricks.com/applications/mlflow/model-registry-example.html
#for the following script to work, you need some experiments already run (e.g. run 06_sklearn_hyperopt.py)
from distutils import archive_util
import mlflow
from mlflow.tracking import MlflowClient
import datetime

mlflow.set_tracking_uri("sqlite:///mlflow.db")
year_month = "2022-01"
exp_name = year_month


client = MlflowClient()

# grab best two models
experiment = client.get_experiment_by_name(
    exp_name
)  # potevo usare anche mlflow.get_experiment_by_name
print("experiment_id", experiment.experiment_id)
runs = client.search_runs(
    experiment_ids=experiment.experiment_id,
    run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
    max_results=2,
    order_by=["metrics.rmse_test ASC"],
)

#FAREI: REGISTRA BANALMENTE IL MODELLO MOGLIORE E METTILO IN PRODUZIONE
#POI ARCHIVIA IL VECCHIO

# register best models
model_name = "NY" + year_month
client.delete_registered_model(
    name=model_name
)  # delete all versions of the model. DO it if you lunch this script several times (you can't register a model if it already exists)
# to delete single versions client.delete_model_version(name=model_name, version=version)
for run in runs:
    run_id = run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, model_name)

# convert all registered models in staging and add description
for model_version in client.search_model_versions(f"name = '{model_name}'"):
    print(f" \nmodel version = {model_version.version}")
    print(f" \n full model log \n {model_version}")
    client.transition_model_version_stage(
        name=model_name, version=model_version.version, stage="Staging"
    )
    client.update_model_version(
        name=model_name,
        version=model_version.version,
        description=f"transitioned to Staging on {datetime.datetime.now()}",
    )

#in the model registry tab, you will see one model, if you click it you will see two versions for that model with relative descriptions. 
#Playing with the api, you can transition models in production or archive them