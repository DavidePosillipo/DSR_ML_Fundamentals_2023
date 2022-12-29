#grid search example from mlflow mlflow/example
# a parent run is created with best model
#child runs for each grid search are created. No artifacts are saved for these runs
#I don't like the fact that child runs are created (one for each grid search combination) and that the parent run has different set of valued parameters and metrics with respect to its child runs

from pprint import pprint

import pandas as pd
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
import pickle

import mlflow

local_dir = './local_models_download'
exp_name = 'Iris-sklearn'
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(experiment_name=exp_name)

def yield_artifacts(run_id, path=None):
    """Yield all artifacts in the specified run"""
    client = mlflow.tracking.MlflowClient()
    for item in client.list_artifacts(run_id, path):
        if item.is_dir:
            yield from yield_artifacts(run_id, item.path)
        else:
            yield item.path


def fetch_logged_data(run_id):
    """Fetch params, metrics, tags, and artifacts in the specified run"""
    client = mlflow.tracking.MlflowClient()
    data = client.get_run(run_id).data
    print('experiment',client.get_run(run_id).info.experiment_id )
    # Exclude system tags: https://www.mlflow.org/docs/latest/tracking.html#system-tags
    #tags = {k: v for k, v in data.tags.items() if not k.startswith("mlflow.")}
    tags = {k: v for k, v in data.tags.items()}
    artifacts = list(yield_artifacts(run_id))
    return {
        "params": data.params,
        "metrics": data.metrics,
        "tags": tags,
        "artifacts": artifacts,
    }


def main():
    mlflow.sklearn.autolog()

    iris = datasets.load_iris()
    parameters = {"kernel": ("rbf", "linear" ), "C": [1, 10]}
    svc = svm.SVC()
    clf = GridSearchCV(svc, parameters)

    clf.fit(iris.data, iris.target)
    run_id = mlflow.last_active_run().info.run_id
    print('run_id',run_id)

    # show data logged in the parent run
    print("========== parent run ==========")
    for key, data in fetch_logged_data(run_id).items():
        print("\n---------- logged {} ----------".format(key))
        pprint(data)

    # show data logged in the child runs
    filter_child_runs = "tags.mlflow.parentRunId = '{}'".format(run_id)
    runs = mlflow.search_runs(filter_string=filter_child_runs)
    param_cols = ["params.{}".format(p) for p in parameters.keys()]
    metric_cols = ["metrics.mean_test_score"]

    print("\n========== child runs ==========\n")
    pd.set_option("display.max_columns", None)  # prevent truncating columns
    print(runs[["run_id", *param_cols, *metric_cols]])





if __name__ == "__main__":
    main()


    ############ grid search salva di default 2 modelli nella parent run: best_estimator e model: sono uguali? SI'
    iris = datasets.load_iris()
    from mlflow.tracking import MlflowClient
    client = MlflowClient()
    print(mlflow.last_active_run())
    run_id = mlflow.last_active_run().info.run_id
    print('run_id',run_id)
    client.download_artifacts(run_id,"model",local_dir) 
    client.download_artifacts(run_id,"best_estimator",local_dir)
    best_estimator = pickle.load(open(local_dir+'/best_estimator/model.pkl', 'rb'))
    model = pickle.load(open(local_dir+'/model/model.pkl', 'rb'))
    y_best = best_estimator.predict(iris.data)
    y_model = model.predict(iris.data)
    assert (y_best == y_model).all()
    