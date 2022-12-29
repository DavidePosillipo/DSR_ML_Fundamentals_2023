import mlflow

#mlflow.set_tracking_uri("sqlite:///mlflow.db")
with mlflow.start_run() as run:
    for i in range(10):
        mlflow.log_metric(key='mse', value=i*10, step=i)