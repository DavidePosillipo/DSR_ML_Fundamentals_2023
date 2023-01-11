import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from pickle import dump
from hyperopt import STATUS_OK

from scripts.Training import Training

class BinaryClassificationTraining(Training):

    def objective_logistic_regression(self,
            params,
            X_train,
            X_test,
            Y_train,
            Y_test,
            run_name: str = 'Unnamed',
            threshold=0.5):
        '''
        Fitting function for Logistic Regression
        '''
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag('model_type', 'LogisticRegression')
            mlflow.set_tag('year_month', self.year_month)

            mlflow.log_param('model_type', 'LogisticRegression')
            mlflow.log_param('data', self.input_data_path)
            mlflow.log_param('threshold', threshold)
            mlflow.log_params(params)

            cl = LogisticRegression(**params)

            cl.fit(X_train, Y_train)

            Y_pred_train_prob = cl.predict_proba(X_train)
            Y_pred_test_prob = cl.predict_proba(X_test)

            
            cl_metrics = self.classification_evaluation(
                Y_train=Y_train, 
                Y_test=Y_test, 
                Y_pred_train_prob=Y_pred_train_prob, 
                Y_pred_test_prob=Y_pred_test_prob,
                threshold=threshold)

            mlflow.log_metrics(cl_metrics)

            mlflow.log_artifact(
                    local_path = self.local_path_save + run_name + '_ohe.pkl',
                    artifact_path='preprocessing'
                )
            mlflow.log_artifact(
                    local_path = self.local_path_save + run_name + '_scaler.pkl',
                    artifact_path='preprocessing'
                )

            mlflow.sklearn.log_model(cl, artifact_path='model')

            return {'loss': cl_metrics['roc_auc_test'], 'status': STATUS_OK}

