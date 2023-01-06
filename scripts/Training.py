from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import (accuracy_score,
        precision_score,
        recall_score,
        roc_auc_score,
        f1_score)
import numpy as np
import mlflow
from hyperopt import STATUS_OK

from scripts.config import positive_label


class Training:

    def __init__(self, input_data_path, local_path_save, year_month, model_name):

        self.input_data_path = input_data_path
        self.local_path_save = local_path_save
        self.year_month = year_month
        self.model_name = model_name


    def set_hyperparameter_space(self, hyperparameters_space):
        self.hp_space = hyperparameters_space


    def regression_evaluation(self, Y_train, Y_test, Y_pred_train, Y_pred_test):
        '''
        Function for evaluation of regression models results 
        '''
        rmse_train = mean_squared_error(Y_train, Y_pred_train)**0.5
        rmse_test = mean_squared_error(Y_test, Y_pred_test)**0.5
        
        mae_train = mean_absolute_error(Y_train, Y_pred_train)
        mae_test = mean_absolute_error(Y_test, Y_pred_test)
        
        return rmse_train, rmse_test, mae_train, mae_test
        
        
    def classification_evaluation(self, Y_train, Y_test, Y_pred_train, Y_pred_test):
        '''
        Function for evaluation of binary classification models results
        '''
        accuracy_train = accuracy_score(Y_train, Y_pred_train)
        accuracy_test = accuracy_score(Y_test, Y_pred_test)

        precision_train = precision_score(Y_train, Y_pred_train, pos_label=positive_label)
        precision_test = precision_score(Y_test, Y_pred_test, pos_label=positive_label)

        recall_train = recall_score(Y_train, Y_pred_train, pos_label=positive_label)
        recall_test = recall_score(Y_test, Y_pred_test, pos_label=positive_label)

        #roc_auc_train = roc_auc_score(Y_train, Y_pred_train)
        #roc_auc_test = roc_auc_score(Y_test, Y_pred_test)

        f1_score_train = f1_score(Y_train, Y_pred_train, pos_label=positive_label)
        f1_score_test = f1_score(Y_test, Y_pred_test, pos_label=positive_label)

        results = {'accuracy_train': accuracy_train,
                'accuracy_test': accuracy_test,
                'precision_train': precision_train,
                'precision_test': precision_test,
                'recall_train': recall_train,
                'recall_test': recall_test,
                #'roc_auc_train': roc_auc_train,
                #'roc_auc_test': roc_auc_test,
                'f1_score_train': f1_score_train,
                'f1_score_test': f1_score_test}

        return results 


    def calculate_classification_baseline(self, Y_train, Y_test, run_name: str = 'Unnamed'):
        '''
        Fitting function for binary classification baseline
        '''
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag('model_type', 'ClassificationBaseline')
            mlflow.set_tag('year_month', self.year_month)
            mlflow.log_param('model_type', 'ClassificationBaseline')
            mlflow.log_param('data', self.input_data_path)

            most_frequent_class = Y_train.mode()[0]
            Y_pred_train = np.repeat(most_frequent_class, Y_train.shape[0])
            Y_pred_test = np.repeat(most_frequent_class, Y_test.shape[0])

            baseline_metrics = self.classification_evaluation(
                    Y_train, Y_test, Y_pred_train, Y_pred_test
            )

            mlflow.log_metrics(baseline_metrics)

            print(baseline_metrics)

            mlflow.log_artifact(local_path = self.local_path_save + run_name + '_ohe.pkl', artifact_path='preprocessing') 
            mlflow.sklearn.log_model(most_frequent_class, artifact_path='model')
        
        return {'loss': baseline_metrics['accuracy_test'], 'status': STATUS_OK}
    


    def calculate_regression_baseline(self, Y_train, Y_test, run_name: str = 'Unnamed'):
        '''
        Fitting function for Baseline
        '''
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag('model_type', 'RegressionBaseline')
            mlflow.set_tag('year_month', self.year_month)
            mlflow.log_param('model_type', 'RegressionBaseline')
            mlflow.log_param('data', self.input_data_path)

            average_y_train = Y_train.mean()

            Y_pred_train = np.repeat(average_y_train, Y_train.shape[0])
            Y_pred_test = np.repeat(average_y_train, Y_test.shape[0])
            rmse_train, rmse_test, mae_train, mae_test = self.regression_evaluation(
                    Y_train, Y_test, Y_pred_train, Y_pred_test
            )

            mlflow.log_metrics({'rmse_train': rmse_train, 
                                'rmse_test': rmse_test,
                                'mae_train': mae_train,
                                'mae_test': mae_test})

            print('rmse_train = ', rmse_train, 
                  '\n rmse_test', rmse_test,
                  '\n mae_train', mae_train,
                  '\n mae_test', mae_test)

            mlflow.log_artifact(local_path = self.local_path_save + run_name + '_ohe.pkl', artifact_path='preprocessing') 
            mlflow.sklearn.log_model(average_y_train, artifact_path='model')
        
        return {'loss': rmse_test, 'status': STATUS_OK}
    
