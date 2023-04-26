import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from pickle import dump
from hyperopt import STATUS_OK

import xgboost as xgb

from scripts.Training import Training
from scripts.config import xgb_params_fit


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


    def objective_decision_tree(self,
            params,
            X_train,
            X_test,
            Y_train,
            Y_test,
            run_name: str = 'Unnamed',
            threshold=0.5):
        '''
        Fitting function for Classification Tree
        '''
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag('model_type', 'ClassificationTree')
            mlflow.set_tag('year_month', self.year_month)

            mlflow.log_param('model_type', 'ClassificationTree')
            mlflow.log_param('data', self.input_data_path)
            mlflow.log_param('threshold', threshold)
            mlflow.log_params(params)

            cl = DecisionTreeClassifier(**params)

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


    def objective_random_forest(self,
            params,
            X_train,
            X_test,
            Y_train,
            Y_test,
            run_name: str = 'Unnamed',
            threshold=0.5):
        '''
        Fitting function for the Random Forest Classifier
        '''

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag('model_type', 'RandomForestClassifier')
            mlflow.set_tag('year_month', self.year_month)

            mlflow.log_param('model_type', 'RandomForestClassifier')
            mlflow.log_param('data', self.input_data_path)
            mlflow.log_param('threshold', threshold)
            mlflow.log_params(params)

            cl = RandomForestClassifier(**params)

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



    def objective_gradient_boosting(self,
            params,
            X_train,
            X_test,
            Y_train,
            Y_test,
            run_name: str = 'Unnamed',
            threshold=0.5):
        '''
        Fitting function for the GB Classifier
        '''

        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag('model_type', 'GradientBoostingClassifier')
            mlflow.set_tag('year_month', self.year_month)

            mlflow.log_param('model_type', 'GradientBoostingClassifier')
            mlflow.log_param('data', self.input_data_path)
            mlflow.log_param('threshold', threshold)
            mlflow.log_params(params)

            cl = GradientBoostingClassifier(**params)

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



    def objective_xgb(self,
            params,
            X_train,
            Y_train,
            X_test,
            Y_test,
            save_ohe = True,
            run_name: str = 'Unnamed',
            threshold=0.5):
        '''
        Fitting function for the XGB classifier. 
        It uses the xgb implementation
        '''
        with mlflow.start_run(run_name=run_name):
            mlflow.set_tag('model_type','XGboostClassifier')
            mlflow.set_tag('year_month',self.year_month)
            mlflow.log_param('model_type','XGboostClassifier')
            mlflow.log_param('data',self.input_data_path)
            mlflow.log_params(params)
            mlflow.log_params(xgb_params_fit)

            cl = xgb.XGBClassifier(**params)
            cl.fit(X_train, Y_train, **xgb_params_fit, eval_set=[(X_train, Y_train), (X_test, Y_test)])

            Y_pred_train_prob = cl.predict_proba(X_train)
            Y_pred_test_prob = cl.predict_proba(X_test)

            cl_metrics = self.classification_evaluation(
                Y_train=Y_train, 
                Y_test=Y_test, 
                Y_pred_train_prob=Y_pred_train_prob, 
                Y_pred_test_prob=Y_pred_test_prob,
                threshold=threshold)

            mlflow.log_metrics(cl_metrics)

            #save ohe only if available
            if save_ohe:
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


    def objective_SVM(self):
        '''
        Fitting function for the Support Vector Machine Classifier. 
        It uses the sklearn implementation. 
        '''

        # TODO write here your implementation

        pass


