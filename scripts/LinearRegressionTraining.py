import mlflow

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso 
from sklearn.linear_model import ElasticNet 

from hyperopt import STATUS_OK

from scripts.Training import Training

class LinearRegressionTraining(Training):

	def objective_lr(self, X_train, X_test, Y_train, Y_test, run_name: str = 'Unnamed'):
		'''
		Fitting function for Linear Regression
		'''
		with mlflow.start_run(run_name=run_name):	
			mlflow.set_tag('model_type', 'LinearRegression')
			mlflow.set_tag('year_month', self.year_month)
			
			mlflow.log_param('model_type','LinearRegression')
			mlflow.log_param('data', self.input_data_path)
			
			lr = LinearRegression()
			
			lr.fit(X_train, Y_train)
			
			Y_pred_train = lr.predict(X_train)
			Y_pred_test = lr.predict(X_test)
			rmse_train, rmse_test, mae_train, mae_test = self.regression_evaluation(
					Y_train, Y_test, Y_pred_train, Y_pred_test
			)
			
			mlflow.log_metrics({'rmse_train': rmse_train, 
								'rmse_test': rmse_test,
								'mae_train': mae_train,
								'mae_test': mae_test})
			
			mlflow.log_artifact(local_path = self.local_path_save + run_name + '_ohe.pkl', artifact_path='preprocessing') 
			mlflow.sklearn.log_model(lr, artifact_path='model')
			
		return {'loss': rmse_test, 'status': STATUS_OK} 
		

	def objective_lr_ridge(self, params, X_train, X_test, Y_train, Y_test, run_name: str = 'Unnamed'):
		'''
		Fitting function for Linear Regression with 
		Ridge Regularization
		'''
		with mlflow.start_run(run_name=run_name):
		    mlflow.set_tag('model_type', 'RidgeLinearRegression')
		    mlflow.set_tag('year_month', self.year_month)
		    mlflow.log_param('model_type', 'RidgeLinearRegression')
		    mlflow.log_param('data', self.input_data_path)
		    mlflow.log_params(params)
		    
		    lr = Ridge(**params)
		    
		    lr.fit(X_train, Y_train)
		    
		    Y_pred_train = lr.predict(X_train)
		    Y_pred_test = lr.predict(X_test)
		    rmse_train, rmse_test, mae_train, mae_test = self.regression_evaluation(
		    		Y_train, Y_test, Y_pred_train, Y_pred_test
		    )
		    
		    mlflow.log_metrics({'rmse_train': rmse_train, 
		    					'rmse_test': rmse_test,
		    					'mae_train': mae_train,
		    					'mae_test': mae_test})
		    
		    mlflow.log_artifact(local_path = self.local_path_save + run_name + '_ohe.pkl', artifact_path='preprocessing') 
		    mlflow.sklearn.log_model(lr, artifact_path='model')

		return {'loss': rmse_test, 'status': STATUS_OK}
	
	
	def objective_lr_lasso(self, params, X_train, X_test, Y_train, Y_test, run_name: str = 'Unnamed'):
	    '''
	    Fitting function for Linear Regression with 
	    Lasso Regularization
	    '''
	    with mlflow.start_run(run_name=run_name):
	        mlflow.set_tag('model_type', 'LassoLinearRegression')
	        mlflow.set_tag('year_month', self.year_month)
	        mlflow.log_param('model_type', 'LassoLinearRegression')
	        mlflow.log_param('data', self.input_data_path)
	        mlflow.log_params(params)
	        
	        lr = Lasso(**params)
	        
	        lr.fit(X_train, Y_train)
	        
	        Y_pred_train = lr.predict(X_train)
	        Y_pred_test = lr.predict(X_test)
	        rmse_train, rmse_test, mae_train, mae_test = self.regression_evaluation(
	        		Y_train, Y_test, Y_pred_train, Y_pred_test
	        )
	        
	        mlflow.log_metrics({'rmse_train': rmse_train, 
	        					'rmse_test': rmse_test,
	        					'mae_train': mae_train,
	        					'mae_test': mae_test})
	        
	        mlflow.log_artifact(local_path = self.local_path_save + run_name + '_ohe.pkl', artifact_path='preprocessing') 
	        mlflow.sklearn.log_model(lr, artifact_path='model')

	    return {'loss': rmse_test, 'status': STATUS_OK}
	
	
	def objective_lr_elastic_net(self, params, X_train, X_test, Y_train, Y_test, run_name: str = 'Unnamed'):
	    '''
	    Fitting function for Linear Regression with 
	    Elastic Net Regularization
	    '''
	    with mlflow.start_run(run_name=run_name):
	        mlflow.set_tag('model_type', 'ElasticNetLinearRegression')
	        mlflow.set_tag('year_month', self.year_month)
	        mlflow.log_param('model_type', 'ElasticNetLinearRegression')
	        mlflow.log_param('data', self.input_data_path)
	        mlflow.log_params(params)
	        
	        lr = ElasticNet(**params)
	        
	        lr.fit(X_train, Y_train)
	        
	        Y_pred_train = lr.predict(X_train)
	        Y_pred_test = lr.predict(X_test)
	        rmse_train, rmse_test, mae_train, mae_test = self.regression_evaluation(
	        		Y_train, Y_test, Y_pred_train, Y_pred_test
	        )
	        
	        mlflow.log_metrics({'rmse_train': rmse_train, 
	        					'rmse_test': rmse_test,
	        					'mae_train': mae_train,
	        					'mae_test': mae_test})
	        
	        mlflow.log_artifact(local_path = self.local_path_save + run_name + '_ohe.pkl', artifact_path='preprocessing') 
	        mlflow.sklearn.log_model(lr, artifact_path='model')

	    return {'loss': rmse_test, 'status': STATUS_OK}
	
