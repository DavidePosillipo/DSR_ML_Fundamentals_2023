from pickle import load
from mlflow.tracking import MlflowClient
import mlflow
from scripts.training_regression_models import Preprocess
import pandas as pd
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

class Monitoring:
    """
    Compare predictions with observed target, if the observed target is available
    """

    def __init__(self, input_data_path, scored_data_path, model_name_pref, year_month, local_path_save):
        self.input_data_path = input_data_path
        self.scored_data_path = scored_data_path
        self.model_name = model_name_pref + year_month
        self.local_path_save = local_path_save
        self.client = MlflowClient()


    def read_observed_target(self):
        '''
        read observed target
        '''
        prepr = Preprocess(self.input_data_path)
        X, Y = prepr.read_dataframe(request_tgt=True)
        return Y

    def read_predicted_target(self):
        '''
        read predicted target
        '''
        return pd.read_pickle(self.scored_data_path)

    def monitor(self):
        '''
        compare predictions with observed target and save results in mlflow, modifying the run_id associated to the previous month production model
        '''
        print('model name to monitor =', self.model_name)

        Y_obs = self.read_observed_target()
        Y_pred = self.read_predicted_target()
        rmse_obs = mean_squared_error(Y_obs, Y_pred)**0.5
        print('rmse_obs',rmse_obs)

        #retrieve the expected rmse (i.e. the generalization error rmse_test) calculated during training 
        #get run_id of the latest version of Archived model (by construction, last version should always be 1)
        last_version = self.client.get_latest_versions(name = self.model_name, stages = ['Archived'])
        last_version_run_id = last_version[0].run_id
        run_last_version = self.client.get_run(last_version_run_id)
        rmse_exp = run_last_version.data.metrics['rmse_test']
        print('rmse_exp',rmse_exp)


        pl = plt.bar(['rmse_observed','rmse_expected'],[rmse_obs,rmse_exp])
        pl[0].set_color('g')
        pl[1].set_color('r')
        plt.bar_label(pl)
        plt.savefig(self.local_path_save+'metrics_monitoring.png')
        self.client.log_artifact(run_id=last_version_run_id, local_path=self.local_path_save+'/metrics_monitoring.png', artifact_path='monitor')
        plt.close()
        #self.client
        #rmse_exp = run_last_version

        
        

