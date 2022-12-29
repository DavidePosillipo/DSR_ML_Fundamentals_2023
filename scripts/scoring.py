from pickle import load
from mlflow.tracking import MlflowClient
import mlflow
from scripts.training_regression_models import Preprocess
import pandas as pd


class Scoring:
    """
    preprocess and score data using production model and preprocessor relative to production model run
    """

    def __init__(self, year_month, model_name_pref, local_path_save):
        self.model_name = model_name_pref + year_month
        self.local_path_save = local_path_save
        self.client = MlflowClient()


    def get_preprocessing_artifacts(self):
        '''
        download preprocessing artifact(s) and load it in memory
        '''

        #get run_id of the latest version of model in production (by construction, last version should always be 1)
        last_version = self.client.get_latest_versions(name = self.model_name, stages = ['Production'])
        run_id_last_version = last_version[0].run_id

        #download ohe artifact and load it in memory
        self.client.download_artifacts(run_id_last_version,"preprocessing/ohe.pkl",self.local_path_save) 
        ohe = load(open(self.local_path_save+'ohe.pkl', 'rb'))
        return ohe
    

    def get_production_model(self):
        '''
        load in memory last version of the production model
        '''
        return mlflow.pyfunc.load_model(f"models:/{self.model_name}/Production")


    def preprocess_and_predict(self, input_data_path, scored_data_path):
        '''
        preprocess data and score them with the production model
        '''

        prepr = Preprocess(input_data_path)
        X, Y = prepr.read_dataframe(request_tgt=False)

        #preprocessing
        ohe = self.get_preprocessing_artifacts()
        shape_pre = X.shape[0]
        X, _ = prepr.preprocess(df=X, fit_ohe=False, ohe=ohe)
        assert shape_pre == X.shape[0]
        
        model = self.get_production_model()
        Y_pred = model.predict(X)

        #from sklearn.metrics import mean_squared_error
        #print(mean_squared_error(Y,Y_pred)**0.5)


        print(f'Saving predictions in  {scored_data_path} ')
        pd.DataFrame({'Y_pred':Y_pred}).to_pickle(scored_data_path)

        return Y_pred
