import mlflow
from mlflow.tracking import MlflowClient
import datetime

class ModelRegistry:


    def __init__(self, exp_name, year_month, model_name_pref):
        self.exp_name = exp_name
        self.year_month = year_month
        self.model_name_pref = model_name_pref
        self.client = MlflowClient()
        self.of_treshold = None


    def get_best_run(self, check_of=False):
        '''
            get the best run (smallest rmse test) whit a defined treshold of overfitting (10%) (only if check_of=True)
        '''
        
        experiment = self.client.get_experiment_by_name(self.exp_name)

        #sort eligible runs for rmse_test ascending
        eligible_runs = self.client.search_runs(
            experiment_ids=experiment.experiment_id,
            run_view_type=mlflow.entities.ViewType.ACTIVE_ONLY,
            max_results=5,
            filter_string=f"tags.year_month = '{self.year_month}'",
            order_by=["metrics.rmse_test ASC"],
        )

        #keep the first non overfitting run
        if check_of:
            for run in eligible_runs:
                if abs(run.data.metrics['rmse_train'] - run.data.metrics['rmse_test'])/run.data.metrics['rmse_train'] < self.of_treshold:
                    return run
        else:
            return eligible_runs[0]


    def register_best_run(self, of_treshold, year_month=None):
        '''
            Register the best run model in the model registry and stage it in production
        '''
        self.of_treshold = of_treshold

        if year_month == None:
            year_month  = self.year_month

        best_run = self.get_best_run()
        # register best models
        model_name = self.model_name_pref + year_month
        #delete model if exists (useful if for some reason the script is launched several times)
        try:
            self.client.delete_registered_model(
                name=model_name
            )  # delete all versions of the model. # to delete single versions client.delete_model_version(name=model_name, version=version)
            print('previous models found: deleting them')
        except:
            pass

        run_id = best_run.info.run_id
        model_uri = f"runs:/{run_id}/model"
        mlflow.register_model(model_uri, model_name)

        #register last version of model_name (since we deleted model_name aboce, last version should always be 1)
        model_versions = self.client.search_model_versions(f"name = '{model_name}'")
        last_version = model_versions[-1].version

        print(f'registering version {last_version} of model {model_name} into Production stage')

        self.client.transition_model_version_stage(
            name=model_name, version=last_version, stage="Production"
        )

        self.client.update_model_version(
            name=model_name,
            version=last_version,
            description=f"transitioned to Production on {datetime.datetime.now()}",
        )  


    def archive_models(self, year_month=None):
        '''
            Archive models of previous year_month
        '''     

        if year_month == None:
            year_month  = self.year_month
        model_name = self.model_name_pref + year_month

        #we exept (see register_best_run()) only version 1 to be present (or [] if it is the first passage to production of the script)
        for model_version in self.client.search_model_versions(f"name = '{model_name}'"):

            print(f'Archiving version {model_version.version} of model {model_name} from Production stage')

            self.client.transition_model_version_stage(
                name=model_name, version=model_version.version, stage="Archived"
            )

            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=f"transitioned to Archived on {datetime.datetime.now()}",
            )



