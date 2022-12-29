#AUTOLOG: this pgm is identical to 03_sklearn_raw but 
# autolog is used: automatically logs all metrics of model + some tags + model artifact
# all other log_metrics and log_params (only model params) have been commented, also log_model.
#as you can see, at least for sklearn, automated logging infos are not so complete, especially the history of metrics is missing. Alsso, probably too many params are logged,
# I personally prefer to log only handled params, not the default ones
#OSS:
#If no active run exists when autolog() captures data, MLflow will automatically create a run to log information to. Also, 
#MLflow will then automatically end the run once training ends via calls to tf.estimator.train(), tf.keras.fit(), tf.keras.fit_generator(), 
#keras.fit() or keras.fit_generator(), or once tf.estimator models are exported via tf.estimator.export_saved_model().
#
#If a run already exists when autolog() captures data, MLflow will log to that run but not automatically end that run after training.

#new york taxi data used (https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)
#goal: predict trip duration given a set of (a priori chosen) trip features
# I chose these data because they have monthly updates, similar to ISP case

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pickle import dump
import mlflow


year_month = '2022-01'
raw_data_path = './data/green_tripdata_'+year_month+'.parquet'
seme = 123
#set parameters for 3 experiments (oss: I will log only the explicitly specified parameters, autolog would log all parameters, see following script)
parameters = [{'n_estimators' : 5,'max_depth' : 1, 'random_state' : seme}
        ,{'n_estimators' : 10,'max_depth' : 2, 'random_state' : seme}
        ,{'n_estimators' : 5,'max_depth' : 5, 'random_state' : seme}
        ]
exp_name = year_month
path_save='./local_artifacts_tmp/' #local path where to save objects to log as artifacts


mlflow.set_tracking_uri("sqlite:///mlflow.db")
experiment_id = mlflow.set_experiment(exp_name)


def read_dataframe(filename: str):

    df = pd.read_parquet(filename)

    df['duration'] = df.lpep_dropoff_datetime - df.lpep_pickup_datetime
    df.duration = df.duration.apply(lambda td: td.total_seconds() / 60)
    df = df[(df.duration >= 1) & (df.duration <= 60)].reset_index(drop=True) #suggested online  

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)

    return df


def preprocess(df: pd.DataFrame,  fit_ohe: bool = False, ohe: OneHotEncoder = None):

    '''
    ohe is used as input only if fit_ohe=False, otherwise it is initialized
    '''

    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']
    categorical_cols = ['PU_DO']
    numerical_cols = ['trip_distance']

    #missing imputation
    numeric_means = df[numerical_cols].mean()
    categ_modes = df[categorical_cols].mode().iloc[0]

    df = df.fillna(numeric_means).fillna(categ_modes)

    # one hot encoding 
    if fit_ohe:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse = False)
        ohe.fit(df[categorical_cols])
        cat_ohe = ohe.transform(df[categorical_cols]) # it is an array, convert it in df with column names
        ohe_df = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = categorical_cols), index = df.index)
        X = pd.concat([df[numerical_cols], ohe_df], axis=1)

    else:
        cat_ohe = ohe.transform(df[categorical_cols]) # it is an array, convert it in df with column names
        ohe_df = pd.DataFrame(cat_ohe, columns = ohe.get_feature_names_out(input_features = categorical_cols), index = df.index)
        X = pd.concat([df[numerical_cols], ohe_df], axis=1)

    return X, ohe

def train_gb_evaluation(gb, params, X_train, X_test, Y_train, Y_test, Y_pred_train, Y_pred_test, draw = False):

    rmse_train = mean_squared_error(Y_train,Y_pred_train)**0.5
    rmse_test = mean_squared_error(Y_test,Y_pred_test)**0.5
    
    #track history
    history_rmse_train=[None for i in range(params["n_estimators"])]
    history_rmse_test=[None for i in range(params["n_estimators"])]
    for i,Y_pred_train in enumerate(gb.staged_predict(X_train)):
        history_rmse_train[i] = mean_squared_error(Y_train, Y_pred_train)**0.5
        #mlflow.log_metric('history_rmse_train', history_rmse_train[i], step=i) # with autolog this is done automatically
    for i,Y_pred_test in enumerate(gb.staged_predict(X_test)):
        history_rmse_test[i] = mean_squared_error(Y_test, Y_pred_test)**0.5
        #mlflow.log_metric('history_rmse_test', history_rmse_test[i], step=i)

    # example of saving a plot as artifact (oss: it is under experimentation the possibility to save custom metric (graphs, df ertc., see mlflow.evaluate))
    x=range(params["n_estimators"])
    plt.plot(x,list(history_rmse_train),label='train_acc')
    plt.plot(x,list(history_rmse_test),label='test_acc')
    plt.legend()

    plt.savefig(path_save+'iteration_plot.png')
    mlflow.log_artifact(local_path = path_save+"iteration_plot.png")

    if draw:
        plt.show()
    plt.close()

    return rmse_train, rmse_test



def main():

    df = read_dataframe(raw_data_path)

    Y = df['duration']
    X = df.drop(['duration'],axis=1)

    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=seme)

    #preprocessing
    shapes_pre = (X_train.shape[0], X_test.shape[0])
    X_train, ohe = preprocess(df=X_train, fit_ohe=True)
    X_test, _ = preprocess(df=X_test, fit_ohe=False, ohe=ohe)
    assert shapes_pre == (X_train.shape[0], X_test.shape[0])
    dump(ohe, open(path_save+'ohe.pkl', 'wb'))
    mlflow.sklearn.autolog()
    for params in parameters:
        with mlflow.start_run() as run:
            
            mlflow.set_tag('model_type','GradientBoostingRegressor')
            mlflow.set_tag('year_month',year_month)
            mlflow.log_param('data',raw_data_path)
            #mlflow.log_params(params)

            gb = GradientBoostingRegressor(**params, verbose = 1)
            gb.fit(X_train, Y_train)
            Y_pred_train=gb.predict(X_train)
            Y_pred_test=gb.predict(X_test)
            rmse_train, rmse_test = train_gb_evaluation(gb, params, X_train, X_test, Y_train, Y_test, Y_pred_train, Y_pred_test)

            #mlflow.log_metrics({'rmse_train':rmse_train,'rmse_test':rmse_test})
            print('rmse_train = ', rmse_train, '\n rmse_test', rmse_test)
            
            mlflow.log_artifact(local_path = path_save+"ohe.pkl",)       
            #mlflow.sklearn.log_model(gb,  artifact_path='model')

if __name__ == '__main__':
    main()



#next: [autolog], gridsearch o simili, model registry, monitoring