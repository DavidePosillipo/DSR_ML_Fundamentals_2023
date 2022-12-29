#new york taxi data used (https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)
#goal: predict trip duration given a set of (a priori chosen) trip features
# GRID SEARCH SKLEARN: a parent run is created with best model (also model artifacts are saved. Dublicated best_model and model, but they are the same)
#child runs for each grid search are created. No artifacts are saved for these runs
#I don't like the fact that child runs are created (one for each grid search combination) and that the parent run has different set of valued parameters and metrics with respect to its child runs
# grid search with autolog + custom metrics for best model
#I don't like the fact that child runs are created (one for each grid search combination) and that the parent run has different set of valued parameters and metrics.
# if you want to ssave all grid search attempts, you need to heavily program this behaviour. I didn't do it. See hyperopt for an easier approach
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pickle import dump
import mlflow
from sklearn.model_selection import GridSearchCV

import sklearn

year_month = '2022-01'
raw_data_path = './data/green_tripdata_'+year_month+'.parquet'
seme = 123
#set fixed parameters + set of gs parameters
params = {'n_estimators' : 5, 'validation_fraction':0.1, 'n_iter_no_change':5,  'random_state' : seme}
parameters_gs = {
            'max_depth' : (1,2)
            ,'learning_rate' : (0.05,0.1)
            }
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

def gb_evaluation(gb, params, X_train, X_test, Y_train, Y_test, Y_pred_train, Y_pred_test, draw = False):

    rmse_train = mean_squared_error(Y_train,Y_pred_train)**0.5
    rmse_test = mean_squared_error(Y_test,Y_pred_test)**0.5
    
    #track history
    history_rmse_train=[None for i in range(params["n_estimators"])]
    history_rmse_test=[None for i in range(params["n_estimators"])]
    for i,Y_pred_train in enumerate(gb.staged_predict(X_train)):
        history_rmse_train[i] = mean_squared_error(Y_train, Y_pred_train)**0.5
        mlflow.log_metric('history_rmse_train', history_rmse_train[i], step=i) # with autolog this is done automatically
    for i,Y_pred_test in enumerate(gb.staged_predict(X_test)):
        history_rmse_test[i] = mean_squared_error(Y_test, Y_pred_test)**0.5
        mlflow.log_metric('history_rmse_test', history_rmse_test[i], step=i)

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


    with mlflow.start_run() as run:
        mlflow.set_tag('model_type','GradientBoostingRegressor')
        mlflow.set_tag('year_month',year_month)
        mlflow.log_param('data',raw_data_path)
        mlflow.log_params(params)

        gb = GradientBoostingRegressor(**params)
        # Instantiate g_s
        mlflow.sklearn.autolog()
        g_s = GridSearchCV(estimator=gb, 
                            param_grid=parameters_gs,
                            scoring='neg_root_mean_squared_error',
                            cv=10,
                            n_jobs=-1,
                             verbose = 2)

        g_s.fit(X_train, Y_train)

        print(g_s.cv_results_)

        #add the same metrics of 03_sklearnraw for the best model (which is the parent run)

        best_model = g_s.best_estimator_

        Y_pred_train=best_model.predict(X_train)
        Y_pred_test=best_model.predict(X_test)
        rmse_train, rmse_test = gb_evaluation(best_model, params, X_train, X_test, Y_train, Y_test, Y_pred_train, Y_pred_test)

        mlflow.log_metrics({'rmse_train':rmse_train,'rmse_test':rmse_test})
        print('rmse_train = ', rmse_train, '\n rmse_test', rmse_test)
        
        mlflow.log_artifact(local_path = path_save+"ohe.pkl",)       
        #mlflow.sklearn.log_model(gb,  artifact_path='model')

if __name__ == '__main__':
    main()



#next: autolog, gridsearch o somili, model registry, monitoring