# replace grid search whit hyper opt (bayesan approach --> http://hyperopt.github.io/hyperopt/  https://www.phdata.io/blog/bayesian-hyperparameter-optimization-with-mlflow/)
#in this way each experiment will have its own run (no nested/child runs). It's just a matter of preference

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pickle import dump
import numpy as np
import mlflow

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from hyperopt.pyll import scope
from functools import partial

import sklearn

year_month = '2022-01'
raw_data_path = './data/green_tripdata_'+year_month+'.parquet'
seme = 123

# unlike sklearn grid search, you can put both 'fixed' and 'variable' parameters in the same object
search_space = {
    'n_estimators' : 5,
    'validation_fraction':0.1, 
    'n_iter_no_change':5,  
    'random_state' : seme,
    'max_depth': hp.quniform('max_depth', 3, 5, 1),
    'learning_rate': hp.uniform('learning_rate', 0.05, 0.1),
    'loss' : hp.choice('loss',['squared_error','absolute_error'])
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

def objective(params, X_train, X_test, Y_train, Y_test):
    # oss: params is not something that I can call explictly: when calling fmin, it automatically pass the params for the search. IF,as in this case, I want to pass other parameters to the
    # objective function other than params, I have to use partial call of params.
    #you could call single elements of params as params['max_depth'] etc.

    # no nested runs: if I want nested runs, I have to wrap fmin in a with mlrun.start_run(), and below add nested=True
    with mlflow.start_run():
        mlflow.set_tag('model_type','GradientBoostingRegressor')
        mlflow.set_tag('year_month',year_month)
        mlflow.log_param('data',raw_data_path)
        mlflow.log_params(params)

        gb = GradientBoostingRegressor(**params)

        gb.fit(X_train, Y_train)

        Y_pred_train=gb.predict(X_train)
        Y_pred_test=gb.predict(X_test)
        rmse_train, rmse_test = gb_evaluation(gb, params, X_train, X_test, Y_train, Y_test, Y_pred_train, Y_pred_test)

        mlflow.log_metrics({'rmse_train':rmse_train,'rmse_test':rmse_test})
        print('rmse_train = ', rmse_train, '\n rmse_test', rmse_test)

        mlflow.log_artifact(local_path = path_save+"ohe.pkl") 
        mlflow.sklearn.log_model(gb,  artifact_path='model')
    
    #I'm minimizing test score. I could also minimize cross val score (above I would need to do cross_val_score(gb, X_train, Y_train, cv=10, scoring='neg_root_mean_squared_error', njobs=-1).mean() e loss = -neg_root_mean_squared_error )
    return {'loss': rmse_test, 'status': STATUS_OK}



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

    trials = Trials() # to track the iterations of hyperopt. I don't really need it since i'm using MLFlow for tracking
    best_result = fmin(
        fn=partial(objective, X_train = X_train, X_test = X_test, Y_train = Y_train, Y_test= Y_test), # or only objective if the objective function doesn't have other parameters but params
        space=search_space,
        algo=tpe.suggest,
        max_evals=5,
        trials=trials,
        rstate = np.random.default_rng(seme)
    )

    # if I want to show the results stored in trials i can do the following (oss, having mlflow it is not necessary)
    print('best_result \n',best_result, '\n')
    for trial in trials:
        print(trial)
        print(trial['misc']['vals'],' ',trial['result']['loss']) #oss nei parametri di hp.coiche (max depth nel mio caso)
            # non ti da il vero valore ma l'indice relatvo!!! https://github.com/hyperopt/hyperopt/issues/800 , https://github.com/hyperopt/hyperopt/issues/761#issuecomment-718004328
            #cmq le soluzioni dei link sopra non funzionano. Ma ancora, ho mlflow che risolve il problema

if __name__ == '__main__':
    main()

