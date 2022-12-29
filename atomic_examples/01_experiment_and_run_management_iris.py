#simple scenario (default): everything (metadata and artifacts) is installed on local filesystem as 'simple' files,
#without a sqlalchemy backend

#to access the UI run on terminal, positioned above the folder ./mlruns (created by default where the python script is run)
#mlflow ui
#and navigate with a browser to the specified path
#if it doesn't update, delete cookies
import mlflow

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pandas as pd
from pickle import dump,load
import os

path_save='./local_artifacts_tmp/' #local path where to save objects to log as artifacts

if not os.path.exists(path_save):
    os.makedirs(path_save)
seme = 123
exp_name = 'Iris-sklearn' #experiment name


# https://www.mlflow.org/docs/latest/tracking.html
# Scenario 1: se voglio che tutti i dati (parametri, dati, metriche, artifacts..) vengano salvati in ./mlruns non devo fare set_tracking_uri. E per vedere la UI
# mi basta, da terminale eseguire "mlflow ui" e aprire il link generato.
# Scenario 2: Se voglio (consigliato, altrimenti non funziona il model registry) che gli artifact vengano salvati in ./mlruns e tutto il resto in db sql-like (es sqlite) sempre su locale ./mlflow.db  
# allora devo fare set_tracking_uri e per accedere all'UI da terminale devo eseguire "mlflow ui --backend-store-uri sqlite:///mlflow.db" (o altri sqlalchemy db, come myswql e postgres)
mlflow.set_tracking_uri("sqlite:///mlflow.db")


print('\n','#'*100,'\n create experiment \n', '#'*100, '\n')
try:    
    #creo esperimento (il relativo experiment_id non è settabile, è incrementale). Se esiste già va in errore
    experiment_id = mlflow.create_experiment(exp_name) #l'assegnazione non era necessaria, potevo fare solo mlflow.create_experiment(exp_name)
    #setto esperimento: così di default il mlflow.start_run() mette le run nell'esperimento settato (altrimenti devo fare mlflow.start_run(experiment_id='..'))
    #se non setto l'esperimento, mlflow.start_run() scrive nell'experiment Default, il cui id è '0'
    #oss: in realtà se setti e l'esperimento non è creato, lo crea comunque
    mlflow.set_experiment(experiment_name=exp_name)
except:
    mlflow.set_experiment(experiment_name=exp_name)
    # o se voglio settarlo da id
    experiment = mlflow.get_experiment_by_name(exp_name)
    print('\n experiment \n', experiment)
    experiment_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=experiment_id)


print('\n', '#'*100,'\n run experiment \n', '#'*100, '\n')  
#START A RUN: 
# OSS1 with the autologs, start-run is not necessary
# OSS2 sometimes it is not necessary to explicitly start a run (e.g. with log_metric() it is started automatically but as a best practice, 
# it is better to always use start_run() 
# OSS3: each time you run the python script, a new experiment is created under the default experiment_id='0' (see ./mlruns/0/...)
#first way
#mlflow.start_run()
#.. code ..
#mlflow.end_run()

#second way
with mlflow.start_run() as run:

    ################################# run general infos ############################################################################
    #get current run infos
    print('\n general run infos \n',mlflow.active_run().info)
    #or easier using the run named 'run' (unlike above, it can be accessed also outside the with statement)
    print('\n general run infos v2 \n',run.info) #run.info.experiment_id or run.info.run_id ... to call single elements of info
    #OSS: run_uuid is a deprecation of run_id

    #get current tracking uri ("./mlruns" by default, where '.' is the path from which you launched the python command)
    print('\n current tracking uri: \n', mlflow.tracking.get_tracking_uri())

    #get path where to save artifacts
    print('\n current artifact uri: \n', mlflow.get_artifact_uri())
    #or equivalently using the infos
    print('\n current artifact uri v2: \n',run.info.artifact_uri)
 
    ############################## train and monitor a simple model ################################################################
    X, Y = load_iris(return_X_y=True, as_frame=True)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=seme)
    #standardizz le var
    scaler = preprocessing.StandardScaler().fit(X_train)
    X_train=pd.DataFrame(scaler.transform(X_train),columns=X_train.columns)
    X_test=pd.DataFrame(scaler.transform(X_test),columns=X_train.columns)

    params = {"C": 0.1, "random_state": seme}
    #loggo i parametri (posso mettere ciò che voglio, anche la versione dei dati usati, se ha senso)
    mlflow.log_params(params)

    #metto dei tag (es. modello)
    mlflow.set_tag('model_type','LogisticRegression')

    lr = LogisticRegression(**params).fit(X_train, Y_train)
    Y_pred_train = lr.predict(X_train)
    Y_pred_test = lr.predict(X_test)

    #carico le metriche (per modelli più complessi posso caricare metriche ad ogni step/epoch)
    mlflow.log_metric("accuracy_train", accuracy_score(Y_train, Y_pred_train))
    mlflow.log_metric("accuracy_test", accuracy_score(Y_test, Y_pred_test))

    #carico artifacts per riprodurre i modello e/o fare inferenza (nel mio semplice caso lo scaler per il preproc e il modello sklearn)
    #mlflow.log_artifacts: carica un file già SALVATO SU LOCALE nella  artifact_uri (default, oppure in un'altra artifact_path da speciifcare come argomento di log_artifact)
    dump(scaler, open(path_save+'scaler.pkl', 'wb'))
    mlflow.log_artifact(local_path = path_save+"scaler.pkl",)

    #potrei fare stessa cosa per il modello (salvo su locale + log_artifact). Per modelli però è preferibile salvarli con log_model (disponibile per una vasta gamma di librerie di ML, non per
    #  oggetti custom); vedrai nella UI che ciò permette di avere info aggiuntive, nonché la possibilità di fare mdoel registry). OSS log_model carica oggetto salvato in memory, non su disco come log_artifact
    mlflow.sklearn.log_model(lr, artifact_path="model")


#eliminare esperimenti:
#mlflow.delete_experiment(experiment.experiment_id) 
#il problema è che rimangono nella trash e non puoi creare esperimento con stesso nome, 
#a meno che non cancelli fisicamente la trash 
#mlflow.delete_experiment('0')

