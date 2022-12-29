# data_raw come from from: (https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)
#data are too big to run fast experiments. Let's 'sample' them, deleting some of the least represented zones (PULocationID+DOLocationID) (one-hot-encoding will generate less variables)

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from pickle import dump

from sqlalchemy import desc


#for a not too aggressive cut, set pu_do_ts = 10, frac = 1
pu_do_ts = 50 # min freq of pu_do to keep. The higher, the smaller the final dataset
frac = 0.1 #random select frac % of rows

year_months = ['2022-01', '2022-02', '2022-03', '2022-04']
raw_data_path = './data_raw/green_tripdata_'

path_save='./green_tripdata_' #local path where to save objects to log as artifacts

for year_month in year_months:
    print('############## year ',year_month,'#######################')
    df = pd.read_parquet(raw_data_path+year_month+'.parquet')
    print('df.shape \n',df.shape)
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
    #this is the main variable used in the model: 6966 distinc categories!! Data too big., 
    df['PU_DO'] = df['PULocationID'] + '_' + df['DOLocationID']

    #delete PU_DO that occur less= than pu_do_ts times. From 6966 distinct categories (and 62495 rows) to 809 distinc categories (and 48672 rows) for green_tripdata_2022-01
    pu_do = df['PU_DO'].value_counts() #series
    print('distinct pu_do \n',pu_do.shape)
    print('total df rows \n',pu_do.sum()) # equal df.shape[0]
    print(f'distinct pu_do>{pu_do_ts} \n',pu_do[pu_do>pu_do_ts].shape)
    print(f'total rows pu_do>{pu_do_ts} \n',pu_do[pu_do>pu_do_ts].sum())

    pu_do_count = pu_do.reset_index().rename(columns={'PU_DO':'count', 'index':'PU_DO'})
    print(pu_do_count.head())
    pu_do_count_filtered = pu_do_count.loc[pu_do_count['count']>pu_do_ts].drop(columns=['count'],axis=1)
    df_filtered = pd.merge(left=df,right=pu_do_count_filtered, on='PU_DO').drop(columns=['PU_DO'],axis=1)

    assert df_filtered.shape[0]==pu_do[pu_do>pu_do_ts].sum()

    df_filtered = df_filtered.sample(frac = frac)
    print('shape after subsample',df_filtered.shape)
    #save results
    df_filtered.to_parquet(path_save+year_month+'.parquet')
