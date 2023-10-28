import pandas as pa 
import numpy as np
import warnings
import config
import modeldispatcher
from sklearn import metrics
warnings.filterwarnings("ignore")

Data = pa.read_csv(config.trainFileName)
Data.drop(columns=["Unnamed: 0"], inplace=True)

def run_fold(model, fold = 0):
    print("Current Fold: ", fold)
    print("--------")
    training_data_X = Data[Data["Kfold"] != fold].reset_index(drop=True).drop(columns="Price").values
    training_data_Y = Data[Data["Kfold"] != fold].reset_index(drop=True)["Price"].values
    testing_data_X = Data[Data["Kfold"] == fold].reset_index(drop=True).drop(columns="Price").values
    testing_data_Y = Data[Data["Kfold"] == fold].reset_index(drop=True)["Price"].values
    fetch = modeldispatcher.models[model]
    fetch.fit(training_data_X,training_data_Y)
    print(fetch.get_params())
    result = fetch.predict(testing_data_X)

    return metrics.mean_absolute_error(testing_data_Y, result)

def get_models(file):
    Table = pa.DataFrame(columns= modeldispatcher.models.keys())
    for keys,values in modeldispatcher.models.items():
        print("Current model: ", keys)
        print("-----------")
        mean = 0
        for i in range(0,3):
           Table.loc[i, keys] = run_fold(keys,i)
           mean += Table.loc[i,keys]    
        Table.loc[3, keys] = mean / 3
        print("-----------")
 #   Table.transpose()
    Table.to_csv("../input/model_result_"+ file +".csv")