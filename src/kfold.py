import pandas as pa
import config
from sklearn import model_selection

Data = pa.read_csv(config.TrainingData)
Data = Data.sample(frac=1).reset_index(drop=True)
Data["Kfold"] = -1
Kfolder = model_selection.KFold(n_splits=10)

for fold, (train_array, test_array) in enumerate( Kfolder.split(X = Data)):
    Data.loc[test_array, "Kfold"] = fold

Data.tail()

Data.to_csv(config.TrainingData[:-4] + "folded.csv", index=False)