import pandas as pd
from pandas.core.api import DataFrame
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def preprocess(path):
    data = pd.read_csv(path, header=None)
    rotations = data.iloc[:, -1]
    data = data.iloc[:, :-1]

    # add 256 to all negatives
    # negative_columns = data.columns[(data < 0).any()]
    # data[negative_columns] = data[negative_columns] 
    # remove negative columns
    data = data.loc[:, (data >= 0).all()]
    # remove columns greater then 255
    #data = data.loc[:, (data <=255).all()]
    # scale between 0 - 255
    scaler = MinMaxScaler(feature_range=(0, 255))

    columns_to_scale = data.columns[(data.max() > 255)]
    data[columns_to_scale] = scaler.fit_transform(data[columns_to_scale])

    # columns_with_255 = data.columns[data.isin([255]).any()]
    # data = data[columns_with_255]
    #

    # log transformation either use this or the scaler (can use both, need to test more to find a proper answer)
    #high_value_columns = data.columns[(data.max() > 255)]
    #data[high_value_columns] = data[high_value_columns].apply(lambda x: np.log1p(x))

    # this is dogshit
    # scaler = StandardScaler()
    # data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
    #data = data.to_numpy()
    # col_means = np.mean(data[data != 255], axis=0) 
    # data[data == 255] = 0
    #data = DataFrame(data)

    # create a new dataframe since otherwise we get fragmentation due to calling a dataframe a lot
    rotations = pd.DataFrame(rotations.values, columns=['rotations'])
    data = pd.concat([data, rotations], axis=1)
    
    return data.to_numpy()
