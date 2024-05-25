import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = pd.read_csv('traindata.txt', header=None)
rotations = data.iloc[:, -1]
data = data.iloc[:, :-1]

# add 256 to all negatives
negative_columns = data.columns[(data < 0).any()]
data[negative_columns] = data[negative_columns] + 256

# scale between 0 - 255
scaler_0_255 = MinMaxScaler(feature_range=(0, 255))

columns_to_scale = data.columns[(data.max() > 255)]
data[columns_to_scale] = scaler_0_255.fit_transform(data[columns_to_scale])

# log transformation either use this or the scaler (can use both, need to test more to find a proper answer)
high_value_columns = data.columns[(data.max() > 255)]
data[high_value_columns] = data[high_value_columns].apply(lambda x: np.log1p(x))

# this is dogshit
# scaler = StandardScaler()
# data = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# create a new dataframe since otherwise we get fragmentation due to calling a dataframe a lot
rotations = pd.DataFrame(rotations.values, columns=['rotations'])
data = pd.concat([data, rotations], axis=1)
data.to_csv('preprocessed_traindata.txt', index=False, header=False)

