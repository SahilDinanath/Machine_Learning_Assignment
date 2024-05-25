import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

# # Load the dataset
# data = pd.read_csv("../traindata.txt", delimiter=",")
# labels = pd.read_csv("../trainlabels.txt")
# #data = data.sample(n=500, random_state=37)
#
# # Compute the correlation matrix
# #corr_matrix = data.corr().abs()
#
# # Select upper triangle of the correlation matrix
# #upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#
# # Find index of feature columns with correlation greater than 0.95
# #to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
#
# # Drop features
# #data = data.drop(columns=to_drop)
#
#
# # Apply Variance Threshold
# #selector = VarianceThreshold(threshold=0.1)  # Adjust threshold as needed
# #data = selector.fit_transform(data)
#
# threshold = 256
# ## Loop through each column in the DataFrame
# data = pd.DataFrame(data)
# for column in data.columns:
#     # Check if any element in the current column exceeds the threshold
#     if (data[column] > threshold).any():
#         # Divide the entire column by 10
#         data[column] = data[column] / 10
# data = data.to_numpy()
#
# def invert_bits(number, bit_length=16):
#     # Convert the number to binary with the specified bit length
#      binary_str = format(number if number >= 0 else (1 << bit_length) + int(number), f'0{bit_length}b')
#
#      # Invert the bits
#      inverted_binary_str = ''.join('1' if bit == '0' else '0' for bit in binary_str)
#
#      # Convert the inverted binary string back to a decimal number
#      inverted_number = int(inverted_binary_str, 2)
#
#      # Handle signed conversion if necessary
#      if inverted_binary_str[0] == '1':  # if the number is negative in two's complement form
#          inverted_number -= 1 << bit_length
#      return inverted_number
#
# data = pd.DataFrame(data)
# for column in data.columns:
#      if (data[column] < 0).any():
#          data[column] = data[column].apply(lambda x: invert_bits(x))
# data = data.to_numpy()
#
# # Apply PCA
# scaler = StandardScaler()
# data = scaler.fit_transform(data)
#
# pca = PCA(n_components=5)
# data = pca.fit_transform(data)
# print(np.unique(labels))
#
# #convert to pandas data frame
# data = pd.DataFrame(data)
# data['label'] = labels
# data = data.sample(n=200, random_state=37)
# # Create the scatter matrix
# sns.pairplot(data, hue="label")
#
# # Show the plot
# plt.show()

import numpy as np
import matplotlib.pyplot as plt

# Load data from the file
data = pd.read_csv("preprocessed_traindata.txt", delimiter=",")
data = data.values.flatten()
# Plot histogram
plt.hist(data, bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Frequency Histogram')
plt.grid(True)
plt.show()

