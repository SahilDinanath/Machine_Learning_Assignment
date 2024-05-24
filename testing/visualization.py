import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold

# Load the dataset
data = pd.read_csv("traindata.txt", delimiter=",")
data = data.sample(n=500, random_state=37)

# Compute the correlation matrix
#corr_matrix = data.corr().abs()

# Select upper triangle of the correlation matrix
#upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find index of feature columns with correlation greater than 0.95
#to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]

# Drop features
#data = data.drop(columns=to_drop)


# Apply Variance Threshold
#selector = VarianceThreshold(threshold=0.1)  # Adjust threshold as needed
#data = selector.fit_transform(data)

# Apply PCA
pca = PCA(n_components=3)
data = pca.fit_transform(data)


#convert to pandas data frame
data = pd.DataFrame(data)
# Create the scatter matrix
sns.pairplot(data)

# Show the plot
plt.show()
