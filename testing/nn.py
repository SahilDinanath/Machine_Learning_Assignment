import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2
import pandas as pd
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.feature_selection import RFE
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
##############################
# Setup
##############################
## trains with NVIDIA GPU if availible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("running on ", device)











#####################################################################
# splitting data step
#####################################################################


#################
#process raw data 
#################
#remove columns etc
data = np.array(np.genfromtxt("traindata.txt", delimiter=","))

################
#use pandas to perform statistical cleaning
################
data = pd.DataFrame(data)


labels = np.array(np.genfromtxt("trainlabels.txt", delimiter="\n"))
desc_stats = data.describe()
print("Descriptive Statistics:\n", desc_stats)

#select rows that have 0 in the last column
# mask = data[data.columns[-1]] == 0
# data = data[mask]
# labels = labels[mask]

#
# ##
# ## INFO: divide columns with above 255 values by 10
# ##
threshold = 256
# Loop through each column in the DataFrame
for column in data.columns:
    # Check if any element in the current column exceeds the threshold
    if (data[column] > threshold).any():
        # Divide the entire column by 10
        data[column] = data[column] / 10


# INFO: replace any columns with 255 with the average
def replace_max_with_mean(df, max_value=255):
    # Iterate over each column
    for col in df.columns:
        # Calculate the mean of non-maxed-out values
        mean_value = df[df[col] != max_value][col].mean()
        
        # Replace maxed-out values with the mean
        df[col] = df[col].replace(max_value, mean_value)
    
    return df

data = replace_max_with_mean(data)

##
## INFO: convert negative columns to binary then invert the binary
##
def invert_bits(number, bit_length=16):
    # Convert the number to binary with the specified bit length
     binary_str = format(number if number >= 0 else (1 << bit_length) + int(number), f'0{bit_length}b')

     # Invert the bits
     inverted_binary_str = ''.join('1' if bit == '0' else '0' for bit in binary_str)

     # Convert the inverted binary string back to a decimal number
     inverted_number = int(inverted_binary_str, 2)

     # Handle signed conversion if necessary
     if inverted_binary_str[0] == '1':  # if the number is negative in two's complement form
         inverted_number -= 1 << bit_length
     return inverted_number

# for column in data.columns:
#      if (data[column] < 0).any():
#          data[column] = data[column].apply(lambda x: invert_bits(x))
#
# INFO: Drop columns which are not in these bounds
# remove rows that have value 255
#

# min_value = 0
# max_value = 256
# # # # # Identify columns to drop
# cols_to_drop = [col for col in data.columns if not ((data[col] >= min_value) & (data[col] <= max_value)).all()]
# # # # Drop the columns
# data = data.drop(columns=cols_to_drop)
# desc_stats = data.describe()
# print("Descriptive Statistics:\n", desc_stats)


# very shitty minimal nonexistent preprocessing
# print(data)
scaler = StandardScaler()
data = scaler.fit_transform(data)

#split the data
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

#write test data to testdata.txt
np.savetxt("testdata.txt", X_test,delimiter=',')
np.savetxt("testlabels.txt", y_test, delimiter='\n',fmt="%d")

#maintain compatability with previous code variable names
data = X_train
labels = y_train

####################################################################
























################################################################
# Training Neural Network step
################################################################


# split
train_data = torch.tensor(data, dtype=torch.float32).to(device)
train_labels = torch.tensor(labels, dtype=torch.long).to(device)

train_dataset = TensorDataset(train_data,train_labels)

 # # architecture
class NeuralNet(nn.Module):
     def __init__(self, input_dim, num_classes):
         super(NeuralNet, self).__init__()
         self.fc1 = nn.Linear(input_dim, input_dim)
         self.fc2 = nn.Linear(input_dim, input_dim)
         self.fc3 = nn.Linear(input_dim, input_dim)
         self.fc4 = nn.Linear(input_dim, input_dim)
         self.fc5 = nn.Linear(input_dim, num_classes)
        
     def forward(self, x):
         x = torch.relu(self.fc1(x))
         x = torch.relu(self.fc2(x))
         x = torch.relu(self.fc3(x))
         x = torch.relu(self.fc4(x))
         x = self.fc5(x)
         return x


# initialise nn model
model = NeuralNet(input_dim=train_data.shape[1], num_classes=len(np.unique(labels))).to(device)


#hyper parameters
num_epochs = 300
batch_size = 128

# loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


for epoch in range(num_epochs):
    model.train()
    for batch_data, batch_labels in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

torch.save(model.state_dict(), "pretrained_model.pth")
