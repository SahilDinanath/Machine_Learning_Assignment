import torch.nn.functional as F
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
from torch.optim import Adam, lr_scheduler

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
data = np.array(np.genfromtxt("preprocessed_traindata.txt", delimiter=","))
labels = np.array(np.genfromtxt("trainlabels.txt", delimiter="\n"))

################
#use pandas to perform statistical cleaning
################
#data = pd.DataFrame(data)


#desc_stats = data.describe()
#print("Descriptive Statistics:\n", desc_stats)

#select rows that have 0 in the last column
# mask = data[data.columns[-1]] == 0
# data = data[mask]
# labels = labels[mask]

##
## ##
## ## INFO: divide columns with above 255 values by 10
## ##
# threshold = 256
# ## Loop through each column in the DataFrame
# data = pd.DataFrame(data)
# for column in data.columns:
#     # Check if any element in the current column exceeds the threshold
#     if (data[column] > threshold).any():
#         # Divide the entire column by 10
#         data[column] = data[column] / 10
# data = data.to_numpy()
# #
## INFO: replace any columns with 255 with the average
#def replace_max_with_mean(df, max_value=255):
#    # Iterate over each column
#    for col in df.columns:
#        # Calculate the mean of non-maxed-out values
#        mean_value = df[df[col] != max_value][col].mean()
#
#        # Replace maxed-out values with the mean
#        df[col] = df[col].replace(max_value, mean_value)
#
#    return df
#
#data = replace_max_with_mean(data)
#
###
### INFO: convert negative columns to binary then invert the binary
###
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
## INFO: Drop columns which are not in these bounds
## remove rows that have value 255
##
#
# data = pd.DataFrame(data)
# min_value = 0
# max_value = 75
# # # # # Identify columns to drop
# cols_to_drop = [col for col in data.columns if not ((data[col] >= min_value) & (data[col] <= max_value)).all()]
# # # # Drop the columns
# data = data.drop(columns=cols_to_drop)
# desc_stats = data.describe()
# print("Descriptive Statistics:\n", desc_stats)
# data = data.to_numpy()
# #
## very shitty minimal nonexistent preprocessing
## print(data)
#scaler = StandardScaler()
#data = scaler.fit_transform(data)
#
##split the data
#X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)
#
##write test data to testdata.txt
#np.savetxt("testdata.txt", X_test,delimiter=',')
#np.savetxt("testlabels.txt", y_test, delimiter='\n',fmt="%d")
#
##maintain compatability with previous code variable names
#data = X_train
#labels = y_train

####################################################################


################################################################
# Training Neural Network step
################################################################


# split
# train_data = torch.tensor(data, dtype=torch.float32).to(device)
# train_labels = torch.tensor(labels, dtype=torch.long).to(device)
#
# train_dataset = TensorDataset(train_data,train_labels)
#
 # # architecture
class NeuralNet(nn.Module):    
    def __init__(self, input_dim, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(1024)
        self.batch_norm2 = nn.BatchNorm1d(512)
        self.batch_norm3 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = self.dropout(F.relu(self.batch_norm1(self.fc1(x))))
        x = self.dropout(F.relu(self.batch_norm2(self.fc2(x))))
        x = self.dropout(F.relu(self.batch_norm3(self.fc3(x))))
        x = self.fc4(x)
        return x


# initialise nn model
# model = NeuralNet(input_dim=train_data.shape[1], num_classes=len(np.unique(labels))).to(device)

accuracies = []

for category in range(4):
    print(f"Processing category {category}...")
    indices = data[:, -1] == category
    data_cat = data[indices]
    labels_cat = labels[indices]
    data_cat = data_cat[:, :-1]
    # Select only the columns corresponding to the selected features
    #data_cat = data_cat[:, selected_features]

    data_cat = torch.from_numpy(data_cat).float()
    labels_cat = torch.from_numpy(labels_cat).long()

    X_train, X_test, y_train, y_test = train_test_split(data_cat, labels_cat, test_size=0.2, random_state=42)

    input_dim = data_cat.shape[1]
    num_classes = len(np.unique(labels_cat))
    model = NeuralNet(input_dim, num_classes)
    optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    model.train()
    best_loss = np.inf
    patience, trials = 20, 0
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = F.cross_entropy(outputs, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if loss.item() < best_loss:
            trials = 0
            best_loss = loss.item()
        else:
            trials += 1
            if trials >= patience:
                print(f"Early stopping on epoch {epoch}")
                break
    
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test).sum().item()
        accuracy = 100 * correct / len(y_test)
        accuracies.append(accuracy)
        print(f'Accuracy for category {category}: {accuracy}%')

average_accuracy = np.mean(accuracies)
print(f'Average accuracy: {average_accuracy}%')

torch.save(model.state_dict(), "pretrained_model.pth")
