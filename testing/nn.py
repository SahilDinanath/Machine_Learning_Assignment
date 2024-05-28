import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import ParameterGrid, train_test_split
import torch
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from preprocessor import preprocess

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
data = preprocess('traindata.txt')
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
    def __init__(self, input_dim, hidden_units,num_classes, drop_out):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units)
        self.fc2 = nn.Linear(hidden_units, hidden_units)
        self.fc3 = nn.Linear(hidden_units, hidden_units)
        self.fc4 = nn.Linear(hidden_units, hidden_units)
        self.fc5 = nn.Linear(hidden_units, num_classes)
        self.dropout = nn.Dropout(drop_out)
        self.batch_norm1 = nn.BatchNorm1d(hidden_units)
        self.batch_norm2 = nn.BatchNorm1d(hidden_units)
        self.batch_norm3 = nn.BatchNorm1d(hidden_units)
        self.batch_norm4 = nn.BatchNorm1d(hidden_units)

    def forward(self, x):
        x = self.dropout(F.relu(self.batch_norm1(self.fc1(x))))
        x = self.dropout(F.relu(self.batch_norm2(self.fc2(x))))
        x = self.dropout(F.relu(self.batch_norm3(self.fc3(x))))
        x = self.dropout(F.relu(self.batch_norm4(self.fc4(x))))
        x = self.fc5(x)
        return x

# Define hyperparameters to tune
params_grid = {
    'lr': [0.001, 0.01, 0.1],
    'hidden_units': [64,128,256,512,1024],
    'drop_out': [0.2,0.3, 0.5, 0.7]
    # Add other hyperparameters to tune
}

# initialise nn model
# model = NeuralNet(input_dim=train_data.shape[1], num_classes=len(np.unique(labels))).to(device)

X_test = preprocess('testdata.txt')
y_test = np.array(np.genfromtxt("./targetlabels.txt", delimiter="\n"))

#data, X_test, labels, y_test = train_test_split(data, labels, test_size=0.1, random_state=32)
#used to test model later

criterion = nn.CrossEntropyLoss()

for category in range(4):
    best_accuracy = 0.0
    for params in ParameterGrid(params_grid):
        print(f"Processing category {category}...")
        indices = data[:, -1] == category
        data_cat = data[indices]
        labels_cat = labels[indices]
        #data_cat = data_cat[:, :-1]
        # Select only the columns corresponding to the selected features
        #data_cat = data_cat[:, selected_features]

        data_cat = torch.from_numpy(data_cat).float().to(device)
        labels_cat = torch.from_numpy(labels_cat).long().to(device)

        X_train, X_valid, y_train, y_valid = train_test_split(data_cat, labels_cat, test_size=0.05, random_state=32)
        #Added validation 
        # Split temporary set into validation and test sets

        #removes last column now, we need it for testing
        X_train = X_train[:, :-1].to(device)
        X_valid = X_valid[:, :-1].to(device)
        data_cat = data_cat[:, :-1]

        input_dim = data_cat.shape[1]
        num_classes = 21
        model = NeuralNet(input_dim=input_dim, hidden_units=params['hidden_units'], num_classes=num_classes, drop_out=params['drop_out']).to(device)
        optimizer = Adam(model.parameters(), lr=params['lr'], weight_decay=0.01)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

        model.train()
        best_loss = np.inf
        patience, trials = 20, 0
        for epoch in range(500):
            optimizer.zero_grad()
            outputs = model(X_train)
            loss = criterion(outputs, y_train)
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
        
        #Validate Model
        # Validate model
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
                # Forward pass
            outputs = model(X_valid)
            _, predicted = torch.max(outputs.data, 1)
            total += len(y_valid)
            correct += (predicted == y_valid).sum().item()
            accuracy = correct / total
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_params = params
                torch.save({
                    'model_state_dict':model.state_dict(),
                    'hidden_units' :params['hidden_units'],
                    'input_dim' : input_dim,
                    'class_dim' : num_classes,
                    'drop_out' : params['drop_out']
                }, f'best_model_cat_{category}.pth')

    print("category {}:".format(category))
    print("Best Accuracy:", best_accuracy)
    print("Best Hyperparameters:", best_params)

#Test Model
accuracies = []
for category in range(4):
    print(f"Processing category {category}...")
    indices = X_test[:, -1] == category
    data_cat = X_test[indices]
    labels_cat = y_test[indices]
    
    indices = X_test[:, -1] == category
    data_cat = X_test[indices]
    labels_cat = y_test[indices]


     #data_cat = data_cat[:, :-1]
     # Select only the columns corresponding to the selected features
     #data_cat = data_cat[:, selected_features]
    data_cat = data_cat[:, :-1]
    data_cat = torch.from_numpy(data_cat).float()
    labels_cat = torch.from_numpy(labels_cat).long()

    model_data = torch.load(f"best_model_cat_{category}.pth")
    model = NeuralNet(model_data["input_dim"],model_data['hidden_units'], model_data['class_dim'], model_data['drop_out'])
    model.load_state_dict(model_data['model_state_dict'])

    model.eval()
    with torch.no_grad():
        outputs = model(data_cat)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == labels_cat).sum().item()
        accuracy = 100 * correct / len(labels_cat)
        accuracies.append(accuracy)
        print(f'Accuracy for category {category}: {accuracy}%')

average_accuracy = np.mean(accuracies)
print(f'Average accuracy: {average_accuracy}%')

#torch.save(model.state_dict(), "pretrained_model.pth")
