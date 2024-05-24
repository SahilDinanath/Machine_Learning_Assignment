import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
# Load the test data
# for now just train data
#change to testdata.txt later
test_data = np.genfromtxt("testdata.txt", delimiter=",")

scaler = StandardScaler()
test_data = scaler.fit_transform(test_data)
test_data = torch.tensor(test_data, dtype=torch.float32)


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




# Get the input dimensions from the test data
input_dim = test_data.shape[1]

# Initialize the neural network model
model = NeuralNet(input_dim=input_dim, num_classes=21)  

# https://pytorch.org/tutorials/beginner/saving_loading_models.html
model.load_state_dict(torch.load("pretrained_model.pth", map_location=torch.device('cpu')))

# Make predictions on the test data
with torch.no_grad():
    outputs = model(test_data)
    _, predicted_labels = torch.max(outputs, 1)

# Save predicted labels to output file
np.savetxt("predlabels.txt", predicted_labels.numpy(), delimiter='\n',fmt="%d")



###remove later 
true_labels = np.genfromtxt("testlabels.txt", delimiter="\n")
predicted_labels = np.genfromtxt("predlabels.txt", delimiter="\n")

print(
    "Model accuracy: {:.2f}%".format(
        accuracy_score(true_labels, predicted_labels) * 100
    )
)
