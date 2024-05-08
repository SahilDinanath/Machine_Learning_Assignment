import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn

# Load the test data
# for now just train data
test_data = np.genfromtxt("traindata.txt", delimiter=",")

scaler = StandardScaler()
test_data = scaler.fit_transform(test_data)

test_data = torch.tensor(test_data, dtype=torch.float32)

class NeuralNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
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
np.savetxt("predlabels.txt", predicted_labels.numpy(), fmt="%d")

true_labels = np.genfromtxt("trainlabels.txt", delimiter=",")
predicted_labels = np.genfromtxt("predlabels.txt", delimiter=",")
print(
    "Model accuracy: {:.2f}%".format(
        accuracy_score(true_labels, predicted_labels) * 100
    )
)