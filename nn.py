import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

data = np.genfromtxt("traindata.txt", delimiter=",")
labels = np.genfromtxt("trainlabels.txt", delimiter="\n")

# very shitty minimal nonexistent preprocessing
scaler = StandardScaler()
data = scaler.fit_transform(data)

# split
train_data = torch.tensor(data, dtype=torch.float32)
train_labels = torch.tensor(labels, dtype=torch.long)

# architecture
class NeuralNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# initialise nn model
model = NeuralNet(input_dim=train_data.shape[1], num_classes=len(np.unique(labels)))

# loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 100
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(train_data), batch_size):
        inputs = train_data[i:i+batch_size]
        targets = train_labels[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

torch.save(model.state_dict(), "pretrained_model.pth")