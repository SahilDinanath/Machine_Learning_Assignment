import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

## trains with NVIDIA GPU if availible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("running on ", device)


#####################################################################
# splitting data step
#####################################################################
data = np.array(np.genfromtxt("traindata.txt", delimiter=","))
labels = np.array(np.genfromtxt("trainlabels.txt", delimiter="\n"))
#split the data

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

#write test data to testdata.txt
np.savetxt("testdata.txt", X_test,delimiter=',', fmt="%d")
np.savetxt("testlabels.txt", y_test, delimiter='\n',fmt="%d")



#maintain compatability with previous code variable names
data = X_train
labels = y_train

####################################################################

################################################################
# Training Neural Network step
################################################################
# very shitty minimal nonexistent preprocessing
scaler = StandardScaler()
data = scaler.fit_transform(data)

# split
train_data = torch.tensor(data, dtype=torch.float32).to(device)
train_labels = torch.tensor(labels, dtype=torch.long).to(device)

train_dataset = TensorDataset(train_data,train_labels)

# architecture
class NeuralNet(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# initialise nn model
model = NeuralNet(input_dim=train_data.shape[1], num_classes=len(np.unique(labels))).to(device)
# loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

num_epochs = 200
batch_size = 64


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
