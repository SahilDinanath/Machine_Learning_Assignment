import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim

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
batch_size = 8
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
