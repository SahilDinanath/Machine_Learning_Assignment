#  Import any Python standard libraries you wish   #
# - I.e. libraries that do not require pip install with fresh
#   install of Python #
import os
##################################
# ALLOWED NON-STANDARD LIBRARIES #
##################################
# Un-comment out the ones you use
import numpy as np
import pandas as pd
import sklearn
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessor import preprocess
# import matplotlib
# import seaborn as sns
##################################

def main():
    # TODO
    # pass
    X_test = preprocess("testdata.txt")

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

    models = {}
    for category in range(4):
        model_data = torch.load(f"best_model_cat_{category}.pth")
        model = NeuralNet(model_data["input_dim"], model_data['hidden_units'], model_data['class_dim'], model_data['drop_out'])
        model.load_state_dict(model_data['model_state_dict'])
        model.eval()
        models[category] = model

    predictions = []

    for row in X_test:
        category = int(row[-1])
        data_point = torch.from_numpy(row[:-1].astype(np.float32)).unsqueeze(0)  # Add batch dimension

        model = models[category]

        with torch.no_grad():
            output = model(data_point)
            _, predicted = torch.max(output, 1)
            predictions.append(predicted.item())

    np.savetxt("predlabels.txt", predictions, fmt='%d')

if __name__ == "__main__":
    main()
