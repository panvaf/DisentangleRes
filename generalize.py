"""
Check if network generalizes to other tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Data
n_train = 100000
n_test = 100

train_data = np.zeros((n_train,3))
test_data = np.zeros((n_test,3))

train_data[:,0:2] = np.random.random((n_train,2))
test_data[:,0:2] = np.random.random((n_test,2))

train_data[:,2] = train_data[:,0] * train_data[:,1]
test_data[:,2] = test_data[:,0] * test_data[:,1]

# Dataset

class MultDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, ind):
        x = self.data[ind][0:2]
        y = self.data[ind][2]
        return x, y
    
train_set = MultDataset(train_data)
test_set  = MultDataset(test_data)

batch_size = 16
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False)

# Feedforward neural network that learns multiplication

model = nn.Sequential(
        nn.Linear(2,100),
        nn.Sigmoid(),
        nn.Linear(100,1))

optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

# Train model
epochs = 10

model.train()

for epoch in range(epochs):
    losses = []
    for batch_num, input_data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.float()
        y = y.float()

        output = model(x)
        loss = criterion(output[:,0], y)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()

        if batch_num % 40 == 0:
            print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
    print('Epoch %d | Loss %6.2f' % (epoch, sum(losses)/len(losses)))


model.eval()

test_losses = []
with torch.no_grad():
    for x, y in test_loader:
        x = x.float()
        y = y.float()
        
        output = model(x)
        test_losses.append((output[:,0] - y).detach().numpy())