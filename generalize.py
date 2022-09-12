"""
Check if network generalizes to other tasks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from RNN import RNN
import tasks
import neurogym as ngym
from random import randint
import util

# Parameters
n_neu = 64          # number of recurrent neurons
dt = 100            # step size
tau = 100           # neuronal time constant (synaptic+membrane)
n_sd = 2            # standard deviation of injected noise
n_in = 3            # number of inputs
n_ff = 100          # number of neurons in feedforward neural net
n_out = 12          # number of outputs
batch_sz = 16       # batch size
n_batch = 1e4       # number of batches
trial_sz = 88       # drawing multiple trials in a row
print_every = int(n_batch/100)

# Load network
data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
net_file = 'LinCent64batch1e5Noise2nTask48'

net = RNN(n_in,n_neu,48,n_sd,tau,dt)
checkpoint = torch.load(os.path.join(data_path,net_file + '.pth'))
net.load_state_dict(checkpoint['state_dict'])

# Feedforward neural network that learns multiplication

ff_net = nn.Sequential(
        nn.Linear(n_neu,n_ff),
        nn.Sigmoid(),
        nn.Linear(n_ff,n_out)
        )

# Tasks
task = {'MultiplyClassification':tasks.MultiplyClassification}
#task_rules = util.assign_task_rules(task)
n_task = len(task)

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}

tenvs = [value(timing=timing,sigma=n_sd,n_task=n_out) for key, value in task.items()]

datasets = [ngym.Dataset(tenv,batch_size=batch_sz,seq_len=trial_sz) for tenv in tenvs]

# Optimizer
opt = optim.Adam(ff_net.parameters(), lr=0.003)

# Train feedforward neural net only
net.eval()
for param in net.parameters():
    param.requires_grad = False
ff_net.train()

# Train RNN
total_loss = 0; k = 0
loss_hist = np.zeros(100)

for i in range(int(n_batch)):
    # Randomly pick task
    dataset = datasets[randint(0,n_task-1)]
    # Generate data for current batch
    inputs, target = dataset()
    
    # Reshape so that batch is first dimension
    inputs = np.transpose(inputs,(1,0,2))
    target = np.transpose(target,(1,0,2))
    
    # Turn into tensors
    inputs = torch.from_numpy(inputs).type(torch.float)
    target = torch.from_numpy(target).type(torch.long)
    
    # Empty gradient buffers
    opt.zero_grad()
    
    # Forward run
    _, fr = net(inputs)
    output = ff_net(fr)
    
    # Compute loss
    loss, _ = util.MSELoss_weighted(output, target, torch.ones_like(target))
    total_loss += loss.item()
    
    # Backpopagate loss
    loss.backward()
    
    # Update weights
    opt.step()
    
    # Store history of average training loss
    if (i % print_every == 0):
        total_loss /= print_every
        print('{} % of the simulation complete'.format(round(i/n_batch*100)))
        print('Loss {:0.3f}'.format(total_loss))
        loss_hist[k] = total_loss
        loss = 0; k += 1


'''
# Synthetic data example

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

# Optimizer

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
'''