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
import matplotlib.pyplot as plt

# Parameters
n_neu = 64          # number of recurrent neurons
dt = 100            # step size
tau = 100           # neuronal time constant (synaptic+membrane)
n_sd = 2            # standard deviation of injected noise
n_in = 3            # number of inputs
n_ff = 96           # number of neurons in feedforward neural net
n_out = 2           # number of outputs
batch_sz = 16       # batch size
n_batch = 1e3       # number of batches for training
n_test = 100        # number of test batches
trial_sz = 88       # drawing multiple trials in a row
print_every = int(n_batch/100)

# Load network
data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
net_file = 'LinMult64batch2e4Noise2nTask48'

net = RNN(n_in,n_neu,96,n_sd,tau,dt)
checkpoint = torch.load(os.path.join(data_path,net_file + '.pth'))
net.load_state_dict(checkpoint['state_dict'])

# Feedforward neural network that learns multiplication

ff_net = nn.Sequential(
        nn.Linear(n_ff,n_out)
        #nn.Sigmoid(),
        #nn.Linear(n_ff,n_out)
        )

# Tasks
task = {'DenoiseQuads':tasks.DenoiseQuads}
#task_rules = util.assign_task_rules(task)
n_task = len(task)

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}

tenvs = [value(timing=timing,sigma=n_sd,n_task=n_out,quad_num=np.array([1,2,3])) for key, value in task.items()]

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
    target = torch.from_numpy(target).type(torch.float)
    
    # Empty gradient buffers
    opt.zero_grad()
    
    # Forward run
    net_out, fr = net(inputs)
    output = ff_net(net_out)
    
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


# Evaluate

#task = {'Denoise':tasks.Denoise}

tenvs = [value(timing=timing,sigma=n_sd,n_task=n_out,quad_num=np.array([4])) for key, value in task.items()]

datasets = [ngym.Dataset(tenv,batch_size=batch_sz,seq_len=trial_sz) for tenv in tenvs]

ff_net.eval()

errors = []
with torch.no_grad():
    for i in range(n_test):
        dataset = datasets[randint(0,n_task-1)]
        # Generate data for current batch
        inputs, target = dataset()
        
        # Reshape so that batch is first dimension
        inputs = np.transpose(inputs,(1,0,2))
        target = np.transpose(target,(1,0,2))
        
        # Turn into tensors
        inputs = torch.from_numpy(inputs).type(torch.float)
        target = torch.from_numpy(target).type(torch.float)
        
        # Forward run
        net_out, fr = net(inputs)
        output = ff_net(net_out)
        
        a = output.detach().numpy()
        b = target.detach().numpy()
        c = b - a
        
        errors.append(np.reshape(c[:,[21,43,65,87],:],(-1,n_out)))
        
errors = np.reshape(np.asarray(errors),(-1,n_out))
err = np.abs(errors) > .5

# Plot

# Fontsize appropriate for plots
SMALL_SIZE = 10
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)     # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)     # legend fontsize
plt.rc('figure', titlesize=MEDIUM_SIZE)   # fontsize of the figure title

# Misclassification distance
plt.hist(np.sum(err,axis=1))
plt.xlabel('Misclassification distance')
plt.ylabel('Count')
plt.imshow()

# Classification lines
x = np.linspace(0,.5,100)
for a in tenvs[0].thres:
    plt.plot(x,np.sqrt(a**2-x**2))
plt.ylim([0,.5])
plt.xlabel('True evidence 1')
plt.ylabel('True evidence 2')
plt.title('Classification lines')
plt.imshow()

env = dataset.env

fig,ax = plt.subplots(figsize=(3,.75))
plt.plot(env.ob[:,1],label='Noisy')
plt.plot(np.ones(22)*env.trial['stim'][0],label='True')
    
plt.ylabel('Evidence 2')
plt.xlabel('Time')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -1))
ax.spines['bottom'].set_position(('data', -.1))
#plt.legend(prop={'size': SMALL_SIZE},frameon=False,ncol=2,bbox_to_anchor=(.1,1))
plt.xlim([0,22])
plt.ylim([0,.7])
plt.savefig('evidence2.png',bbox_inches='tight',format='png',dpi=300)
plt.imshow()

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