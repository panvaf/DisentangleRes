"""
Analyze learned RNNs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from RNN import RNN
import os
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import util
import tasks
from random import randint

# Parameters
n_neu = 64          # number of recurrent neurons
dt = 100            # step size
tau = 100           # neuronal time constant (synaptic+membrane)
n_sd = 1            # standard deviation of injected noise
n_in = 5            # number of inputs
n_out = 3           # number of outputs
n_trial = 100      # number of example trials to plot

# Tasks
task = {'TwoAlternativeForcedChoice':tasks.TwoAlternativeForcedChoice,
        'AttributeIntegration':tasks.AttributeIntegration}
task_rules = util.assign_task_rules(task)
n_task = len(task)

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}
tenvs = [value(timing=timing,rule_vec=task_rules[key]) for key, value in task.items()]

# Load network
data_path = str(Path(os.getcwd()).parent) + '\\saved_networks\\'
net_file = 'Joint64Noise1'

net = RNN(n_in,n_neu,n_out,n_sd,tau,dt)
checkpoint = torch.load(os.path.join(data_path,net_file + '.pth'))
net.load_state_dict(checkpoint['state_dict'])
net.eval()

# Visualize RNN activity
activity_dict = {}; output_dict = {}; trial_info = {}

for i in range(n_trial):
    
    # Pick environment and generate a trial
    tenv = tenvs[randint(0,n_task-1)]
    tenv.new_trial()
    ob, gt = tenv.ob, tenv.gt
    inp = torch.from_numpy(ob[:, np.newaxis, :]).type(torch.float)
    output, rnn_activity = net(inp)
    rnn_activity = rnn_activity[0, :, :].detach().numpy()
    output = output[0, :, :].detach().numpy()
    activity_dict[i] = rnn_activity
    output_dict[i] = output
    trial_info[i] = tenv.trial
    
activity = np.concatenate(list(activity_dict[i] for i in range(len(n_trial))), axis=0)
    
# Perform PCA
pca = PCA(n_components=3)
pca.fit(activity)

# Find approximate fixed points, depending on network initial conditions

# Freeze for parameters in the recurrent network
for param in net.parameters():
    param.requires_grad = False

batch_size = 8
fixedpoints = np.empty([n_trial,batch_size,n_neu])

for j in range(n_trial):
    print('Trial {} out of {}'.format(i+1,n_trial))
    
    # Inputs are zero, so that internal representation is not affected
    inp = np.tile(np.concatenate([1, 0, 0],list(task_rules.values())[randint(0,n_task-1)]),
                  (batch_size, 1))
    inp = torch.tensor(input, dtype=torch.float32)

    # Initialize hidden activity randomly                                                                         
    hidden = torch.tensor(np.random.rand(batch_size, n_neu)*3,
                      requires_grad=True, dtype=torch.float32)
    
    # Use Adam optimizer
    optimizer = optim.Adam([hidden], lr=0.001)
    criterion = nn.MSELoss()
    
    running_loss = 0
    for i in range(1000):
        optimizer.zero_grad()   # zero the gradient buffers
        
        # Take the one-step recurrent function from the trained network
        new_h = net.rnn.rec_dynamics(inp, hidden)
        loss = criterion(new_h, hidden)
        loss.backward()
        optimizer.step()    # Does the update
    
        running_loss += loss.item()
        if i % 100 == 99:
            running_loss /= 100
            print('Step {}, Loss {:0.4f}'.format(i+1, running_loss))
            running_loss = 0
            
    fxdpoints = hidden.detach().numpy()
    fixedpoints[j] = fxdpoints

fixedpoints_pc_trans = pca.transform(fixedpoints.reshape(-1,n_neu))