"""
Train recurrent neural network.
"""

import neurogym as ngym
import tasks
import util
from RNN import RNN
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from random import randint
import os
from pathlib import Path

# Tasks
task = {'TwoAlternativeForcedChoice':tasks.TwoAlternativeForcedChoice}
task_rules = util.assign_task_rules(task)
n_task = len(task)

# Constants
n_neu = 64          # number of recurrent neurons
batch_sz = 16       # batch size
n_batch = 1e3       # number of batches
dt = 100            # step size
tau = 100           # neuronal time constant (synaptic+membrane)
n_sd = 0            # standard deviation of injected noise
print_every = int(n_batch/100)

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}
grace = 200
#trial_sz = int(sum(timing.values())/dt) 
trial_sz = 88
n_grace = int(grace/dt); n_decision = int(timing['decision']/dt)

# Save location
data_path = str(Path(os.getcwd()).parent) + '\\trained_networks\\'
net_file = 'Perc' + str(n_neu) + \
            (('batch' + format(n_batch,'.0e').replace('+0','')) if not n_batch==1e4 else '') + \
            (('Noise' + str(n_sd)) if n_sd else '') + \
            (('tau' + str(tau)) if tau != 100 else '')

# Make supervised datasets
#tenvs = [value(timing=timing,rule_vec=task_rules[key]) for key, value in task.items()]
tenvs = ['PerceptualDecisionMaking-v0']
kwargs = {'dt': 100, 'sigma': 1}

datasets = [ngym.Dataset(tenv,env_kwargs=kwargs,batch_size=batch_sz,seq_len=trial_sz) for tenv in tenvs]

# A sample environment from dataset
env = datasets[0].env
# Visualize the environment with 2 sample trials
_ = ngym.utils.plot_env(env, num_trials=2)

# Network input and output size
n_in = env.observation_space.shape[0]
n_out = env.action_space.n

# Mask to weight errors during integration and decision equally
mask_w = (sum(timing.values()) - grace - timing['decision'])/timing['decision']
mask = np.ones((trial_sz,batch_sz)); mask[-n_decision-n_grace:-n_decision,:] = 0
mask[-n_decision:,:] = mask_w

# Initialize RNN  
net = RNN(n_in,n_neu,n_out,n_sd,tau,dt)

# Optimizer
opt = optim.Adam(net.parameters(), lr=0.01)

# Loss
criterion = nn.CrossEntropyLoss()

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
    target = np.transpose(target,(1,0))
    
    # Turn into tensors
    inputs = torch.from_numpy(inputs).type(torch.float)
    target = torch.from_numpy(target).type(torch.long)
    
    # Empty gradient buffers
    opt.zero_grad()
    
    # Forward run
    output, fr = net(inputs)
    
    # Compute loss
    loss = criterion(output.view(-1,n_out),target.flatten())
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
        
# Save network
torch.save({'state_dict': net.state_dict(),'loss_hist': loss_hist},
                    data_path + net_file + '.pth')