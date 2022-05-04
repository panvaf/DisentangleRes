"""
Train recurrent neural network.
"""

import neurogym as ngym
from neurogym import spaces
import tasks
import util
from RNN import RNN
import torch.optim as optim
import torch
import numpy as np
from random import randint

# Tasks
task = {'TwoAlternativeForcedChoice':tasks.TwoAlternativeForcedChoice,
        'AttributeIntegration':tasks.AttributeIntegration}
task_rules = util.assign_task_rules(task)
n_task = len(task)

# Constants
n_neu = 64          # number of recurrent neurons
seq_len = 100       # size of trial sequence
n_batch = 1e5       # number of batches
dt = 1e-1           # step size
tau = 1e-1          # neuronal time constant (synaptic+membrane)
n_sd = .1           # standard deviation of injected noise
print_every = int(n_batch/100)

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}
grace = 200
# Mask to weight errors during integration and decision equally
mask_w = (sum(timing.values()) - grace - timing['decision'])/timing['decision']
mask = torch.ones(n_batch,n_t,n_in); mask[:,0:n_grace,:] = 0

tenvs = [value(timing=timing,grace=grace,rule_vec=task_rules[key]) for key, 
         value in task.items()]

# Make supervised datasets
datasets = [ngym.Dataset(tenv, batch_size=16, seq_len=seq_len) for tenv in tenvs]

# A sample environment from dataset
env = datasets[0].env
# Visualize the environment with 2 sample trials
_ = ngym.utils.plot_env(env, num_trials=2)

# Network input and output size
n_in = env.observation_space.shape[0]
n_out = env.action_space.n

# Initialize RNN  
net = RNN(n_in,n_neu,n_out,n_sd,tau*1e3,dt*1e3)

# Optimizer
opt = optim.Adam(net.parameters(), lr=0.001)

# Train RNN
loss = 0; k = 0
loss_hist = np.zeros(100)

for i in range(int(n_batch)):
    # Randomly pick task
    dataset = datasets[randint(n_task)]
    # Generate data for current batch
    inputs, target = dataset()
    
    inputs = torch.from_numpy(inputs).type(torch.float)
    target = torch.from_numpy(target.flatten()).type(torch.long)
    
    # Empty gradient buffers
    opt.zero_grad()
    
    # Forward run
    output, fr = net(inputs)
    
    # Compute loss
    _, normed_loss = util.MSELoss_weighted(output,target,mask)
    loss += normed_loss.item()
    
    # Backpopagate loss
    normed_loss.backward()
    
    # Update weights
    opt.step()
    
    # Store history of average training loss
    if (i % print_every == 0):
        loss /= print_every
        print('{} % of the simulation complete'.format(round(i/n_batch*100)))
        print('Loss {:0.3f}'.format(loss))
        loss_hist[k] = loss
        loss = 0; k += 1