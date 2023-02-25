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
task = {"LinearClassification":tasks.LinearClassification}
#task_rules = util.assign_task_rules(task)
n_task = len(task)

# Constants
n_neu = 64          # number of recurrent neurons
batch_sz = 16       # batch size
n_batch = 2e4       # number of batches
dt = 100            # step size
tau = 100           # neuronal time constant (synaptic+membrane)
n_sd = 2            # standard deviation of injected noise
n_ff = 100          # size of feedforward layer
print_every = int(n_batch/100)
n_out = 2          # number of outputs per task
bal_err = False     # whether to balance penalization of decision vs. integration

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}
grace = 200
#thres = np.array([0.005, 0.02, 0.04, 0.07, 0.11, 0.15])\
thres = np.array([0.005, 0.01, 0.018, 0.027, 0.04, 0.052, 0.07, 0.085, 0.105, 0.125, 0.15, 0.18])
#trial_sz = int(sum(timing.values())/dt) 
trial_sz = 88
n_grace = int(grace/dt); n_decision = int(timing['decision']/dt); n_trial = int(sum(timing.values())/dt)

# Save location
data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
net_file = 'LinCenter' + str(n_neu) + \
            (('batch' + format(n_batch,'.0e').replace('+0','')) if not n_batch==1e4 else '') + \
            (('Noise' + str(n_sd)) if n_sd else '') + \
            (('tau' + str(tau)) if tau != 100 else '') + \
            (('nTask' + str(n_out)) if n_out != 2 else '')  + \
            (('Delay' + str(timing['delay'])) if timing['delay'] != 0 else '')  + \
            ('BalErr' if bal_err else '')
     
# Make supervised datasets
tenvs = [value(timing=timing,sigma=n_sd,n_task=n_out, thres=thres) for key, value in task.items()]
#tenvs = ['PerceptualDecisionMaking-v0']
#kwargs = {'dt': 100, 'sigma': 1}

datasets = [ngym.Dataset(tenv,batch_size=batch_sz,seq_len=trial_sz) for tenv in tenvs]

# A sample environment from dataset
env = datasets[0].env
# Visualize the environment with 2 sample trials
#_ = ngym.utils.plot_env(env, num_trials=2)

# Network input and output size
n_in = env.observation_space.shape[0]
#n_out = env.action_space.n

# Mask to weight errors during integration and decision equally
if bal_err:
    mask_w = (sum(timing.values()) - grace - timing['decision'])/timing['decision']
    mask = np.ones((batch_sz,n_trial,1)); mask[:,-n_decision-n_grace:-n_decision] = 0
    mask[:,-n_decision:] = mask_w; mask = np.tile(mask,(1,4,n_out))
else:
    mask = np.ones((batch_sz,trial_sz,n_out))
    
# Initialize RNN  
net = RNN(n_in,n_neu,n_out*n_task,n_sd,tau,dt)

# Feedforward NN
ff_net = nn.Sequential(
        nn.Linear(n_out*n_task,n_out*n_task)
        #nn.Sigmoid(),
        #nn.Linear(n_ff,n_out*n_task)
        )

# Optimizer
opt = optim.Adam(net.parameters(), lr=0.003)
opt.add_param_group({'params': ff_net.parameters()})

# Loss
#criterion = util.CrossEntropyLoss()

# Train RNN
total_loss = 0; k = 0
loss_hist = np.zeros(100)

for i in range(int(n_batch)):
    # Randomly pick task
    task = randint(0,n_task-1)
    dataset = datasets[task]
    # Generate data for current batch
    inputs, target = dataset()
    
    # Reshape so that batch is first dimension
    inputs = np.transpose(inputs,(1,0,2))
    target = np.transpose(target,(1,0,2))
    
    # Reshape for multiple tasks
    masker = np.zeros((batch_sz,trial_sz,n_out*n_task))
    masker[:,:,task*n_out:(task+1)*n_out] = mask
    targets = np.zeros((batch_sz,trial_sz,n_out*n_task))
    targets[:,:,task*n_out:(task+1)*n_out] = target
    
    # Turn into tensors
    inputs = torch.from_numpy(inputs).type(torch.float)
    targets = torch.from_numpy(targets).type(torch.long)
    masker = torch.from_numpy(masker).type(torch.long)
    
    # Empty gradient buffers
    opt.zero_grad()
    
    # Forward run
    net_out, fr = net(inputs)
    output = ff_net(net_out)
    
    # Compute loss
    #loss = criterion(output.view(-1,n_out),target.flatten())
    loss, _ = util.MSELoss_weighted(output, targets, masker)
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