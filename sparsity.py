"""
Estimate sparsity of trained networks.
"""

import torch
import torch.nn as nn
from RNN import RNN
import os
from pathlib import Path
import numpy as np
import util
import tasks
import neurogym as ngym

# Parameters
n_neu = 64          # number of recurrent neurons
dt = 100            # step size
tau = 100           # neuronal time constant (synaptic+membrane)
n_sd_in = 0         # standard deviation of input noise
n_dim = 2          # dimensionality of state space
n_in = n_dim + 1    # number of inputs
n_sample = 1000     # Number of stimuli values to sample
n_runs = 5
filename = 'LinProbSigmoidSL64batch1e5LR0.001Noise2NetN0nTrial1CElossnTask'
leaky = False if 'NoLeak' in filename else True
encode = True

if encode:
    n_feat = 40 + (1 if n_in>n_dim else 0)
else:
    n_feat = n_in

# Tasks
task = {"LinearClassificationHighDim":tasks.LinearClassificationHighDim}
task_rules = util.assign_task_rules(task)
task_num = len(task)

n_tasks = np.array([3,6,9,12])

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}

t_task = int(sum(timing.values())/dt)

# Device

device = torch.device('cpu')

# initialize
sparsity = np.zeros((len(n_tasks),n_runs))
sparsity_gallant = np.zeros((len(n_tasks),n_runs))

for n, n_task in enumerate(n_tasks):
    
    # Environments
    tenvs = [value(timing=timing,sigma=n_sd_in,n_task=n_task,
                         n_dim=n_dim) for key, value in task.items()]
    
    # Datasets
    datasets = [ngym.Dataset(tenv,batch_size=n_sample,
                     seq_len=t_task) for tenv in tenvs]
    dataset = datasets[0]
    
    for run in np.arange(n_runs):
    
        print('Network {} out of {} trained on {} tasks'.format(run+1,n_runs,n_task))
        
        # Load network
        data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
        net_file = filename + str(n_task) + ('Mix' if encode else '') + (('run' + str(run)) if run != 0 else '')
        
        if 'LSTM' in net_file:
            net = nn.LSTM(n_feat,n_neu,batch_first=True).to(device)
        else:
            net = RNN(n_feat,n_neu,n_task,0,'relu',tau,dt,leaky).to(device)
        checkpoint = torch.load(os.path.join(data_path,net_file + '.pth'))
        net.load_state_dict(checkpoint['state_dict'])
        
        if encode:
            # Initialize encoder
            encoder = nn.Sequential(
                    nn.Linear(n_dim,100),
                    nn.ReLU(),
                    nn.Linear(100,100),
                    nn.ReLU(),
                    nn.Linear(100,40)
                    )
            
            encoder.load_state_dict(checkpoint['encoder'])
                
            
        inputs, _ = dataset()
        inputs = np.transpose(inputs,(1,0,2))[:,:,-n_in:]
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
                    
        # Forward pass
        if encode:
            inputs = util.encode(encoder,inputs,n_dim,n_in)
    
        fr, _ = net(inputs)
        fr = fr[:,-2,:].detach().numpy()
        
        sparsity[n,run] = (fr>0).sum()/fr.size*100
        
        sparsity_gallant[n,run] = np.mean(util.compute_sparseness(fr))*100