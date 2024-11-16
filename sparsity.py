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
import matplotlib.pyplot as plt

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
save = False

if encode:
    n_feat = 40 + (1 if n_in>n_dim else 0)
else:
    n_feat = n_in

# Tasks
task = {"LinearClassificationHighDim":tasks.LinearClassificationHighDim}
task_rules = util.assign_task_rules(task)
task_num = len(task)

n_tasks = np.array([2,3,6,12,24])

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
sparsity_gallant = np.zeros((len(n_tasks),n_runs,n_neu))

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
        
        sparsity_gallant[n,run] = util.compute_sparseness(fr)*100
        

# plots

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

# Compute mean and confidence intervals for each number of tasks
means, ci = util.network_mean_and_ci(sparsity_gallant)

# add jitter
jitter_scale = 0.1  # Adjust for desired amount of spread
jitter = np.random.uniform(-jitter_scale, jitter_scale, means.shape)

# plot
fig, ax = plt.subplots(figsize=(2.5,2))
for i in range(means.shape[1]):  # Loop through the 5 entries per task
    ax.scatter(n_tasks * (1 + jitter[:, i]), means[:, i], color = 'tab:blue', s=10, label=f'Category {i+1}' if i == 0 else None)
ax.set_ylabel('Sparsity ($\%$)')
ax.set_xlabel('# of tasks')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 1.5))
ax.spines['bottom'].set_position(('data', -5))
ax.set_xscale("log")
ax.set_xticks([2,10,30])
ax.set_xticklabels([2,10,30])
plt.ylim([0,100])
#plt.legend(frameon=False,ncol=1,bbox_to_anchor=(1,.4),title='RT')
#plt.savefig('r_squared.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('r_squared.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
plt.show()

# Save
if save:
    np.savez('sparsity.npz',sparsity=means,confidence_intervals=ci,n_tasks=n_tasks)