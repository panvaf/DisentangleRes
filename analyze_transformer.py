"""
Analyze transformer activations
"""

import torch
import torch.nn as nn
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import util
from transformers import GPT2Config
from transformer import GPT2ContinuousInputs
from matplotlib.colors import LinearSegmentedColormap

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


# Parameters
d_model = 64        # embedding dimension
n_layer = 1         # number of layers
n_head = 8          # number of heads
n_dim = 2           # dimensionality of state space
t_task = 22         # trial duration
n_in = n_dim + 1    # number of inputs
n_task = 24         # number of tasks
n_sweep = 8         # Number of stimuli values to sweep
run = 0

n_feat = 40 + (1 if n_in>n_dim else 0)

# device
device = torch.device('cpu')

# Load network
data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
#net_file = 'Joint64batch1e3'
net_file = 'LinProbSigmoidSL64gpt-2batch2e4LR0.001Noise2NetN0nTrial1nLayer1nHead8CElossnTask' + \
            str(n_task) + 'Mix'  + (('run' + str(run)) if run != 0 else '')
            
# Encoder
encoder = nn.Sequential(
        nn.Linear(n_dim,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,40)
        )

config = GPT2Config(
    vocab_size=1,
    n_embd=d_model,
    n_layer=n_layer,
    n_head=n_head,
    n_positions=t_task,
    n_ctx=t_task,
    n_in=n_feat,
)
net = GPT2ContinuousInputs(config).to(device)

checkpoint = torch.load(os.path.join(data_path,net_file + '.pth'))
net.load_state_dict(checkpoint['state_dict'])
net.eval()

encoder.load_state_dict(checkpoint['encoder'])
encoder.eval()


# Activations raster plot

act = np.zeros((n_sweep,n_sweep,n_head,int(d_model/n_head)))
x1s = np.linspace(-.5,.5,n_sweep)
x2s = np.linspace(-.5,.5,n_sweep)

for i, x1 in enumerate(x1s):
    for j, x2 in enumerate(x2s):
        
        # Construct inputs
        inp = np.tile([1, x1, x2],(1, t_task, 1))
        inp = torch.tensor(inp, dtype=torch.float32)
        
        inp = util.encode(encoder,inp,n_dim,n_in)
        
        transformer_outputs = net(
            inputs_embeds=inp,
            return_dict=True,
        )
        
        values = transformer_outputs.past_key_values[0][1]
        
        act[i,j] = values[0, :, -1, :].detach().numpy()
        
# Correlations
act_flat = act.reshape((n_sweep,n_sweep,-1))
x1corr = util.corr2_coeff(x1s[np.newaxis,:],np.mean(act_flat,axis=1).T)
x2corr = util.corr2_coeff(x2s[np.newaxis,:],np.mean(act_flat,axis=0).T)

# Reshape        
x1corr = np.reshape(x1corr,(n_head,int(d_model/n_head)))
x2corr = np.reshape(x2corr,(n_head,int(d_model/n_head)))

# First, compute the maximum absolute value across all act[i,j]
max_abs = max(np.abs(act[i,j]).max() for i in range(n_sweep) for j in range(n_sweep))

# Define a custom diverging colormap from grey to gold
cmap = LinearSegmentedColormap.from_list('GreyGold', ['darkgoldenrod', 'white', 'dimgrey'])

# Plot activity
fig, axes = plt.subplots(8, 8, figsize=(8, 8))

for i in range(n_sweep):
    for j in range(n_sweep):
        ax = axes[-j-1,i]
        im = ax.imshow(act[i,j], cmap=cmap, vmin=-max_abs, vmax=+max_abs)
        ax.set_xticks([])
        ax.set_yticks([])

# Add labels to the top-left plot (first row, first column)
axes[-1, 0].set_xlabel('Heads')
axes[-1, 0].set_ylabel('Neurons')

# Create a common colorbar
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Final embedding activations')

# Common axis labels
fig.text(0.5, 0.08, '$x_1$', ha='center')
fig.text(0.08, 0.5, '$x_2$', va='center', rotation='vertical')

#plt.savefig('activ_gpt.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('activ_gpt.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
plt.show()


# Plot correlations

fig, axes = plt.subplots(1, 2, figsize=(4, 2))

im1 = axes[0].imshow(x1corr, cmap='RdBu')
axes[0].set_title('$x_1$')
axes[0].set_xticks([])
axes[0].set_yticks([])        

im2 = axes[1].imshow(x2corr, cmap='RdBu')
axes[1].set_title('$x_2$')
axes[1].set_xticks([])
axes[1].set_yticks([])        

# Create a common colorbar
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
cbar = plt.colorbar(im1, cax=cax)
cbar.set_label('Correlation coefficient')

#plt.savefig('corr_gpt.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('corr_gpt.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)

plt.show()