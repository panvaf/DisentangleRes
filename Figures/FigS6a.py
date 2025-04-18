"""
Reproduce Figure S6a.
"""

# Imports

import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np

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


# Load data

data_path = str(Path(os.getcwd()).parent) + '/Data/FigS6/arc/'

spars_leak = np.load(data_path+'sparsity_RNN.npz')
spars_NoLeak = np.load(data_path+'sparsity_RNN_NoLeak.npz')
spars_LSTM = np.load(data_path+'sparsity_LSTM.npz')

# Sparsity scores

leak = spars_leak['sparsity']
noLeak = spars_NoLeak['sparsity']
LSTM = spars_LSTM['sparsity']
n_tasks = spars_leak['n_tasks']

# Plot

jitter_scale = 0.1  # Adjust for desired amount of spread
fig, ax = plt.subplots(figsize=(2.5,2))
for i in range(leak.shape[1]):
    jitter = np.random.uniform(-jitter_scale, jitter_scale, leak.shape)
    ax.scatter(n_tasks * (1 + jitter[:, i]), leak[:, i], color = 'tab:blue',
               s=10, label='RNN' if i == 0 else None)
for i in range(noLeak.shape[1]):
    jitter = np.random.uniform(-jitter_scale, jitter_scale, noLeak.shape)
    ax.scatter(n_tasks * (1 + jitter[:, i]), noLeak[:, i], color = 'firebrick',
               s=10, label='Non-leaky RNN' if i == 0 else None)
for i in range(LSTM.shape[1]):
    jitter = np.random.uniform(-jitter_scale, jitter_scale, LSTM.shape)
    ax.scatter(n_tasks * (1 + jitter[:, i]), LSTM[:, i], color = 'green',
               s=10, label='LSTM' if i == 0 else None)
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
plt.legend(frameon=False,ncol=1,bbox_to_anchor=(1,.5))
#plt.savefig('sparsity_arc.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('sparsity_arc.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
plt.show()