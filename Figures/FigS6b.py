"""
Reproduce Figure S6b.
"""

# Imports

import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from matplotlib.ticker import MultipleLocator

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

data_path = str(Path(os.getcwd()).parent) + '/Data/FigS6/HighDim/'

spars_D2 = np.load(data_path+'sparsity_D2.npz')
spars_D4 = np.load(data_path+'sparsity_D4.npz')
spars_D6 = np.load(data_path+'sparsity_D6.npz')
spars_D8 = np.load(data_path+'sparsity_D8.npz')
spars_D10 = np.load(data_path+'sparsity_D10.npz')

# Sparsity scores

D2 = spars_D2['sparsity']
D4 = spars_D4['sparsity']
D6 = spars_D6['sparsity']
D8 = spars_D8['sparsity']
D10 = spars_D10['sparsity']
n_tasks = spars_D2['n_tasks']
n_dim = [2,4,6,8,10]

sparsity = np.stack([D2,D4,D6,D8,D10])

# Plot

colors = ['tab:blue','firebrick','purple','green']
SMALL_SIZE = 10

n_runs = sparsity.shape[2]

jitter_scale = 0.3  # Adjust for desired amount of spread
fig, ax = plt.subplots(figsize=(2.5,2))
for j in range(len(n_tasks)):
    for i in range(n_runs):
        jitter = np.random.uniform(-jitter_scale, jitter_scale, sparsity[:,j,:].shape)
        ax.scatter(n_dim + jitter[:, i], sparsity[:, j, i], color = colors[j],
                   s=10, label=n_tasks[j] if i == 0 else None)
#ax.axhline(0.9,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
#ax.axhline(0.997,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
ax.set_ylabel('Sparsity ($\%$)')
ax.set_xlabel('Input dimensionality $D$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 1.2))
ax.spines['bottom'].set_position(('data', -.02))
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.set_ylim([0,100])
plt.legend(prop={'size': SMALL_SIZE},frameon=False,ncol=1,title='# tasks',loc=(1.2,.15))
#plt.savefig('sparsity_HighDim.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('sparsity_HighDim.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
plt.show()