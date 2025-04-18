"""
Reproduce Figure 5c.
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

data_path = str(Path(os.getcwd()).parent) + '/Data/Fig5c/'

n1 = np.load(data_path+'r_sq_fixed_NetN0_Enc_n1.npz')
n2 = np.load(data_path+'r_sq_fixed_NetN0_Enc_n2.npz')
n3 = np.load(data_path+'r_sq_fixed_NetN0_Enc_n3.npz')
n4 = np.load(data_path+'r_sq_fixed_NetN0_Enc_n4.npz')
n6 = np.load(data_path+'r_sq_fixed_NetN0_Enc_n6.npz')


# Compute percentiles

r_sq_n1 = n1['r_sq']
r_sq_n2 = n2['r_sq']
r_sq_n3 = n3['r_sq']
r_sq_n4 = n4['r_sq']
r_sq_n6 = n6['r_sq']
n_tasks = n1['n_tasks']
n_std = ['0.1','0.2','0.3','0.4','0.6']

perc_n1 = np.percentile(r_sq_n1,[25,50,75],axis=(1,2,3))
perc_n2 = np.percentile(r_sq_n2,[25,50,75],axis=(1,2,3))
perc_n3 = np.percentile(r_sq_n3,[25,50,75],axis=(1,2,3))
perc_n4 = np.percentile(r_sq_n4,[25,50,75],axis=(1,2,3))
perc_n6 = np.percentile(r_sq_n6,[25,50,75],axis=(1,2,3))

perc = np.stack([perc_n1, perc_n2, perc_n3, perc_n4, perc_n6],axis=1)

# Plot

colors = ['tab:blue','firebrick','green']
SMALL_SIZE = 10
ofst = .15
offsets = [0,-ofst,ofst]
x_pos = np.arange(len(n_std))

fig, ax = plt.subplots(figsize=(2.5,2))
for i in range(perc.shape[2]):
    x_vals = x_pos + offsets[i]
    ax.scatter(x_vals, perc[1,:,i], color=colors[i], label=n_tasks[i], zorder=3)
    ax.errorbar(x_vals, perc[1,:,i],
                yerr=[perc[1,:,i] - perc[0,:,i], perc[2,:,i] - perc[1,:,i]],
                linestyle='', color=colors[i], zorder=2)
    ax.plot(x_vals, perc[1,:,i], linestyle='dotted', color=colors[i], zorder=1)

# Relabel x-ticks to the original string values
ax.set_xticks(x_pos)
ax.set_xticklabels(n_std)
#ax.axhline(0.9,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
#ax.axhline(0.997,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
ax.set_ylabel('Out-of-distribution $r^2$')
ax.set_xlabel('Training noise std $\sigma$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.5))
ax.spines['bottom'].set_position(('data', .68))
ax.yaxis.set_major_locator(MultipleLocator(.1))
ax.set_ylim([0.7,1])
plt.legend(prop={'size': SMALL_SIZE},frameon=False,ncol=1,title='# tasks',loc=(1.2,.15))
#plt.savefig('r_sq_nStd.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('r_sq_nStd.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
plt.show()