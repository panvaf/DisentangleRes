"""
Reproduce Figure 3b,c.
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

data_path = str(Path(os.getcwd()).parent) + '/Data/Fig3b,c/'

leak = np.load(data_path+'r_sq_RNN_CELoss.npz')
noLeak = np.load(data_path+'r_sq_noLeak_CELoss.npz')
LSTM = np.load(data_path+'r_sq_LSTM_CELoss.npz')
gpt = np.load(data_path+'r_sq_GPT.npz')


# Compute percentiles

r_sq_l = leak['r_sq'][:5]
r_sq_nl = noLeak['r_sq']
r_sq_LSTM = LSTM['r_sq']
r_sq_gpt = gpt['r_sq'][:5]
n_tasks = noLeak['n_tasks']

perc_l = np.percentile(r_sq_l,[25,50,75],axis=(1,2,3))
perc_nl = np.percentile(r_sq_nl,[25,50,75],axis=(1,2,3))
perc_LSTM = np.percentile(r_sq_LSTM,[25,50,75],axis=(1,2,3))
perc_gpt = np.percentile(r_sq_gpt,[25,50,75],axis=(1,2,3))

# Plot

offset = 0.1; off_p = 1+offset; off_m = 1-offset 

fig, ax = plt.subplots(figsize=(2.5,2))
ax.scatter(n_tasks,perc_l[1],label='RNN')
ax.errorbar(n_tasks,perc_l[1],yerr=[perc_l[1]-perc_l[0],perc_l[2]-perc_l[1]],linestyle='')
ax.scatter(n_tasks*off_m,perc_nl[1],color='firebrick',label='Non-leaky RNN')
ax.errorbar(n_tasks*off_m,perc_nl[1],yerr=[perc_nl[1]-perc_nl[0],
                    perc_nl[2]-perc_nl[1]],linestyle='',color='firebrick')
ax.scatter(n_tasks*off_p,perc_LSTM[1],color='green',label='LSTM')
ax.errorbar(n_tasks*off_p,perc_LSTM[1],yerr=[perc_LSTM[1]-perc_LSTM[0],
                    perc_LSTM[2]-perc_LSTM[1]],linestyle='',color='green')
ax.scatter(n_tasks,perc_gpt[1],label='GPT',color='darkgoldenrod')
ax.errorbar(n_tasks,perc_gpt[1],yerr=[perc_gpt[1]-perc_gpt[0],perc_gpt[2]-perc_gpt[1]],linestyle='',color='darkgoldenrod')
ax.axhline(0.9,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
ax.axhline(0.997,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
ax.set_ylabel('Out-of-distribution $r^2$')
ax.set_xlabel('# of tasks')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 1.5))
ax.spines['bottom'].set_position(('data', .48))
ax.set_xscale("log")
ax.set_xticks([2,10,30])
ax.set_xticklabels([2,10,30])
plt.ylim([0.5,1])
plt.legend(frameon=False,ncol=1,bbox_to_anchor=(1.2,.7))
#plt.savefig('r_sq_arc.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('r_sq_arc.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
plt.show()


# angles vs architectures

RNN_ang = np.load(data_path+'decoders_RNN_CEloss.npz')
gpt_ang = np.load(data_path+'decoders_GPT.npz')


# Compute percentiles

angles_RNN = RNN_ang['angles']
angles_gpt = gpt_ang['angles'][:5]
n_tasks = RNN_ang['n_tasks']

perc_RNN = np.nanpercentile(angles_RNN,[25,50,75],axis=(1,2,3,4,5))
perc_gpt = np.nanpercentile(angles_gpt,[25,50,75],axis=(1,2,3,4,5))
n_dim = 2

# plot

offset = 0.1; off_p = 1+offset; off_m = 1-offset 

fig, ax = plt.subplots(figsize=(2.5,2))
ax.scatter(n_tasks*off_m,perc_RNN[1],label='RNN')
ax.errorbar(n_tasks*off_m,perc_RNN[1],yerr=[perc_RNN[1]-perc_RNN[0],perc_RNN[2]-perc_RNN[1]],linestyle='')
ax.scatter(n_tasks*off_p,perc_gpt[1],color='darkgoldenrod',label='GPT')
ax.errorbar(n_tasks*off_p,perc_gpt[1],yerr=[perc_gpt[1]-perc_gpt[0],perc_gpt[2]-perc_gpt[1]],linestyle='',color='darkgoldenrod')
ax.axhline(90,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
ax.set_ylabel('Angle of latents (deg)')
ax.set_xlabel('# of tasks')
ax.set_title('Input dimensionality $D={}$'.format(n_dim))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 1.5))
#ax.spines['bottom'].set_position(('data', .48))
ax.set_xscale("log")
ax.set_xticks([2,10,30])
ax.set_xticklabels([2,10,30])
plt.legend(frameon=False)
#plt.savefig('ang_arc.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('ang_arc.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
plt.show()
