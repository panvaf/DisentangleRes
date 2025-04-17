"""
Reproduce Figure 5b, S12 RNN.
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

data_path = str(Path(os.getcwd()).parent) + '/Data/Fig5b,S12 RNN/'

D2 = np.load(data_path+'r_sq_CE_d2.npz')
D4 = np.load(data_path+'r_sq_CE_d4.npz')
D6 = np.load(data_path+'r_sq_CE_d6.npz')
D8 = np.load(data_path+'r_sq_CE_d8.npz')
D10 = np.load(data_path+'r_sq_CE_d10.npz')


# Compute percentiles

r_sq_2 = D2['r_sq']
r_sq_4 = D4['r_sq']
r_sq_6 = D6['r_sq']
r_sq_8 = D8['r_sq']
r_sq_10 = D10['r_sq']
n_tasks = D2['n_tasks']
n_dim = [2,4,6,8,10]

perc_2 = np.percentile(r_sq_2,[25,50,75],axis=(1,2,3))
perc_4 = np.percentile(r_sq_4,[25,50,75],axis=(1,2,3))
perc_6 = np.percentile(r_sq_6,[25,50,75],axis=(1,2,3))
perc_8 = np.percentile(r_sq_8,[25,50,75],axis=(1,2,3))
perc_10 = np.percentile(r_sq_10,[25,50,75],axis=(1,2,3))

perc = np.stack([perc_2, perc_4, perc_6, perc_8, perc_10],axis=1)

# Plot

colors = ['tab:blue','firebrick','purple','green']
SMALL_SIZE = 10

fig, ax = plt.subplots(figsize=(2.5,2))
for i in range(perc.shape[2]):
    ax.scatter(n_dim,perc[1,:,i],color=colors[i],label=n_tasks[i])
    ax.errorbar(n_dim,perc[1,:,i],yerr=[perc[1,:,i]-perc[0,:,i],perc[2,:,i]-perc[1,:,i]],linestyle='',color=colors[i])
    ax.plot(n_dim,perc[1,:,i],linestyle='dotted',color=colors[i])
#ax.axhline(0.9,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
#ax.axhline(0.997,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
ax.set_ylabel('Out-of-distribution $r^2$')
ax.set_xlabel('Input dimensionality $D$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 1.2))
ax.spines['bottom'].set_position(('data', -.02))
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(.2))
ax.set_ylim([0,1])
plt.legend(prop={'size': SMALL_SIZE},frameon=False,ncol=1,title='# tasks',loc=(1.2,.15))
#plt.savefig('r_sq_HighDim.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('r_sq_HighDim.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
plt.show()


# decoder angles

D2_ang = np.load(data_path+'decoders_CE_d2.npz')
D4_ang = np.load(data_path+'decoders_CE_d4.npz')
D6_ang = np.load(data_path+'decoders_CE_d6.npz')
D8_ang = np.load(data_path+'decoders_CE_d8.npz')
D10_ang = np.load(data_path+'decoders_CE_d10.npz')

# Compute percentiles

angles_D2 = D2_ang['angles']
angles_D4 = D4_ang['angles']
angles_D6 = D6_ang['angles']
angles_D8 = D8_ang['angles']
angles_D10 = D10_ang['angles']
n_tasks = D2_ang['n_tasks']

perc_2_ang = np.nanpercentile(angles_D2,[25,50,75],axis=(1,2,3,4,5))
perc_4_ang = np.nanpercentile(angles_D4,[25,50,75],axis=(1,2,3,4,5))
perc_6_ang = np.nanpercentile(angles_D6,[25,50,75],axis=(1,2,3,4,5))
perc_8_ang = np.nanpercentile(angles_D8,[25,50,75],axis=(1,2,3,4,5))
perc_10_ang = np.nanpercentile(angles_D10,[25,50,75],axis=(1,2,3,4,5))

perc_ang = np.stack([perc_2_ang, perc_4_ang, perc_6_ang, perc_8_ang, perc_10_ang],axis=1)

# plot

colors = ['tab:blue','firebrick','purple','green']
SMALL_SIZE = 10

ffig, ax = plt.subplots(figsize=(2.5,2))
for i in range(perc_ang.shape[2]):
    ax.scatter(n_dim,perc_ang[1,:,i],color=colors[i],label=n_tasks[i])
    ax.errorbar(n_dim,perc_ang[1,:,i],yerr=[perc_ang[1,:,i]-perc_ang[0,:,i],
                perc_ang[2,:,i]-perc_ang[1,:,i]],linestyle='',color=colors[i])
    ax.plot(n_dim,perc_ang[1,:,i],linestyle='dotted',color=colors[i])
ax.axhline(90,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
ax.set_ylabel('Angle of latents (deg)')
ax.set_xlabel('Input dimensionality $D$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 1.2))
#ax.spines['bottom'].set_position(('data', -.02))
ax.xaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_major_locator(MultipleLocator(15))
#ax.yaxis.set_minor_locator(MultipleLocator(15))
ax.set_ylim([60,120])
plt.legend(prop={'size': SMALL_SIZE},frameon=False,ncol=1,title='# tasks',loc=(1.2,.15))
#plt.savefig('dec_HighDim.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('dec_HighDim.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
plt.show()
