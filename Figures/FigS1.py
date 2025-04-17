"""
Reproduce Figure S1.
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

data_path = str(Path(os.getcwd()).parent) + '/Data/FigS1/'

quad = np.load(data_path+'r_sq_Quad.npz')
relu = np.load(data_path+'r_sq_RNN_CELoss.npz')


# Compute percentiles

r_sq_quad = quad['r_sq']
r_sq_relu = relu['r_sq']
n_tasks = relu['n_tasks']

perc_quad = np.percentile(r_sq_quad,[25,50,75],axis=(1,2,3))
perc_relu = np.percentile(r_sq_relu,[25,50,75],axis=(1,2,3))

# Plot

offset = 0.1; off_p = 1+offset; off_m = 1-offset 

fig, ax = plt.subplots(figsize=(2.5,2))
ax.scatter(n_tasks*off_p,perc_relu[1],label='ReLU')
ax.errorbar(n_tasks*off_p,perc_relu[1],yerr=[perc_relu[1]-perc_relu[0],
                    perc_relu[2]-perc_relu[1]],linestyle='')
ax.scatter(n_tasks*off_m,perc_quad[1],color='firebrick',label='Quadratic')
ax.errorbar(n_tasks*off_m,perc_quad[1],yerr=[perc_quad[1]-perc_quad[0],
                    perc_quad[2]-perc_quad[1]],linestyle='',color='firebrick')
ax.axhline(0.9,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
ax.axhline(0.997,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
ax.set_ylabel('$r^2$')
ax.set_xlabel('# of tasks')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 1.5))
ax.spines['bottom'].set_position(('data', .48))
ax.set_xscale("log")
ax.set_xticks([2,10,30])
ax.set_xticklabels([2,10,30])
plt.ylim([0.5,1])
plt.legend(frameon=False,ncol=1,bbox_to_anchor=(1,.6),title='Nonlinearity')
#plt.savefig('r_sq_ID_OOD.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('r_sq_ID_OOD.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
plt.show()