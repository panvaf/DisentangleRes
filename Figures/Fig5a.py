"""
Reproduce Figure 5a.
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

data_path = str(Path(os.getcwd()).parent) + '/Data/Fig5a/'

freeRT = np.load(data_path+'r_sq_free_timing_enc.npz')

# Compute percentiles

r_sq = freeRT['r_sq']
timing = freeRT['timing']

perc = np.percentile(r_sq,[25,50,75],axis=(1,2,3))
theor = 1 - 0.48/(timing+1) # equation 7 from paper, +1 because we have decision time

# Plot

fig, ax = plt.subplots(figsize=(2.5,2))
ax.scatter(timing/10,perc[1],label='Experiment')
ax.errorbar(timing/10,perc[1],yerr=[perc[1]-perc[0],
                    perc[2]-perc[1]],linestyle='')
ax.scatter(timing/10,theor,80,marker='_',color='firebrick',label='Theory')
ax.set_ylabel('Out-of-distribution $r^2$')
ax.set_xlabel('Trial duration (s)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', .35))
ax.spines['bottom'].set_position(('data', .68))
plt.ylim([0.7,1])
plt.legend(frameon=False,ncol=1,bbox_to_anchor=(1,.4))
#plt.savefig('r_sq_free_timing.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('r_sq_free_timing.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
plt.show()