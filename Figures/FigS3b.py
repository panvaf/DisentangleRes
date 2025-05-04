"""
Reproduce Figure S3b.
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

data_path = str(Path(os.getcwd()).parent) + '/Data/FigS3b/'

corr = np.load(data_path+'r_sq_fixed_NetN0_Enc_corr.npz')

# Compute percentiles

perc = corr['perc']
cor = corr['corr']

# Plot

fig, ax = plt.subplots(figsize=(2.5,2))
ax.scatter(cor,perc[1])
ax.errorbar(cor,perc[1],yerr=[perc[1]-perc[0],perc[2]-perc[1]],linestyle='')
ax.axhline(0.9,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
ax.axhline(0.997,color='lightblue',linestyle='--',zorder=-1,linewidth=1)
ax.set_ylabel('Out-of-distribution $r^2$')
ax.set_xlabel('Factor correlation')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -0.3))
ax.spines['bottom'].set_position(('data', .48))
plt.ylim([0.5,1])
#plt.savefig('r_sq_corr.png',bbox_inches='tight',format='png',dpi=300,transparent=True)
#plt.savefig('r_sq_corr.eps',bbox_inches='tight',format='eps',dpi=300,transparent=True)
plt.show()