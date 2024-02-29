"""
Analyze learned RNNs.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from RNN import RNN
import os
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import util
import tasks
import matplotlib.pyplot as plt
from random import randint
import neurogym as ngym
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as mcol

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
n_neu = 64          # number of recurrent neurons
dt = 100            # step size
tau = 100           # neuronal time constant (synaptic+membrane)
n_sd_in = 2         # standard deviation of input noise
n_dim = 2           # dimensionality of state space
n_in = n_dim + 1    # number of inputs
n_task = 48         # number of tasks
n_trial = 40        # number of bulk example trials to plot
n_exam = 5          # number of example points to plot with separate colors
thres = 5           # DDM boundary
n_sweep = 8         # Number of stimuli values to sweep
encode = False
activation = 'relu'
run = 0

if encode:
    n_feat = 40 + (1 if n_in>n_dim else 0)
else:
    n_feat = n_in

# Tasks
task = {"LinearClassificationCentOut":tasks.LinearClassificationCentOut}
task_rules = util.assign_task_rules(task)
task_num = len(task)

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}

t_task = int(sum(timing.values())/dt)

#thres = np.array([0.005, 0.01, 0.018, 0.027, 0.04, 0.052, 0.07, 0.085, 0.105, 0.125, 0.15, 0.18])
tenvs = [value(timing=timing,sigma=0,n_task=n_task,n_dim=n_dim,thres=thres,
               rule_vec=task_rules[key]) for key, value in task.items()]

# Load network
data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
#net_file = 'Joint64batch1e3'
net_file = 'LinCentOutTanhSL64batch1e5LR0.001Noise2NetN0nTrial1nTask' + str(n_task) + \
            ('Mix' if encode else '')  + (('run' + str(run)) if run != 0 else '')
            
# Encoder
encoder = nn.Sequential(
        nn.Linear(n_dim,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,40)
        )

net = RNN(n_feat,n_neu,task_num*n_task,0,activation,tau,dt)
checkpoint = torch.load(os.path.join(data_path,net_file + '.pth'))
net.load_state_dict(checkpoint['state_dict'])
net.eval()

if encode:
    encoder.load_state_dict(checkpoint['encoder'])
    encoder.eval()

# Visualize RNN activity
activity_dict = {}; output_dict = {}; trial_info = {}

for i in range(n_trial):
    
    # Pick environment and generate a trial
    tenv = tenvs[randint(0,task_num-1)]
    tenv.new_trial(); ob = tenv.ob
    inp = torch.from_numpy(ob[np.newaxis, :, :]).type(torch.float)[:,:,-n_in:]
    if encode:
        inp = util.encode(encoder,inp,n_dim,n_in)
    output, rnn_activity = net(inp)
    rnn_activity = rnn_activity[0, :, :].detach().numpy()
    output = output[0, :, :].detach().numpy()
    activity_dict[i] = rnn_activity
    output_dict[i] = output
    trial_info[i] = tenv.trial

activity = np.concatenate(list(activity_dict[i] for i in range(n_trial)), axis=0)

# Perform PCA
pca = PCA(n_components=10)
pca.fit(activity)

# Access the explained variance ratio
explained_variance_ratio = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(1.5,1.5))
ax.plot(np.arange(1,10+1),100*explained_variance_ratio[:10])
ax.set_xlabel('Component #')
ax.set_ylabel('Variance %')
ax.set_title('PCA')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.show()

# Find approximate fixed points, depending on network initial conditions

# Freeze for parameters in the recurrent network
for param in net.parameters():
    param.requires_grad = False
    
for param in encoder.parameters():
    param.requires_grad = False

batch_size = 128
fixedpoints = np.empty([batch_size,task_num,n_neu])

for j in range(task_num):
    print('Task {} out of {}'.format(j+1,task_num))

    # Inputs are zero, so that internal representation is not affected
    if n_in>3:
        inp = np.tile(np.concatenate(([1, 0, 0],list(task_rules.values())[j])),(batch_size, 1))
    elif n_in==3:    
        inp = np.tile([1, 0, 0],(batch_size, 1))
    elif n_in==2:
        inp = np.tile([0, 0],(batch_size, 1))
    inp = torch.tensor(inp, dtype=torch.float32)
    
    if encode:
        inputs = encoder(inp[:,-n_dim:])
        if n_in > n_dim:
            inp = torch.cat((inp[:,0].unsqueeze(1),inputs),dim=1)
        else:
            inp = inputs
    
    # Initialize hidden activity                                                                    
    hdn = np.zeros((batch_size, n_neu))
    idx_tr = np.random.choice(n_trial,batch_size,replace=True)
    idx_t = np.random.choice(ob.shape[0],batch_size)
    for i in range(batch_size):
        hdn[i] = torch.from_numpy(activity_dict[idx_tr[i]][idx_t[i]])
    hidden = torch.tensor(np.random.rand(batch_size, n_neu)*5+hdn,
                      requires_grad=True, dtype=torch.float32)
    
    # Use Adam optimizer
    optimizer = optim.Adam([hidden], lr=0.01)
    criterion = nn.MSELoss()
    
    running_loss = 0
    for i in range(10000):
        optimizer.zero_grad()   # zero the gradient buffers
        
        # Take the one-step recurrent function from the trained network
        new_h = net.dynamics(inp, hidden)
        loss = criterion(new_h, hidden)
        loss.backward()
        optimizer.step()    # Does the update
    
        running_loss += loss.item()
        if i % 1000 == 999:
            running_loss /= 1000
            print('Step {}, Loss {:0.4f}'.format(i+1, running_loss))
            running_loss = 0
            
    fxdpoints = hidden.detach().numpy()
    fixedpoints[:,j] = fxdpoints

# Obtain individual simulations to plot and compare location of trajectories
tenvs = [value(timing=timing,sigma=n_sd_in,n_task=n_task,thres=thres,rule_vec=task_rules[key]) for key, value in task.items()]

datasets = [ngym.Dataset(tenv,batch_size=1,seq_len=t_task) for tenv in tenvs]

stims = np.zeros((n_exam,2))

# Visualize RNN activity
ex_activ_dict = {}; ex_trial_info = {}

for i in range(n_exam):
    
    # Randomly pick task
    dataset = datasets[randint(0,task_num-1)]
    # A sample environment from dataset
    env = dataset.env
    env.new_trial()
    
    ob = env.ob
    #ob[:,2] = ob[:,1] if np.random.random() > .5 else -ob[:,1]
    stims[i] = env.trial['stim']
    
    inp = torch.from_numpy(ob[np.newaxis, :, :]).type(torch.float)[:,:,-n_in:]
    if encode:
        inp = util.encode(encoder,inp,n_dim,n_in)
    _, rnn_activity = net(inp)
    rnn_activity = rnn_activity[0, :, :].detach().numpy()
    ex_activ_dict[i] = rnn_activity
    ex_trial_info[i] = env.trial


# Plot network activity and overlay approximate fixed points
colors = [['gold','limegreen'],['dodgerblue','lightcoral']]

plot_full = util.rot_3D_plot(activity_dict,fixedpoints,pca,n_trial,trial_info,
                        net_file,n_in=n_in,colors=colors)
plot_full.plot()


# Plot examples with noise with distinct colors and their location in state space 
cols = list(mcol.TABLEAU_COLORS.values())[:n_exam]

plot_ex = util.rot_3D_plot(ex_activ_dict,fixedpoints,pca,n_exam,ex_trial_info,
                        net_file,n_in=n_in,colors=cols)
plot_ex.plot()


# Examples in state space
fig, ax = plt.subplots(figsize=(1,1))
for i in range(n_exam):
    ax.scatter(stims[i,0],stims[i,1],s=15,c=cols[i])
ax.set_xlim([-.5,.5])
ax.set_ylim([-.5,.5])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.55))
ax.spines['bottom'].set_position(('data', -.55))
ax.xaxis.set_major_locator(MultipleLocator(.5))
ax.yaxis.set_major_locator(MultipleLocator(.5))
ax.set_xticklabels([])
ax.set_yticklabels([])
#plt.savefig('examples.eps',bbox_inches='tight',format='eps',dpi=300)
#plt.savefig('examples.png',bbox_inches='tight',format='png',dpi=300)
plt.show()


# Classification lines
fig, ax = plt.subplots(figsize=(2,2))
x = np.linspace(-.5,.5,100)
for alpha in tenvs[0].alphas:
    ax.plot(x,alpha*x)
ax.set_xlim([-.5,.5])
ax.set_ylim([-.5,.5])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_title('Classification lines')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.55))
ax.spines['bottom'].set_position(('data', -.55))
ax.xaxis.set_major_locator(MultipleLocator(.5))
ax.yaxis.set_major_locator(MultipleLocator(.5))

# ax.scatter(env.trial['stim'][0],env.trial['stim'][1],color='orange')


# Add shades around lines
#import matplotlib.patches as patches
#hsp = .15
#x = [-.5-hsp,.5-hsp,.5+hsp,-.5+hsp]
#y1 = [-.5,.5,.5,-.5]
#y2 = [.5,-.5,-.5,.5]
#ax.add_patch(patches.Polygon(xy=list(zip(x,y1)),fill=True,color='blue',alpha=.2,linewidth=0))
#ax.add_patch(patches.Polygon(xy=list(zip(x,y2)),fill=True,color='orange',alpha=.2,linewidth=0))
 
# plt.savefig('classification_lines_6.png',bbox_inches='tight',format='png',dpi=300)
# plt.savefig('classification_lines_6.eps',bbox_inches='tight',format='eps',dpi=300)

'''
# DDM plot
fig, ax = plt.subplots(figsize=(3,2))
ax.plot(tenvs[0].gt)
ax.set_xlim([0,20])
ax.set_ylim([-5.2,5.2])
ax.set_xlabel('Time')
ax.set_ylabel('$A_a$')
ax.set_title('Accumulators')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.55))
ax.spines['bottom'].set_position(('data', -5.55))
ax.xaxis.set_major_locator(MultipleLocator(5))
ax.yaxis.set_major_locator(MultipleLocator(5))

#plt.savefig('accumulators.png',bbox_inches='tight',format='png',dpi=300)
#plt.savefig('accumulators.eps',bbox_inches='tight',format='eps',dpi=300)
'''


# Time legend
gradient = np.linspace(0, 1, 256)
gradient = np.vstack((gradient, gradient))

fig, ax = plt.subplots(figsize=(1, .4))
ax.imshow(gradient, aspect='auto', cmap='Blues')
ax.set_axis_off()
ax.set_title('Time')
#plt.savefig('time_bar.png',bbox_inches='tight',format='png',dpi=300)
plt.show()

# Quadrant-color mapping
fig, ax = plt.subplots(figsize=(1,1))

pos = .25
width = pos*2
locs = [[(-pos,-pos),(pos,-pos)],[(-pos,pos),(pos,pos)]]
for k in range(2):
    for l in range(2):
        # Create a square with side length 1 and the specified color
        (x, y) = locs[k][l]
        square = plt.Rectangle((x-width/2,y-width/2), width, width,
                               facecolor=colors[k][l], edgecolor=None)
    
        # Add the square to the plot
        ax.add_patch(square)
        
ax.set_xlim([-.5,.5])
ax.set_ylim([-.5,.5])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.55))
ax.spines['bottom'].set_position(('data', -.55))
ax.xaxis.set_major_locator(MultipleLocator(.5))
ax.yaxis.set_major_locator(MultipleLocator(.5))
ax.set_xticklabels([])
ax.set_yticklabels([])
#plt.savefig('quad_colors.png',bbox_inches='tight',format='png',dpi=300)
#plt.savefig('quad_colors.eps',bbox_inches='tight',format='eps',dpi=300)
plt.show()


# Steady-state firing rates raster plot

ss_fr = np.zeros((n_sweep,n_sweep,n_neu))
x1s = np.linspace(-.5,.5,n_sweep)
x2s = np.linspace(-.5,.5,n_sweep)

for i, x1 in enumerate(x1s):
    for j, x2 in enumerate(x2s):
        
        # Construct inputs
        inp = np.tile([1, x1, x2],(1, int(timing['stimulus']/100), 1)) if n_in == 3 else np.tile([x1, x2],(1, int(timing['stimulus']/100), 1))
        inp = torch.tensor(inp, dtype=torch.float32)
        
        if encode:
            inp = util.encode(encoder,inp,n_dim,n_in)
        _, rnn_activity = net(inp)
        ss_fr[i,j] = rnn_activity[0, -1, :].detach().numpy()
        
# Correlations
x1corr = util.corr2_coeff(x1s[np.newaxis,:],np.mean(ss_fr,axis=1).T)
x2corr = util.corr2_coeff(x2s[np.newaxis,:],np.mean(ss_fr,axis=0).T)

# Reshape        
ss_fr = np.reshape(ss_fr,(n_sweep,n_sweep,int(np.sqrt(n_neu)),int(np.sqrt(n_neu))))
x1corr = np.reshape(x1corr,(int(np.sqrt(n_neu)),int(np.sqrt(n_neu))))
x2corr = np.reshape(x2corr,(int(np.sqrt(n_neu)),int(np.sqrt(n_neu))))

# Plot activity
fig, axes = plt.subplots(8, 8, figsize=(8, 8))

for i in range(n_sweep):
    for j in range(n_sweep):
        ax = axes[-j-1,i]
        im = ax.imshow(ss_fr[i,j], cmap='Greys')
        ax.set_xticks([])
        ax.set_yticks([])

# Create a common colorbar
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [x, y, width, height]
cbar = plt.colorbar(im, cax=cax)
cbar.set_label('Firing rate (spikes/s)')

# Common axis labels
fig.text(0.5, 0.08, '$x_1$', ha='center')
fig.text(0.08, 0.5, '$x_2$', va='center', rotation='vertical')

#plt.savefig('activ.png',bbox_inches='tight',format='png',dpi=300)
#plt.savefig('activ.eps',bbox_inches='tight',format='eps',dpi=300)
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

#plt.savefig('corr.png',bbox_inches='tight',format='png',dpi=300)
#plt.savefig('corr.eps',bbox_inches='tight',format='eps',dpi=300)

plt.show()

# TODO: add functionality that visualizes encoder transformation