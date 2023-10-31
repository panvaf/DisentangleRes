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
n_sd = 2            # standard deviation of injected noise
n_in = 3            # number of inputs
n_task = 48         # number of tasks
n_trial = 40        # number of bulk example trials to plot
n_exam = 5         # number of example points to plot with separate colors
thres = 5           # DDM boundary
activation = 'relu'

# Tasks
task = {"LinearClassificationCentOut":tasks.LinearClassificationCentOut}
#task_rules = util.assign_task_rules(task)
task_num = len(task)

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}

t_task = int(sum(timing.values())/dt)

#thres = np.array([0.005, 0.01, 0.018, 0.027, 0.04, 0.052, 0.07, 0.085, 0.105, 0.125, 0.15, 0.18])
tenvs = [value(timing=timing,sigma=0,n_task=n_task,thres=thres) for key, value in task.items()]

# Load network
data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
net_file = 'LinCentOutTanhSL64batch1e5Noise2nTrial1nTask' + str(n_task) + 'run1'

net = RNN(n_in,n_neu,n_task,0,activation,tau,dt)
checkpoint = torch.load(os.path.join(data_path,net_file + '.pth'))
net.load_state_dict(checkpoint['state_dict'])
net.eval()

# Visualize RNN activity
activity_dict = {}; output_dict = {}; trial_info = {}

for i in range(n_trial):
    
    # Pick environment and generate a trial
    tenv = tenvs[randint(0,task_num-1)]
    tenv.new_trial(); ob = tenv.ob
    inp = torch.from_numpy(ob[np.newaxis, :, :]).type(torch.float)[:,:,-n_in:]
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

batch_size = 128
fixedpoints = np.empty([batch_size,1,n_neu])

for j in range(1):
    print('Task {} out of {}'.format(j+1,task_num))

    # Inputs are zero, so that internal representation is not affected
    inp = np.tile([1, 0, 0],(batch_size, 1)) if n_in == 3 else np.tile([0, 0],(batch_size, 1))
    inp = torch.tensor(inp, dtype=torch.float32)
    
    # Initialize hidden activity                                                                    
    hdn = np.zeros((batch_size, n_neu))
    idx_tr = np.random.choice(n_trial,batch_size,replace=True)
    idx_t = np.random.choice(ob.shape[0],batch_size)
    for i in range(batch_size):
        hdn[i] = torch.from_numpy(activity_dict[idx_tr[i]][idx_t[i]])
    hidden = torch.tensor(np.random.rand(batch_size, n_neu)*15+hdn,
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
tenvs = [value(timing=timing,sigma=n_sd,n_task=n_task, thres=thres) for key, value in task.items()]

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
width = pos/2
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
plt.show()