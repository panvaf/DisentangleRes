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
from matplotlib import animation
import matplotlib as mpl 

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

# Directory for ffmpeg

mpl.rcParams['animation.ffmpeg_path'] = r'D:\Lab Resources\ffmpeg\bin\\ffmpeg.exe'

# Rotating animation

def rotate(angle):
    ax.view_init(azim=angle)

# Parameters
n_neu = 64          # number of recurrent neurons
dt = 100            # step size
tau = 100           # neuronal time constant (synaptic+membrane)
n_sd = 0            # standard deviation of injected noise
n_in = 3            # number of inputs
n_out = 48          # number of outputs
n_trial = 50        # number of bulk example trials to plot
n_exam = 12         # number of example points to plot with separate colors
trial_sz = 22

# Tasks
task = {'LinearClassification':tasks.LinearClassification}
#task_rules = util.assign_task_rules(task)
n_task = len(task)

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}
tenvs = [value(timing=timing,sigma=0,n_task=n_out) for key, value in task.items()]

# Load network
data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
net_file = 'LinCent64batch1e5Noise2nTask48'

net = RNN(n_in,n_neu,n_out,n_sd,tau,dt)
checkpoint = torch.load(os.path.join(data_path,net_file + '.pth'))
net.load_state_dict(checkpoint['state_dict'])
net.eval()

# Visualize RNN activity
activity_dict = {}; output_dict = {}; trial_info = {}

for i in range(n_trial):
    
    # Pick environment and generate a trial
    tenv = tenvs[randint(0,n_task-1)]
    tenv.new_trial()
    ob, gt = tenv.ob, tenv.gt
    inp = torch.from_numpy(ob[np.newaxis, :, :]).type(torch.float)
    output, rnn_activity = net(inp)
    rnn_activity = rnn_activity[0, :, :].detach().numpy()
    output = output[0, :, :].detach().numpy()
    activity_dict[i] = rnn_activity
    output_dict[i] = output
    trial_info[i] = tenv.trial

activity = np.concatenate(list(activity_dict[i] for i in range(n_trial)), axis=0)

# Perform PCA
pca = PCA(n_components=3)
pca.fit(activity)

# Find approximate fixed points, depending on network initial conditions

# Freeze for parameters in the recurrent network
for param in net.parameters():
    param.requires_grad = False

batch_size = 128
fixedpoints = np.empty([batch_size,1,n_neu])

for j in range(1):
    print('Task {} out of {}'.format(j+1,n_task))

    # Inputs are zero, so that internal representation is not affected
    inp = np.tile([1, 0, 0],(batch_size, 1))
    inp = torch.tensor(inp, dtype=torch.float32)
    
    # Initialize hidden activity                                                                    
    hdn = np.zeros((batch_size, n_neu))
    idx_tr = np.random.choice(n_trial,batch_size,replace=True)
    idx_t = np.random.choice(ob.shape[0],batch_size)
    for i in range(batch_size):
        hdn[i] = torch.from_numpy(activity_dict[idx_tr[i]][idx_t[i]])
    hidden = torch.tensor(np.random.rand(batch_size, n_neu)*10+hdn,
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
tenvs = [value(timing=timing,sigma=n_sd,n_task=n_out) for key, value in task.items()]

datasets = [ngym.Dataset(tenv,batch_size=1,seq_len=trial_sz) for tenv in tenvs]

stims = np.zeros((n_exam,n_in-1))
ex_activ = np.zeros((n_exam,trial_sz,n_neu))

for i in range(n_exam):
    
    # Randomly pick task
    dataset = datasets[randint(0,n_task-1)]
    # A sample environment from dataset
    env = dataset.env
    env.new_trial()
    
    ob = env.ob
    stims[i] = env.trial['stim']
    
    inp = torch.from_numpy(ob[np.newaxis, :, :]).type(torch.float)
    _, rnn_activity = net(inp)
    rnn_activity = rnn_activity[0, :, :].detach().numpy()
    ex_activ[i] = rnn_activity
    
    
# Plot network activity and overlay approximate fixed points

fig, ax = plt.subplots(figsize=(6, 6))
ax = plt.axes(projection='3d')
for i in range(n_trial):
    activity_pc = pca.transform(activity_dict[i])
    trial = trial_info[i]
    color = 'red' if trial['ground_truth'][0] == 0 else 'blue'
    alpha = .2 if trial['ground_truth'][24] == 0 else 1
    ax.plot3D(activity_pc[:, 0], activity_pc[:, 1], activity_pc[:, 2], 'o-', 
                      color=color, alpha=alpha)

# Fixed points are shown in cross
cols = ['green','yellow']
for i in range(fixedpoints.shape[1]):
    fixedpoints_pc = pca.transform(fixedpoints[:,i])
    hdn_pc = pca.transform(hdn)
    ax.plot3D(fixedpoints_pc[:, 0], fixedpoints_pc[:, 1], fixedpoints_pc[:, 2],
              'x', color=cols[i],markersize=8,markeredgewidth=3)
    #ax.plot3D(hdn_pc[:, 0], hdn_pc[:, 1], hdn_pc[:, 2], 'x', color='magenta')

ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
plt.show()

rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,360.5,.5))
f = r'D:\\Decoupling\\Figures\\rotation.mp4'
writer = animation.FFMpegWriter(fps=60) 
rot_animation.save(f, dpi=300, writer=writer)

# Plot example trials
fig, ax = plt.subplots(figsize=(6, 6))
ax = plt.axes(projection='3d')
for i in range(n_exam):
    ex_activ_pc = pca.transform(ex_activ[i])
    ax.plot3D(ex_activ_pc[:, 0], ex_activ_pc[:, 1], ex_activ_pc[:, 2], 'x-', ms=10)
plt.show()

# Classification lines
fig, ax = plt.subplots()
x = np.linspace(-.5,.5,100)
for alpha in tenvs[0].alphas:
    ax.plot(x,alpha*x)
ax.set_ylim([-.5,.5])
plt.xlabel('True evidence 1')
plt.ylabel('True evidence 2')
ax.set_title('Classification lines')

# Examples in state space
for i in range(n_exam):
    ax.scatter(stims[i,0],stims[i,1],s=50)
plt.show()