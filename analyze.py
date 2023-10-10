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
from matplotlib.ticker import MultipleLocator
import plotly.graph_objects as go
from plotly.colors import label_rgb
import matplotlib.colors as mcol
import plotly.io as pio
pio.renderers.default = "browser"

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
n_task = 48         # number of tasks
n_trial = 40        # number of bulk example trials to plot
n_exam = 10         # number of example points to plot with separate colors
thres = 5           # DDM boundary

# Tasks
task = {"LinearClassification":tasks.LinearClassification}
#task_rules = util.assign_task_rules(task)
task_num = len(task)

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}

t_task = int(sum(timing.values())/dt)

#thres = np.array([0.005, 0.01, 0.018, 0.027, 0.04, 0.052, 0.07, 0.085, 0.105, 0.125, 0.15, 0.18])
tenvs = [value(timing=timing,sigma=n_sd,n_task=n_task,thres=thres) for key, value in task.items()]

# Load network
data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
net_file = 'LinCent64batch1e5Noise2nTask' + str(n_task)

net = RNN(n_in,n_neu,n_task,n_sd,tau,dt)
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
    hidden = torch.tensor(np.random.rand(batch_size, n_neu)*20+hdn,
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
tenvs = [value(timing=timing,sigma=n_sd,n_task=n_task) for key, value in task.items()]

datasets = [ngym.Dataset(tenv,batch_size=1,seq_len=t_task) for tenv in tenvs]

stims = np.zeros((n_exam,2))
ex_activ = np.zeros((n_exam,t_task,n_neu))

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
    ex_activ[i] = rnn_activity


# Plot network activity and overlay approximate fixed points
colors = [['gold','limegreen'],['dodgerblue','lightcoral']]

plot_full = util.rot_3D_plot(activity_dict,fixedpoints,pca,n_trial,trial_info,
                        net_file,colors=colors)
plot_full.plot()


# Save the animation as an html file
#pio.write_html(fig, '3d_scatterplot_animation.html')


'''
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0,360.5,.5))
f = r'D:\\Decoupling\\Figures\\rotationLinBound.mp4'
writer = animation.FFMpegWriter(fps=60) 
rot_animation.save(f, dpi=300, writer=writer)
'''

# Plot example trials
fig, ax = plt.subplots(figsize=(6, 6))
ax = plt.axes(projection='3d')
for i in range(n_exam):
    ex_activ_pc = pca.transform(ex_activ[i])
    ax.plot3D(ex_activ_pc[:, 0], ex_activ_pc[:, 1], ex_activ_pc[:, 2], 'x-', ms=10)
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

# Examples in state space
#for i in range(n_exam):
#    ax.scatter(stims[i,0],stims[i,1],s=50,c=colors[i])
#plt.show()

# ax.scatter(env.trial['stim'][0],env.trial['stim'][1],color='orange')

# Add shades around lines
# import matplotlib.patches as patches
# hsp = .15
# x = [-.5-hsp,.5-hsp,.5+hsp,-.5+hsp]
# y1 = [-.5,.5,.5,-.5]
# y2 = [.5,-.5,-.5,.5]
# ax.add_patch(patches.Polygon(xy=list(zip(x,y1)),fill=True,color='blue',alpha=.2,linewidth=0))
# ax.add_patch(patches.Polygon(xy=list(zip(x,y2)),fill=True,color='orange',alpha=.2,linewidth=0))
# 
# plt.savefig('classification_lines_6.png',bbox_inches='tight',format='png',dpi=300)
# plt.savefig('classification_lines_6.eps',bbox_inches='tight',format='eps',dpi=300)

# DDM plot
#fig, ax = plt.subplots(figsize=(3,2))
#ax.plot(tenvs[0].gt)
#ax.set_xlim([0,20])
#ax.set_ylim([-5.2,5.2])
#ax.set_xlabel('Time')
#ax.set_ylabel('$A_a$')
#ax.set_title('Accumulators')
#ax.spines['top'].set_visible(False)
#ax.spines['right'].set_visible(False)
#ax.spines['left'].set_position(('data', -.55))
#ax.spines['bottom'].set_position(('data', -5.55))
#ax.xaxis.set_major_locator(MultipleLocator(5))
#ax.yaxis.set_major_locator(MultipleLocator(5))

#plt.savefig('accumulators.png',bbox_inches='tight',format='png',dpi=300)
#plt.savefig('accumulators.eps',bbox_inches='tight',format='eps',dpi=300)


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

'''
# Shape legend
nx1, nx2 = (6, 6)
x1 = np.linspace(-.5, .5, nx1)
x2 = np.linspace(-.5, .5, nx2)
x1v, x2v = np.meshgrid(x1, x2)

fig, ax = plt.subplots(figsize=(2,2))
ax.scatter(x1v[0:int(nx1/2),0:int(nx1/2)],x2v[0:int(nx1/2),0:int(nx1/2)],marker='D',c='black',alpha=1)
ax.scatter(x1v[0:int(nx1/2),int(nx1/2):],x2v[0:int(nx1/2),int(nx1/2):],marker='D',c='black',alpha=.4)
ax.scatter(x1v[int(nx1/2):,0:int(nx1/2)],x2v[int(nx1/2):,0:int(nx1/2)],marker='o',c='black',alpha=1)
ax.scatter(x1v[int(nx1/2):,int(nx1/2):],x2v[int(nx1/2):,int(nx1/2):],marker='o',c='black',alpha=.4)
ax.set_xlim([-.55,.55])
ax.set_ylim([-.55,.55])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -.6))
ax.spines['bottom'].set_position(('data', -.6))
ax.xaxis.set_major_locator(MultipleLocator(.5))
ax.yaxis.set_major_locator(MultipleLocator(.5))
#plt.savefig('shape_dim_legend.png',bbox_inches='tight',format='png',dpi=300)
fig.show()
'''