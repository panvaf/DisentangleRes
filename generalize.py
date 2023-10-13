"""
Check if network generalizes to other tasks.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from pathlib import Path
from RNN import RNN
import tasks
import neurogym as ngym
from random import randint
import util
import matplotlib.pyplot as plt

# Parameters
n_neu = 64          # number of recurrent neurons
dt = 100            # step size
tau = 100           # neuronal time constant (synaptic+membrane)
n_sd = 2            # standard deviation of injected noise
n_in = 3            # number of inputs
n_ff = 64           # number of neurons in feedforward neural net
n_out = 2           # number of outputs
batch_sz = 16       # batch size
n_batch = 4e3       # number of batches for training
n_test = 10        # number of test batches
trial_sz = 4        # draw multiple trials in a row
n_runs = 1         # number of runs for each quadrant
print_every = int(n_batch/100)
out_of_sample = True
keep_test_loss_hist = True

n_tasks = np.array([48])
r_sq = np.zeros(np.size(n_tasks))

# Tasks
task = {'DenoiseQuads':tasks.DenoiseQuads}
#task_rules = util.assign_task_rules(task)
task_num = len(task)

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}

t_task = int(sum(timing.values())/dt)
outputs = np.arange(t_task-1,trial_sz*t_task,t_task)

# Store losses
train_loss_hist = np.zeros((np.size(n_tasks),4,n_runs,100))
if keep_test_loss_hist:
    test_loss_hist = np.zeros((np.size(n_tasks),4,n_runs,100))
else:
    test_loss_hist = np.zeros((np.size(n_tasks),4,n_runs))
    
# Begin!

for n, n_task in enumerate(n_tasks):
    
    print('Network trained on {} tasks'.format(n_task))
    
    # Load network
    data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
    net_file = 'LinCent64batch1e5Noise2nTask' + str(n_task)
    
    net = RNN(n_in,n_neu,n_task,n_sd,tau,dt)
    checkpoint = torch.load(os.path.join(data_path,net_file + '.pth'))
    net.load_state_dict(checkpoint['state_dict'])
    
    # Feedforward neural network that learns multiplication
    
    ff_net = nn.Sequential(
            nn.Linear(n_ff,n_out)
            #nn.Sigmoid(),
            #nn.Linear(n_ff,n_out)
            )
    
    # Train feedforward neural net only
    net.eval()
    for param in net.parameters():
        param.requires_grad = False
    
    errors = []
    quads = np.array([1,2,3,4])
    
    # Choose a quadrant for test
    for q, quad_test in enumerate(quads):
        
        # All the other quadrants for train
        quad_train = np.setdiff1d(quads,quad_test)
        
        for run in range(n_runs):
            
            if out_of_sample:
                print("Run {} of {} for quadrant {}".format(run+1,n_runs,quad_test))
            else:
                print("Run {} of {}".format(4*q+run+1,n_runs*4))
                quad_train = quads
                quad_test = quads
            
            # Datasets
            tenvs_train = [value(timing=timing,sigma=n_sd,n_task=n_out,quad_num=quad_train) for key, value in task.items()]
            datasets_train = [ngym.Dataset(tenv,batch_size=batch_sz,seq_len=trial_sz*t_task) for tenv in tenvs_train]
            
            tenvs_test = [value(timing=timing,sigma=n_sd,n_task=n_out,quad_num=quad_test) for key, value in task.items()]
            datasets_test = [ngym.Dataset(tenv,batch_size=batch_sz,seq_len=trial_sz*t_task) for tenv in tenvs_test]
            
            # Reset parameters of decoder for each run
            for layer in ff_net.children():
               if hasattr(layer, 'reset_parameters'):
                   layer.reset_parameters()
            
                        
            # Optimizer
            opt = optim.Adam(ff_net.parameters(), lr=0.003)
            
            # Train decoder
            ff_net.train()
            train_loss = 0; t = 0
            
            for i in range(int(n_batch)):
                # Randomly pick task
                dataset = datasets_train[randint(0,task_num-1)]
                # Generate data for current batch
                inputs, target = dataset()
                
                # Reshape so that batch is first dimension
                inputs = np.transpose(inputs,(1,0,2))[:,:,0:n_in]
                target = np.transpose(target,(1,0,2))
                
                # Turn into tensors
                inputs = torch.from_numpy(inputs).type(torch.float)
                target = torch.from_numpy(target).type(torch.float)
                
                # Empty gradient buffers
                opt.zero_grad()
                
                # Forward run
                net_out, fr = net(inputs)
                output = ff_net(fr)
                
                # Compute loss
                loss, _ = util.MSELoss_weighted(output[:,outputs,:], target[:,outputs,:], 1)
                train_loss += loss.item()
                
                # Backpopagate loss
                loss.backward()
                
                # Update weights
                opt.step()
                
                # Store history of average training loss
                if (i % print_every == 0):
                    train_loss /= print_every
                    print('{} % of the simulation complete'.format(round(i/n_batch*100)))
                    print('Train loss {:0.3f}'.format(train_loss))
                    train_loss_hist[n,q,run,t] = train_loss
                    
                    # Keep track of test loss history
                    if keep_test_loss_hist or round(i/n_batch*100) == 99:
                        
                        # Make sure weights are frozen
                        ff_net.eval()
                        
                        with torch.no_grad():
                            
                            test_loss = 0
                            
                            for _ in range(n_test):
                                dataset = datasets_test[randint(0,task_num-1)]
                                # Generate data for current batch
                                inputs, target = dataset()
                                
                                # Reshape so that batch is first dimension
                                inputs = np.transpose(inputs,(1,0,2))[:,:,0:n_in]
                                target = np.transpose(target,(1,0,2))
                                
                                # Turn into tensors
                                inputs = torch.from_numpy(inputs).type(torch.float)
                                target = torch.from_numpy(target).type(torch.float)
                                
                                # Forward run
                                net_out, fr = net(inputs)
                                output = ff_net(fr)
                                
                                # Compute loss
                                loss, _ = util.MSELoss_weighted(output[:,outputs,:], target[:,outputs,:], 1)
                                test_loss += loss.item()
                                
                        test_loss /= n_test
                        print('Test loss {:0.3f}'.format(test_loss))
                                
                        # Store loss history
                        if keep_test_loss_hist:
                            test_loss_hist[n,q,run,t] = test_loss
                        else:
                            test_loss_hist[n,q,run] = test_loss
                        
                        # Keep final errors
                        if round(i/n_batch*100) == 99:
                            a = output.detach().numpy()
                            b = target.detach().numpy()
                            c = b - a
                            
                            errors.append(np.reshape(c[:,outputs,:],(-1,n_out)))
                            
                        test_loss = 0
                        # Put the network in train mode again
                        ff_net.train()
                        
                    train_loss = 0; t += 1
                
    errors = np.reshape(np.asarray(errors),(-1,n_out))
    err = np.abs(errors) > .5
    
    mse = np.sum(errors**2)/np.size(errors)
    x = np.random.rand(100000) - .5
    var = np.var(x)
    r_sq[n] = 1 - mse/var
    
    plt.hist(errors,100)
    plt.show()

# Plot

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

# Misclassification distance
plt.hist(np.sum(err,axis=1))
plt.xlabel('Misclassification distance')
plt.ylabel('Count')
plt.show()

# r-squared plot
fig, ax = plt.subplots(figsize=(2,2))
ax.scatter(n_tasks,r_sq)
#ax.scatter(n_tasks,r_sq_test,label='Out-of-sample')
ax.set_ylabel('Out-of-sample $R^2$')
ax.set_xlabel('# of tasks')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', 1.6))
ax.spines['bottom'].set_position(('data', .39))
ax.set_xscale("log")
ax.set_xticks([2,10,50])
ax.set_xticklabels([2,10,50])
#plt.xlim([0,22])
plt.ylim([0.4,1.1])
plt.legend(prop={'size': SMALL_SIZE},frameon=False,ncol=1,bbox_to_anchor=(1,1.2))
#plt.savefig('r_squared.png',bbox_inches='tight',format='png',dpi=300)
plt.show()

'''
# Classification lines
x = np.linspace(0,.5,100)
for a in tenvs[0].thres:
    plt.plot(x,np.sqrt(a**2-x**2))
plt.ylim([0,.5])
plt.xlabel('True evidence 1')
plt.ylabel('True evidence 2')
plt.title('Classification lines')
plt.imshow()
'''
env = dataset.env

fig,ax = plt.subplots(figsize=(3,.75))
plt.plot(env.ob[:,1],label='Noisy')
plt.plot(np.ones(22)*env.trial['stim'][0],label='True')
plt.ylabel('Evidence 2')
plt.xlabel('Time')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_position(('data', -1))
ax.spines['bottom'].set_position(('data', -.1))
#plt.legend(prop={'size': SMALL_SIZE},frameon=False,ncol=2,bbox_to_anchor=(.1,1))
plt.xlim([0,22])
plt.ylim([0,.7])
#plt.savefig('evidence2.png',bbox_inches='tight',format='png',dpi=300)
plt.show()


# Plot loss history
t = np.linspace(0,n_batch,100)
fig, ax = plt.subplots(figsize=(2,2))
ax.plot(t,train_loss_hist[0,:,0,:].T,color='blue',alpha = .3)
ax.plot(t,np.average(train_loss_hist[0,:,0,:],axis=0),color='blue',label='Train')
if keep_test_loss_hist:
    ax.plot(t,test_loss_hist[0,:,0,:].T,color='red',alpha = .3)
    ax.plot(t,np.average(test_loss_hist[0,:,0,:],axis=0),color='red',label='Test')
ax.set_ylabel('Loss')
ax.set_xlabel('Batches')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
#ax.spines['left'].set_position(('data', 1.6))
#ax.spines['bottom'].set_position(('data', .39))
#plt.xlim([0,22])
plt.ylim([0,.1])
plt.legend(prop={'size': SMALL_SIZE},frameon=False,ncol=1,bbox_to_anchor=(1,1.2))
#plt.savefig('r_squared.png',bbox_inches='tight',format='png',dpi=300)
plt.show()