"""
Evaluate zero-shot, out-of-distribution generalization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from pathlib import Path
from RNN import RNN
import tasks
import neurogym as ngym
from random import randint
import util
import matplotlib.pyplot as plt
import random
import time
from transformers import GPT2Config
from transformer import GPT2ContinuousInputs

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Parameters
n_neu = 64          # number of recurrent neurons
dt = 100            # step size
tau = 100           # neuronal time constant (synaptic+membrane)
n_sd_in = 2         # standard deviation of input noise
n_sd_net = 0        # standard deviation of network noise
n_dim = 2           # dimensionality of state space
n_in = n_dim + 1    # number of inputs
n_ff = n_neu        # number of neurons in feedforward neural net
n_out = n_dim       # number of outputs
batch_sz = 16       # batch size
n_test = 40         # number of test batches
trial_sz = 1        # draw multiple trials in a row
n_fit = 5           # number of fits for each quadrant
n_runs = 5          # number of trained networks for each number of tasks
# Transformer parameters
n_layer = 1         # number of layers
n_head = 8          # number of heads
out_of_distribution = True
keep_test_loss_hist = False
save = False
save_decoders = False
split = None
split_var = False
activation = 'relu'
filename = 'LinCentOutTanhSL64gpt-2batch2e4LR0.001Noise2NetN0nTrial1nLayer1nHead8nTask'
leaky = False if 'NoLeak' in filename else True
encode = True
noise_enc = False
if encode:
    n_feat = 40 + (1 if n_in>n_dim else 0)
    if noise_enc:
        n_sd_enc = n_sd_in/20
        n_sd_in = 0
else:
    n_feat = n_in

# Reproducibility
seed = 42  # 3

def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(mode=True)
    for env in tenvs_test: env.reset(seed=seed)
    for env in tenvs_train: env.reset(seed=seed)

n_tasks = np.array([24])
n_batch = np.array([1e3])

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

# Compute number of iterations through quadrants and update number of fits
n_quads = 2**n_dim
if split:
    n_repeat = n_quads/split
else:
    n_repeat = 1
n_fits = np.ceil(n_fit/n_repeat).astype(int)
if n_fits == 1 and split:
    # Force stop in certain quadrant 
    q_stop = n_fit*split
else:
    q_stop = n_quads
 
# Split quadrants based on variables
if split_var:
    n_split = n_dim
    n_fits = n_fit
    q_stop = n_dim
else:
    n_split = q_stop

# Store losses
train_loss_hist = np.zeros((np.size(n_tasks),n_runs,n_split,n_fits,100))
r_sq = np.zeros((np.size(n_tasks),n_runs,n_split,n_fits))
if keep_test_loss_hist:
    test_loss_hist = np.zeros((np.size(n_tasks),n_runs,n_split,n_fits,100))
else:
    test_loss_hist = np.zeros((np.size(n_tasks),n_runs,n_split,n_fits))
    
# Store decoders and angles
dec = np.zeros((np.size(n_tasks),n_runs,n_split,n_fits,n_neu,n_dim))
angles = np.zeros((np.size(n_tasks),n_runs,n_split,n_fits,n_dim,n_dim))

# Device

device = torch.device('cpu')

# Baseline for r^2
x = np.random.rand(100000) - .5
var = np.var(x)

# Begin!

start_time = time.time()

with device:

    for n, n_task in enumerate(n_tasks):
        
        print_every = int(n_batch[n]/100)
        
        for run in np.arange(n_runs):
        
            print('Network {} out of {} trained on {} tasks'.format(run+1,n_runs,n_task))
            
            # Load network
            data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
            net_file = filename + str(n_task) + ('Mix' if encode else '') + (('run' + str(run)) if run != 0 else '')
            
            if 'LSTM' in net_file:
                net = nn.LSTM(n_feat,n_neu,batch_first=True).to(device)
            elif 'gpt-2' in net_file:
                config = GPT2Config(
                    vocab_size=1,
                    n_embd=n_neu,
                    n_layer=n_layer,
                    n_head=n_head,
                    n_positions=t_task,
                    n_ctx=t_task,
                    n_in=n_feat,
                )
                net = GPT2ContinuousInputs(config).to(device)
            else:
                net = RNN(n_feat,n_neu,n_task,n_sd_net,activation,tau,dt,leaky)
            checkpoint = torch.load(os.path.join(data_path,net_file + '.pth'))
            net.load_state_dict(checkpoint['state_dict'])
            
            if encode:
                # Initialize encoder
                encoder = nn.Sequential(
                        nn.Linear(n_dim,100),
                        nn.ReLU(),
                        nn.Linear(100,100),
                        nn.ReLU(),
                        nn.Linear(100,40)
                        ).to(device)
                
                encoder.load_state_dict(checkpoint['encoder'])
                
                # Freeze encoder weights
                for param in encoder.parameters():
                    param.requires_grad = False
            
            # Initialize decoder
            ff_net = nn.Sequential(
                    nn.Linear(n_neu,n_out)
                    #nn.Sigmoid(),
                    #nn.Linear(n_ff,n_out)
                    )
            
            # Train decoder only
            net.eval()
            for param in net.parameters():
                param.requires_grad = False
            
            quads = np.arange(2**n_dim)+1
            
            # Choose a quadrant for test
            for q, quad_test in enumerate(quads):
                
                # Break early when multiple runs include same quadrants
                if q == q_stop:
                    break
                
                if out_of_distribution:
                    
                    if split is not None:
                        quad_test = (np.arange(0,len(quads),split) + quad_test - 1) % len(quads) + 1
                    
                    quad_train = np.setdiff1d(quads,quad_test)
                
                else:
                    quad_train = quads
                   
                # Split quads according to sign of a held-out variable
                if split_var:
                    split_pos = (quads - 1) // 2**q % 2
                    quad_test = quads[split_pos.astype(bool)]
                    quad_train = np.setdiff1d(quads,quad_test)
                    
                # Environments
                tenvs_train = [value(timing=timing,sigma=n_sd_in,n_task=n_out,
                                     n_dim=n_dim,quad_num=quad_train) for key, value in task.items()]
                tenvs_test = [value(timing=timing,sigma=n_sd_in,n_task=n_out,
                                    n_dim=n_dim,quad_num=quad_test) for key, value in task.items()]
                
                # Seed
                seed_everything(seed)
                
                # Datasets
                datasets_train = [ngym.Dataset(tenv,batch_size=batch_sz,
                                 seq_len=trial_sz*t_task) for tenv in tenvs_train]
                datasets_test = [ngym.Dataset(tenv,batch_size=batch_sz,
                                 seq_len=trial_sz*t_task) for tenv in tenvs_test]
                
                # Perform several fits for statistics
                for fit in range(n_fits):
                    
                    errors = []
                    
                    print("Fit {} of {} for quadrant {}".format(fit+1,n_fits,quad_test))
                    
                    # Reset parameters of decoder for each run
                    for layer in ff_net.children():
                       if hasattr(layer, 'reset_parameters'):
                           layer.reset_parameters()
                                
                    # Optimizer
                    opt = optim.Adam(ff_net.parameters(), lr=0.003)
                    
                    # Train decoder
                    ff_net.train()
                    train_loss = 0; t = 0
                    
                    for i in range(int(n_batch[n])):
                        # Randomly pick task
                        dataset = datasets_train[randint(0,task_num-1)]
                        # Generate data for current batch
                        inputs, target = dataset()
                        
                        # Reshape so that batch is first dimension
                        inputs = np.transpose(inputs,(1,0,2))[:,:,-n_in:]
                        target = np.transpose(target,(1,0,2))
                        
                        # Turn into tensors
                        inputs = torch.from_numpy(inputs).type(torch.float)
                        target = torch.from_numpy(target).type(torch.float)
                        
                        # Empty gradient buffers
                        opt.zero_grad()
                        
                        # Forward pass
                        if encode:
                            inputs = util.encode(encoder,inputs,n_dim,n_in)
                            if noise_enc:
                                inputs += n_sd_enc * torch.randn_like(inputs)
                        
                        if 'gpt-2' in net_file:
                            transformer_outputs = net(
                                inputs_embeds=inputs,
                                return_dict=True,
                            )
                        
                            # Get hidden states
                            hidden_states = transformer_outputs.last_hidden_state
                        
                            # Pass through decoder
                            output = ff_net(hidden_states)
                        
                        else:
                            fr, _ = net(inputs)
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
                            print('{} % of the simulation complete'.format(round(i/n_batch[n]*100)))
                            print('Train loss {:0.3f}'.format(train_loss))
                            train_loss_hist[n,run,q,fit,t] = train_loss
                            
                            # Keep track of test loss history
                            if keep_test_loss_hist or round(i/n_batch[n]*100) == 99:
                                
                                # Make sure weights are frozen
                                ff_net.eval()
                                
                                with torch.no_grad():
                                    
                                    test_loss = 0
                                    
                                    for _ in range(n_test):
                                        dataset = datasets_test[randint(0,task_num-1)]
                                        # Generate data for current batch
                                        inputs, target = dataset()
                                        
                                        # Reshape so that batch is first dimension
                                        inputs = np.transpose(inputs,(1,0,2))[:,:,-n_in:]
                                        target = np.transpose(target,(1,0,2))
                                        
                                        # Turn into tensors
                                        inputs = torch.from_numpy(inputs).type(torch.float)
                                        target = torch.from_numpy(target).type(torch.float)
                                        
                                        # Forward pass
                                        if encode:
                                            inputs = util.encode(encoder,inputs,n_dim,n_in)
                                            if noise_enc:
                                                inputs += n_sd_enc * torch.randn_like(inputs)
                                        
                                        if 'gpt-2' in net_file:
                                            transformer_outputs = net(
                                                inputs_embeds=inputs,
                                                return_dict=True,
                                            )
                                        
                                            # Get hidden states
                                            hidden_states = transformer_outputs.last_hidden_state
                                        
                                            # Pass through decoder
                                            output = ff_net(hidden_states)
                                        
                                        else:
                                            fr, _ = net(inputs)
                                            output = ff_net(fr)
                                        
                                        # Compute loss
                                        loss, _ = util.MSELoss_weighted(output[:,outputs,:], target[:,outputs,:], 1)
                                        test_loss += loss.item()
                                        
                                        # Keep final errors
                                        if round(i/n_batch[n]*100) == 99:
                                            a = output.detach().numpy()
                                            b = target.detach().numpy()
                                            c = b - a

                                            errors.append(np.reshape(c[:,outputs,:],(-1,n_out)))
                                        
                                test_loss /= n_test
                                print('Test loss {:0.3f}'.format(test_loss))
                                        
                                # Store loss history
                                if keep_test_loss_hist:
                                    test_loss_hist[n,run,q,fit,t] = test_loss
                                else:
                                    test_loss_hist[n,run,q,fit] = test_loss
                                    
                                test_loss = 0
                                # Put the network in train mode again
                                ff_net.train()
                                
                            train_loss = 0; t += 1
                        
                    errors = np.reshape(np.asarray(errors),(-1,n_out))
                    mse = np.sum(errors**2)/np.size(errors)
                    r_sq[n,run,q,fit] = 1 - mse/var
                    
                    dec_weight = ff_net[0].weight.detach().numpy().T
                    dec[n,run,q,fit,:] = dec_weight
                    angles[n,run,q,fit,:] = util.angles_between_vectors(dec_weight)
                    
                    #plt.hist(errors,100)
                    #plt.show()

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

# r-squared plot
perc = np.percentile(r_sq,[25,50,75],axis=(1,2,3))
offset = 0.1; off_p = 1+offset; off_m = 1-offset 

fig, ax = plt.subplots(figsize=(2.5,2))
ax.scatter(n_tasks*off_p,perc[1])
ax.errorbar(n_tasks*off_p,perc[1],yerr=[perc[1]-perc[0],perc[2]-perc[1]],linestyle='')
#ax.scatter(n_tasks*off_m,perc_free[1],color='firebrick',label='Free')
#ax.errorbar(n_tasks*off_m,perc_free[1],yerr=[perc_free[1]-perc_free[0],
#                    perc_free[2]-perc_free[1]],linestyle='',color='firebrick')
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
plt.legend(frameon=False,ncol=1,bbox_to_anchor=(1,.4),title='RT')
#plt.savefig('r_squared.png',bbox_inches='tight',format='png',dpi=300)
#plt.savefig('r_squared.eps',bbox_inches='tight',format='eps',dpi=300)
plt.show()

# angle between latents
perc_ang = np.nanpercentile(angles,[25,50,75],axis=(1,2,3,4,5))

fig, ax = plt.subplots(figsize=(2.5,2))
ax.scatter(n_tasks,perc_ang[1])
ax.errorbar(n_tasks,perc_ang[1],yerr=[perc_ang[1]-perc_ang[0],perc_ang[2]-perc_ang[1]],linestyle='')
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
#plt.savefig('ang_latents.png',bbox_inches='tight',format='png',dpi=300)
#plt.savefig('ang_latents.eps',bbox_inches='tight',format='eps',dpi=300)
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

for n, n_task in enumerate(n_tasks):
    
    t = np.linspace(0,n_batch[n],100)
    
    fig, ax = plt.subplots(figsize=(2,2))
    ax.plot(t,np.average(train_loss_hist[n],axis=1).T.reshape(len(t),-1),color='blue',alpha = .3)
    ax.plot(t,np.average(train_loss_hist[n],axis=(0,1,2)),color='blue',label='Train')
    if keep_test_loss_hist:
        ax.plot(t,np.average(test_loss_hist[n],axis=1).T.reshape(len(t),-1),color='red',alpha = .3)
        ax.plot(t,np.average(test_loss_hist[n],axis=(0,1,2)),color='red',label='Test')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Batches')
    ax.set_title('Trained on {} tasks'.format(n_task))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.ylim([0,.1])
    plt.legend(prop={'size': SMALL_SIZE},frameon=False,ncol=1)
    #plt.savefig('loss.png',bbox_inches='tight',format='png',dpi=300)
    plt.show()

end_time = time.time()
elapsed_time = end_time - start_time
hours, minutes, seconds = util.convert_seconds(elapsed_time)

# Save
if save:
    np.savez('r_squared.npz',r_sq=r_sq,train=train_loss_hist,test=test_loss_hist,n_tasks=n_tasks)
    
if save_decoders:
    np.savez('decoders.npz',dec=dec,angles=angles,train=train_loss_hist,test=test_loss_hist,n_tasks=n_tasks)


print(f"Elapsed time: {hours} hours, {minutes} minutes, and {seconds} seconds.")