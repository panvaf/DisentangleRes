"""
Train recurrent neural network.
"""

import neurogym as ngym
import tasks
import util
from RNN import RNN
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from random import randint
import os
from pathlib import Path
import time
from transformers import GPT2Config
from transformer import GPT2ContinuousInputs


os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

# Tasks
task = {"LinearClassificationCentOut":tasks.LinearClassificationCentOut}
task_rules = util.assign_task_rules(task)
task_num = len(task)

# Constants
n_neu = 64          # number of recurrent neurons
batch_sz = 16       # batch size
n_batch = 2e4       # number of batches
dt = 100            # step size
tau = 100           # neuronal time constant (synaptic+membrane)
n_sd_in = 2         # standard deviation of input noise
n_sd_net = 0        # standard deviation of network noise
n_dim = 2           # dimensionality of state space
print_every = int(n_batch/100)
n_out = 24          # number of outputs per task
bal_err = False     # whether to balance penalization of decision vs. integration
pen_end = False     # only penalize final time point
trial_num = 1       # number of trials drawn in a row
rand_pen = False    # randomly penalize a certain time point in the trial
bound = 5           # DDM boundary
encode = True       # Whether to nonlinearly mix the input features
noise_enc = False   # Whether to noise after the encoder
corr = 0            # Correlation between factors
activation = 'relu' # activation function
leaky = True        # whether the RNN is leaky
network = 'gpt-2'     # Network architecture
init = None         # Initialization for RNN hidden layer
lr = 1e-3           # Learning rate
autocorr = 0        # noise autocorrelation
dist = 'gauss'      # noise distribution
# Transformer parameters
n_layer = 1         # number of layers
n_head = 8          # number of heads
run = 0

if encode:
    if noise_enc:
        # Divide by 10 bc n_sd_in is divided by 10 in tasks and by another 2 
        # because this is roughly what the encoder does for this noise level
        n_sd_enc = n_sd_in/20
        n_sd_in = 0

# Environment
timing = {'fixation': 100,
          'stimulus': 2000,
          'delay': 0,
          'decision': 100}
t_task = int(sum(timing.values())/dt)
grace = 200
#thres = np.array([0.005, 0.02, 0.04, 0.07, 0.11, 0.15])
#thres = np.array([0.005, 0.01, 0.018, 0.027, 0.04, 0.052, 0.07, 0.085, 0.105, 0.125, 0.15, 0.18])

n_grace = int(grace/dt); n_decision = int(timing['decision']/dt); n_trial = int(sum(timing.values())/dt)

# Save location
data_path = str(Path(os.getcwd()).parent) + '/trained_networks/'
net_file = 'LinCentOutTanhSL' + str(n_neu) + (('Bound' + str(bound)) if bound != 5 else '') + \
            (network if network != 'RNN' else '') + (activation if activation != 'relu' else '') + \
            (init if init is not None else '') +  ('NoLeak' if leaky == False else '') + \
            (('batch' + format(n_batch,'.0e').replace('+0','')) if not n_batch==1e4 else '') + \
            (('LR' + str(lr)) if lr != 3e-3 else '')  + \
            (('Noise' + str(n_sd_in)) if n_sd_in else '') + \
            (('Autocorr' + str(autocorr)) if autocorr else '') + \
            (dist if dist != 'gauss' else '') + \
            (('NetN' + str(n_sd_net)) if n_sd_net != 2 else '') + \
            (('tau' + str(tau)) if tau != 100 else '') + \
            (('nTrial' + str(trial_num)) if trial_num != 4 else '')  + \
            (('nDim' + str(n_dim)) if n_dim != 2 else '')  + \
            (('Corr' + str(corr)) if corr else '')  + \
            (('nLayer' + str(n_layer)) if network == 'gpt-2' else '')  + \
            (('nHead' + str(n_head)) if network == 'gpt-2' else '')  + \
            (('nTask' + str(n_out)) if n_out != 2 else '')  + \
            (('Delay' + str(timing['delay'])) if timing['delay'] != 0 else '')  + \
            ('BalErr' if bal_err else '') + ('RandPen' if rand_pen else '') + \
            ('PenEnd' if pen_end else '') + ('Mix' if encode else '') + \
            ('nEnc' if noise_enc else '') + (('run' + str(run)) if run != 0 else '')

# Make supervised datasets
tenvs = [value(timing=timing,sigma=n_sd_in,n_task=n_out,n_dim=n_dim,thres=bound,
               corr=corr,ar_coef=autocorr,dist=dist,rule_vec=task_rules[key]) for key, value in task.items()]

datasets = [ngym.Dataset(tenv,batch_size=batch_sz,seq_len=trial_num*t_task) for tenv in tenvs]

# Create environment
env = datasets[0].env

# Network input and output size
n_in = env.observation_space.shape[0]
#n_out = env.action_space.n

# Mask to weight errors during integration and decision equally
if bal_err:
    mask_w = (sum(timing.values()) - grace - timing['decision'])/timing['decision']
    mask = np.ones((batch_sz,n_trial,1)); mask[:,-n_decision-n_grace:-n_decision] = 0
    mask[:,-n_decision:] = mask_w; mask = np.tile(mask,(1,4,n_out))
elif pen_end:
    mask = np.zeros((batch_sz,t_task,n_out))
    mask[:,-1,:] = 1
    mask = np.tile(mask,(1,trial_num,1))
else:
    mask = np.ones((batch_sz,trial_num*t_task,n_out))
    
# Encoder
encoder = nn.Sequential(
        nn.Linear(n_dim,100),
        nn.ReLU(),
        nn.Linear(100,100),
        nn.ReLU(),
        nn.Linear(100,40)
        )

# Initialize RNN
if encode:
    n_in = encoder[-1].out_features + (1 if n_in>n_dim else 0)

# Device
device = torch.device('cpu')
    
if network == 'RNN':
    net = RNN(n_in,n_neu,n_out*task_num,n_sd_net,activation,tau,dt,leaky,init).to(device)
elif network == 'LSTM':
    net = nn.LSTM(n_in,n_neu,batch_first=True).to(device)
elif network == 'gpt-2':
    
    device = torch.device('mps')
    config = GPT2Config(
        vocab_size=1,
        n_embd=n_neu,
        n_layer=n_layer,
        n_head=n_head,
        n_positions=t_task,
        n_ctx=t_task,
        n_in=n_in,
    )
    
    net = GPT2ContinuousInputs(config).to(device)
    
# size of model
print("Number of params:", sum(p.numel() for p in net.parameters()))

encoder.to(device)

# Freeze encoder weights
for param in encoder.parameters():
    param.requires_grad = False

# Decoder
decoder = nn.Sequential(
        nn.Linear(n_neu,n_out*task_num),
        nn.Tanh()
        ).to(device)

# Optimizer
opt = optim.Adam(net.parameters(), lr=lr)
opt.add_param_group({'params': decoder.parameters()})

# Train RNN
total_loss = 0; k = 0
loss_hist = np.zeros(100)
conf_matr = np.zeros((2,2))
conf_matr_hist = np.zeros((100,2,2))

start_time = time.time()

with device:
    
    for i in range(int(n_batch)):
        # Randomly pick task
        task = randint(0,task_num-1)
        dataset = datasets[task]
        # Generate data for current batch
        inputs, target = dataset()
        
        # Reshape so that batch is first dimension
        inputs = np.transpose(inputs,(1,0,2))
        target = np.transpose(target,(1,0,2))
        
        # Construct mask to penalize specific time moment
        if rand_pen:
            mask = np.zeros((batch_sz,t_task,n_out))
            mask[:,np.random.randint(5,t_task),:] = 1
            mask = np.tile(mask,(1,trial_num,1))
        
        # Reshape for multiple tasks
        masker = np.zeros((batch_sz,trial_num*t_task,n_out*task_num))
        masker[:,:,task*n_out:(task+1)*n_out] = mask
        targets = np.zeros((batch_sz,trial_num*t_task,n_out*task_num))
        targets[:,:,task*n_out:(task+1)*n_out] = target
        
        # Turn into tensors
        inputs = torch.from_numpy(inputs).type(torch.float).to(device)
        targets = torch.from_numpy(targets).type(torch.long).to(device)
        masker = torch.from_numpy(masker).type(torch.long).to(device)
        
        # Empty gradient buffers
        opt.zero_grad()
        
        # Forward pass
        if encode:
            inputs = util.encode(encoder,inputs,n_dim,inputs.shape[2])
            if noise_enc:
                inputs += n_sd_enc * torch.randn_like(inputs)
            
        if network=='gpt-2':
            transformer_outputs = net(
                inputs_embeds=inputs,
                return_dict=True,
            )
        
            # Get hidden states
            hidden_states = transformer_outputs.last_hidden_state
        
            # Pass through decoder
            output = decoder(hidden_states)
        
        else:
            fr, _ = net(inputs)
            if 'Bound' in net_file:
                output = bound*decoder(fr)
            else:
                output = decoder(fr)
        
        # Compute loss
        #loss = criterion(output.view(-1,n_out),target.flatten())
        _, loss = util.MSELoss_weighted(output, targets, masker)
        total_loss += loss.item()
        
        # Backpopagate loss
        loss.backward()
        
        # Update weights
        opt.step()
        
        # Confusion matrix
        if 'Bound' not in net_file:
            conf_matr += util.confusion_matrix(output.cpu().detach().numpy(),targets.cpu().detach().numpy())
        
        # Store history of average training loss
        if (i % print_every == 0):
            total_loss /= print_every
            print('{} % of the simulation complete'.format(round(i/n_batch*100)))
            print('Loss {:0.3f}'.format(total_loss))
            if 'Bound' not in net_file:
                print('Accuracy {:0.1f} %'.format(100*np.trace(conf_matr)/np.sum(conf_matr)))
            loss_hist[k] = total_loss
            conf_matr_hist[k] = conf_matr
            total_loss = 0; conf_matr = np.zeros((2, 2)); k += 1

# Save network
if encode:
    torch.save({'state_dict': net.state_dict(),'encoder': encoder.state_dict(),'loss_hist': loss_hist,
                'conf_matr_hist': conf_matr_hist}, data_path + net_file + '.pth', _use_new_zipfile_serialization=False)
else:
    torch.save({'state_dict': net.state_dict(),'loss_hist': loss_hist,'conf_matr_hist': conf_matr_hist},
              data_path + net_file + '.pth', _use_new_zipfile_serialization=False)

end_time = time.time()
elapsed_time = end_time - start_time
hours, minutes, seconds = util.convert_seconds(elapsed_time)

print(f"Elapsed time: {hours} hours, {minutes} minutes, and {seconds} seconds.")