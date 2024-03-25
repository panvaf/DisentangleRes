"""
Vanilla RNN class.
"""

import torch
import torch.nn as nn

# RNN class

class RNN(nn.Module):
    
    def __init__(self,inp_size,rec_size,out_size,n_sd=.1,activation='relu',
                 tau=100,dt=10,leaky=True):
        super().__init__()
        
        # Constants
        self.inp_size = inp_size
        self.rec_size = rec_size
        self.n_sd = n_sd
        self.tau = tau
        self.alpha = dt / self.tau
        self.leaky = leaky
        
        # Layers
        self.inp_to_rec = nn.Linear(inp_size, rec_size)
        self.rec_to_rec = nn.Linear(rec_size, rec_size)
        self.rec_to_out = nn.Linear(rec_size, out_size)
        
        # Activation function
        if activation == 'relu':
            self.activation = torch.relu
        elif activation == 'tanh':
            self.activation = torch.tanh        


    def init(self,inp_shape):
        # Initializes network activity to zero
        
        n_batch = inp_shape[0]
        r = torch.zeros(n_batch,self.rec_size)
        
        return r


    def dynamics(self,inp,r):
        # Defines dynamics of the network
        
        h = self.inp_to_rec(inp) + self.rec_to_rec(r) + \
                    self.n_sd*torch.randn(self.rec_size)
        if self.leaky:  
            r_new = (1 - self.alpha)*r + self.alpha*self.activation(h)
        else:
            r_new = r + self.alpha*self.activation(h)
        
        return r_new


    def forward(self,inp):
        # Forward pass
        
        # Initialize network
        r = self.init(inp.shape)
        out = []; fr = []
        
        # Simulate
        for i in range(inp.shape[1]):
            r = self.dynamics(inp[:,i],r)
            # Store network output and activity for entire batch
            fr.append(r)
            out.append(self.rec_to_out(r))
            
        fr = torch.stack(fr, dim=1)
        out = torch.stack(out, dim=1)
        
        return out, fr
    
    
    def reset_params(self):
        # Reset everything in the network
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()