"""
Various utilities.
"""

import numpy as np
import torch
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"

def assign_task_rules(tasks):
    """
    Assigns task rules to tasks.

    Parameters
    ----------
    tasks : tasks to train for, list of strings

    Returns
    -------
    task_rules: dictionary mapping task rule to task

    """
    
    task_rules = {}
    I = np.identity(len(tasks))
    
    for i,task in enumerate(tasks):
        
        task_rules[task] = I[i]
    
    return task_rules


# Weighted mean squared error loss

def MSELoss_weighted(output,target,mask):
    loss = torch.sum(mask*(output - target)**2)
    size = torch.numel(target)
    if isinstance(mask, torch.Tensor):    
        norm = torch.sum(mask)
    else:
        norm = 1
    
    avg_loss = loss/size
    normed_loss = loss/norm
    
    return avg_loss, normed_loss


# Creates rotating 3D plot using plotly

class rot_3D_plot():
    
    def __init__(self,activity,fixedpoints,pca,n_trial,trial_info,net_file,
                 n_in=3,colors=None,lines=True):
        
        super().__init__()
        
        # Variables
        self.activity = activity
        self.fixedpoints = fixedpoints
        self.pca = pca
        self.n_trial = n_trial
        self.trial_info = trial_info
        self.net_file = net_file
        self.n_in = n_in
        
        self.t_task = activity[0].shape[0]
        if n_in<=3:
            self.n_task = trial_info[0]['ground_truth'].shape[-1]
        
        # Constants
        self.x_eye = 0
        self.y_eye = 1.0707
        self.z_eye = 1
        
        # Colors
        if colors is not None:
            self.colors = self.get_colors(colors)
            self.lines = True
        else:
            self.lines = False
            self.individual_examples = False
            
        if lines is None:
            self.lines = False
        
    # Get the colors for the final dot in the scatterplot
    def get_colors(self,colors):
        
        if isinstance(colors[0],list):
            self.individual_examples = False
            
            col = []
            
            for i in range(self.n_trial):
                
                trial = self.trial_info[i]
                
                if 'MultFull' in self.net_file:
                    if trial['ground_truth'][0] > 0:
                        quad_col = colors[1][1] # quadrant 1
                    elif trial['ground_truth'][int(self.n_task/4)] > 0:
                        quad_col = colors[1][0] # quadrant 2
                    elif trial['ground_truth'][int(self.n_task/2)] > 0:
                        quad_col = colors[0][0] # quadrant 3
                    elif trial['ground_truth'][-int(self.n_task/4)] > 0:
                        quad_col = colors[0][1] # quadrant 4
                    else:
                        quad_col = None
                else:
                    if self.n_in==3:
                        k = 1 if trial['ground_truth'][0] > 0 else 0
                        l = 1 if trial['ground_truth'][int(self.n_task/2)] > 0 else 0
                    elif self.n_in==2:
                        k = 1 if trial['ground_truth'][-1][0] > 0 else 0
                        l = 1 if trial['ground_truth'][-1][int(self.n_task/2)] > 0 else 0
                    quad_col = colors[k][l]
                  
                col.append(quad_col)
            
        else:
            col = colors
            self.individual_examples = True
            
        return col
    
    
    # Create plot
    def plot(self):
        
        # Configure plotting parameters
        self.plot_params()
        
        fig = go.Figure()
        

        for i in range(self.n_trial):
            
            if self.pca:
                activity_pc = self.pca.transform(self.activity[i])
            else:
                activity_pc = self.activity[i]
            
            if self.individual_examples:
                line_col = self.colors[i]
            else:
                line_col = 'darkblue'
            
            if self.lines:
                fig.add_traces(go.Scatter3d(x=activity_pc[:, 0],y=activity_pc[:, 1],
                           z=activity_pc[:, 2],marker=dict(size=3,color=np.arange(self.t_task),
                           colorscale='blues',symbol='circle'),line=dict(color=line_col,width=self.width)))
            else:
                fig.add_traces(go.Scatter3d(x=activity_pc[:-1, 0],y=activity_pc[:-1, 1],
                           z=activity_pc[:-1, 2],marker=dict(size=3,color=np.arange(self.t_task),
                           colorscale='blues',symbol='circle'),mode='markers'))
            
            if self.colors[i]:
                fig.add_traces(go.Scatter3d(x=np.array(activity_pc[-1, 0]),y=np.array(activity_pc[-1, 1]),
                       z=np.array(activity_pc[-1, 2]),marker=dict(size=self.mark_sz,color=self.colors[i],symbol=self.marker),
                       line=dict(color='darkblue',width=self.width)))
                
        # Fixed points are shown in cross
        cols = ['firebrick','darkorange']
        for i in range(self.fixedpoints.shape[1]):
            fixedpoints_pc = self.pca.transform(self.fixedpoints[:,i])
            fig.add_traces(go.Scatter3d(x=fixedpoints_pc[:, 0],y=fixedpoints_pc[:, 1],
                      z=fixedpoints_pc[:, 2],marker=dict(size=2,color=cols[i],symbol='x'),
                      mode='markers',opacity=self.opacity))
            
        
        # Rotating figure
        
        fig.update_layout(
            title=self.net_file,
            showlegend=False,
            width=800,
            height=700,
            autosize=False,
            scene=dict(
                xaxis_title='PC 1',
                yaxis_title='PC 2',
                zaxis_title='PC 3',
                camera=dict(
                    # Determines 'up' direction
                    up=dict(
                        x=0,
                        y=0,
                        z=1
                    ),
                    # Determines view angle
                    eye=dict(
                        x=self.x_eye,
                        y=self.y_eye,
                        z=self.z_eye,
                    )
                ),
                aspectratio = dict( x=1, y=1, z=0.7 ),
                aspectmode = 'manual'
            ),
            updatemenus=[dict(type='buttons',
                     showactive=False,
                     y=1,
                     x=0.8,
                     xanchor='left',
                     yanchor='bottom',
                     pad=dict(t=45, r=10),
                     buttons=[dict(label='Play',
                                    method='animate',
                                    args=[None, dict(frame=dict(duration=5, redraw=True), 
                                                                transition=dict(duration=0),
                                                                fromcurrent=True,
                                                                mode='immediate'
                                                               )]
                                               )
                                         ]
                                 )
                           ]
        )
        
        fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
        
        # Save frames for rotating video
        frames=[]
        for t in np.arange(0, 6.26, 0.1):
            xe, ye, ze = self.rotate_z(-t)
            frames.append(go.Frame(layout=dict(scene_camera_eye=dict(x=xe, y=ye, z=ze))))
        fig.frames=frames
        
        fig.show()
        
        
    # Create rotating scene
    def rotate_z(self, theta):
        w = self.x_eye+1j*self.y_eye
        return np.real(np.exp(1j*theta)*w), np.imag(np.exp(1j*theta)*w), self.z_eye
    
    
    # Set plotting parameters
    def plot_params(self):
        
        if self.individual_examples:
            self.marker = 'circle'; self.mark_sz = 7; self.width = 3
            self.opacity = 0.4
        else:
            self.marker = 'square'; self.mark_sz = 5; self.width = 2
            self.opacity = 1
    
# Correlation coeffient between different matrices

def corr2_coeff(A, B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None],ssB[None]))


# Convert time to hours, minutes, seconds from seconds

def convert_seconds(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return hours, minutes, seconds

# Get device according to availability

def get_device():

    if torch.cuda.is_available():
        return torch.device('cuda')
    
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    
    else:
        return torch.device('cpu')
    
    
# Mix inputs with encoder

def encode(encoder,inp,n_dim,n_in):
    
    inp_temp = encoder(inp[:,:,-n_dim:])
    
    if n_in > n_dim:
        return torch.cat((inp[:,:,0].unsqueeze(2),inp_temp),dim=2)
    else:
        return inp_temp