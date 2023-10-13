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
                 n_in=3,colors=None):
        
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
        self.n_task = trial_info[0]['ground_truth'].shape[0]
        
        # Constants
        self.x_eye = 0
        self.y_eye = 1.0707
        self.z_eye = 1
        
        # Colors
        self.colors = self.get_colors(colors)
        
    # Get the colors for the final dot in the scatterplot
    def get_colors(self,colors):
        
        if isinstance(colors[0],list):
            self.marker = 'square'; self.mark_sz = 5
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
                    else:
                        k = 1 if trial['ground_truth'][-1][0] > 0 else 0
                        l = 1 if trial['ground_truth'][-1][int(self.n_task/2)] > 0 else 0
                    quad_col = colors[k][l]
                    
                col.append(quad_col)
            
        else:
            col = colors
            self.marker = 'circle'; self.mark_sz = 7
            
        return col
    
                
    # Create plot
    def plot(self):
        
        fig = go.Figure()

        for i in range(self.n_trial):
            activity_pc = self.pca.transform(self.activity[i])
            
            fig.add_traces(go.Scatter3d(x=activity_pc[:, 0],y=activity_pc[:, 1],
                       z=activity_pc[:, 2],marker=dict(size=3,color=np.arange(self.t_task),
                       colorscale='blues',symbol='circle'),line=dict(color='darkblue',width=2)))
            if self.colors[i]:
                fig.add_traces(go.Scatter3d(x=np.array(activity_pc[-1, 0]),y=np.array(activity_pc[-1, 1]),
                       z=np.array(activity_pc[-1, 2]),marker=dict(size=self.mark_sz,color=self.colors[i],symbol=self.marker),
                       line=dict(color='darkblue',width=2)))
                
        # Fixed points are shown in cross
        cols = ['firebrick','yellow']
        for i in range(self.fixedpoints.shape[1]):
            fixedpoints_pc = self.pca.transform(self.fixedpoints[:,i])
            fig.add_traces(go.Scatter3d(x=fixedpoints_pc[:, 0],y=fixedpoints_pc[:, 1],
                      z=fixedpoints_pc[:, 2],marker=dict(size=2,color=cols[0],symbol='x'),
                      mode='markers'))
            
        
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