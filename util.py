"""
Various utilities.
"""

import numpy as np
from scipy import stats
import torch
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.cluster import DBSCAN
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
    
    
# Compute confusion matrix

def confusion_matrix(outputs, targets, threshold=0, labels = [-1,1]):
    """
    Apply thresholding to model outputs, classify them into -1 or 1,
    and compute the confusion matrix for binary classification.

    Parameters:
    - outputs: numpy array of model outputs, shape (batch_sz,)
    - targets: numpy array of targets, shape (batch_sz,)
    - threshold: float, the value used to classify outputs into -1 or 1

    Returns:
    - confusion_matrix: numpy array, shape (2, 2), format [[TN, FP], [FN, TP]]
    """
    # Classify model outputs based on the threshold
    predictions = np.where(outputs >= threshold, labels[1], labels[0])
    
    # Initialize the confusion matrix
    confusion_matrix = np.zeros((2, 2), dtype=int)
    
    # Calculate True Negatives (TN), False Positives (FP), False Negatives (FN), True Positives (TP)
    TN = np.sum((predictions == labels[0]) & (targets == labels[0]))
    FP = np.sum((predictions == labels[1]) & (targets == labels[0]))
    FN = np.sum((predictions == labels[0]) & (targets == labels[1]))
    TP = np.sum((predictions == labels[1]) & (targets == labels[1]))
    
    # Fill in the confusion matrix
    confusion_matrix[0][0] = TN
    confusion_matrix[0][1] = FP
    confusion_matrix[1][0] = FN
    confusion_matrix[1][1] = TP
    
    return confusion_matrix



def angles_between_vectors(vectors):
    """
    Calculate the angles between all pairs of vectors in degrees.
    Returns a matrix with the upper triangular part containing angles,
    and the lower triangular part (including diagonal) filled with NaN.
    
    Parameters:
    vectors (np.ndarray): A 2D numpy array where each column is a vector.
                          Shape: (vector_dim, number_of_vectors)
    
    Returns:
    np.ndarray: A matrix where the upper triangular part contains angles in degrees,
                and the lower triangular part (including diagonal) contains NaN.
    """
    # Normalize the vectors
    normalized = vectors / np.linalg.norm(vectors, axis=0)
    
    # Calculate dot products between all pairs of vectors
    dot_products = np.dot(normalized.T, normalized)
    
    # Clip the dot products to [-1, 1] to avoid numerical errors
    dot_products = np.clip(dot_products, -1.0, 1.0)
    
    # Calculate the angles using arccos and convert to degrees
    angles = np.degrees(np.arccos(dot_products))
    
    # Create a mask for the lower triangle and diagonal
    mask = np.tril(np.ones_like(angles))
    
    # Apply the mask: set lower triangle and diagonal to NaN
    angles[mask == 1] = np.nan
    
    return angles


def mean_confidence_intervals(data, confidence=0.95):
    """
    Calculate the mean and confidence interval for each column in a matrix of measurements.
    
    Parameters:
    data (np.ndarray): 2D array where each column is a set of measurements
    confidence (float): The confidence level, default is 0.95 for 95% CI
    
    Returns:
    tuple: (means, lower_bounds, upper_bounds)
    """
    data = np.array(data)
    
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    
    n_samples, n_columns = data.shape
    
    means = np.nanmean(data, axis=0)
    ses = stats.sem(data, axis=0, nan_policy='omit')
    
    # Calculate the t-value for the given confidence level and degrees of freedom
    t_values = stats.t.ppf((1 + confidence) / 2, n_samples - 1)
    
    margins_of_error = t_values * ses
    
    lower_bounds = means - margins_of_error
    upper_bounds = means + margins_of_error
    
    return lower_bounds, means, upper_bounds



def find_variance_along_directions(data, directions):
    # Standardize the data
    
    Q, R = np.linalg.qr(directions.T)
    orthogonal_directions = Q.T
    
    # Project data onto orthogonalized directions
    projections = np.dot(data, orthogonal_directions.T)
    
    # Compute variance along each orthogonal direction
    variances = np.var(projections, axis=0, ddof=1)  # Using ddof=1 for sample variance
    
    # Compute the total variance of the original data
    total_variance = np.var(data, axis=0, ddof=1).sum()
    
    # Calculate cumulative variance
    cumulative_variance = np.cumsum(variances)
    
    # Percentage of variance explained
    percentage_variance_explained = (variances / total_variance) * 100
    
    return variances, cumulative_variance, percentage_variance_explained


def random_orthogonality_test(vectors, n_samples=1000):
    """
    Compare orthogonality of vectors to that of random vectors and obtain p-values
    """
    n_vectors = vectors.shape[1]
    actual_dots = np.abs(np.dot(vectors.T, vectors))

    random_dots = []
    for _ in range(n_samples):
        random_vecs = np.random.randn(64, n_vectors)
        random_vecs /= np.linalg.norm(random_vecs, axis=0)
        random_dots.append(np.abs(np.dot(random_vecs.T, random_vecs)))

    random_dots = np.array(random_dots)
    print(random_dots.shape)
    p_values = np.mean(random_dots <= actual_dots, axis=0)

    return p_values


def compute_jacobian(net, inp, fp):
    n_hidden = len(fp)
    fp = fp.clone().requires_grad_(True)
    deltah = net.dynamics(inp, fp) - fp
    
    jacobian = torch.zeros(n_hidden, n_hidden)
    for i in range(n_hidden):
        grad_output = torch.zeros_like(deltah)
        grad_output[i] = 1.0
        grad = torch.autograd.grad(deltah, fp, grad_outputs=grad_output, retain_graph=True)[0]
        jacobian[i, :] = grad
    
    return jacobian


def get_unique_fp(fixed_points,eps=0.5,min_samples=1):
    """
    Clusters fixed points according to distance and keeps only unique ones, i.e.
    ones with distance greater than eps.
    
    Eps can be determined by e.g. looking at the elbow at nearest neighbor
    distance graph.
    """
    
    # Perform DBSCAN clustering
    db = DBSCAN(eps=eps, min_samples=min_samples, metric='euclidean').fit(fixed_points)
    
    # Get labels assigned by DBSCAN
    labels = db.labels_
    
    # Extract unique fixed points (one from each cluster)
    unique_fixed_points = []
    unique_labels = set(labels)
    
    for label in unique_labels:
        if label != -1:  # Exclude noise points if any
            indices = np.where(labels == label)[0]
            # Choose the first fixed point in the cluster as representative
            unique_fixed_points.append(fixed_points[indices[0]])
    
    unique_fixed_points = np.array(unique_fixed_points)
    
    print("Number of unique fixed points:", len(unique_fixed_points))
    
    return unique_fixed_points



def compute_sparseness(firing_rates):
    """
    Computes the sparseness of neural responses for a given firing rate matrix,
    handling cases where all firing rates of a neuron are zero.

    Parameters:
        firing_rates (np.ndarray): A 2D array of shape (m, n), where m is the number of movie frames 
                                   and n is the number of neurons. Each element represents the firing rate
                                   of a neuron for a particular frame.

    Returns:
        np.ndarray: A 1D array of sparseness values for each neuron (length n).
                    Returns 100% sparseness for neurons with all zero firing rates.
    """
    # Number of movie frames (m) and neurons (n)
    m, n = firing_rates.shape

    # Compute the sparseness for each neuron
    sparseness_values = []
    for i in range(n):
        ri = firing_rates[:, i]
        if np.all(ri == 0):
            # Handle case where all firing rates are zero
            sparseness_values.append(1.0)  # Maximum sparseness (100%)
        else:
            numerator = (ri.mean()) ** 2
            denominator = (ri ** 2).mean()
            S = (1 - (numerator / denominator)) / (1 - (1 / m))
            sparseness_values.append(S)

    return np.array(sparseness_values)