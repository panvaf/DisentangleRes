"""
Various utilities.
"""

import numpy as np
import torch

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
    norm = torch.sum(mask)
    
    avg_loss = loss/size
    normed_loss = loss/norm
    
    return avg_loss, normed_loss