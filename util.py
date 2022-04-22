"""
Various utilities.
"""

import numpy as np

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