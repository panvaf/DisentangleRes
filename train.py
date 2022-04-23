"""
Train recurrent neural network.
"""

import neurogym as ngym
from neurogym import spaces
import tasks
import util

# Tasks
task = ['TwoAlternativeForcedChoice','AttributeIntegration']
task_rules = util.assign_task_rules(task)

# Environment
seq_len = 100
tenv = tasks.AttributeIntegration(rule_vec=task_rules['AttributeIntegration'])

# Make supervised dataset
dataset = ngym.Dataset(tenv, batch_size=16, seq_len=seq_len)

# A sample environment from dataset
env = dataset.env
# Visualize the environment with 2 sample trials
_ = ngym.utils.plot_env(env, num_trials=2)

# Network input and output size
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
