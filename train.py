"""
Train recurrent neural network.
"""

import neurogym as ngym
import tasks

# Environment
kwargs = {'dt': 100}
seq_len = 100

# Make supervised dataset
dataset = ngym.Dataset(tasks.TwoAlternativeForcedChoice, env_kwargs=kwargs, 
                       batch_size=16, seq_len=seq_len)

# A sample environment from dataset
env = dataset.env
# Visualize the environment with 2 sample trials
_ = ngym.utils.plot_env(env, num_trials=2)

# Network input and output size
input_size = env.observation_space.shape[0]
output_size = env.action_space.n
