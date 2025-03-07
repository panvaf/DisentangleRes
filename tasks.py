"""
Classes for tasks to train on.
"""

import numpy as np
import neurogym as ngym
from neurogym import spaces
from math import tan
from itertools import product
from scipy.stats import norm


class TwoAlternativeForcedChoiceCent(ngym.TrialEnv):
    """Centered alternative forced choice task where two streams of evidence are 
    presented. The participant should decide which stream has provided the 
    greatest amount of evidence.
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
    """
    
    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, rule_vec = None, **kwargs):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        
        # Rule vector
        if rule_vec is None:
            rule_sz = 0
        else:
            self.rule_vec = rule_vec; rule_sz = len(rule_vec)

        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)
        
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(3)

        name = {'fixation': 0, 'stimulus': range(1, 3), 'task': range(3, 3+rule_sz)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3+rule_sz,), dtype=np.float32, name=name)
        self.action_space = spaces.MultiDiscrete(np.repeat([3],1))
        
    def _new_trial(self, **kwargs):
        """
        Initialize a trial.
        Sets the following variables:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            stim: stimulus strenghts (evidence) for the trial
            obs: observation
        """
        # Trial info
        stim = self.rng.rand(2) - .5
        ground_truth = stim[1] > stim[0]
        
        trial = {
            'stim': stim,
            'ground_truth': ground_truth
        }
        trial.update(kwargs)

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')
        
        # Task rule
        if self.rule_vec is not None:
            self.add_ob(self.rule_vec,['fixation', 'stimulus', 'delay', 'decision'],where='task')

        # Ground truth
        self.set_groundtruth(2*ground_truth-1, period='decision')

        return trial
    
    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
    
    
class AttributeIntegrationCent(ngym.TrialEnv):
    """Centered version of Attribute Integration
    
    Two independent streams of evidence are presented for the same stimulus.
    The participant should decide whether the total accumulated evidence exceeds
    a decision threshold. 
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
    """
    
    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, rule_vec = None, **kwargs):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        # Rule vector
        if rule_vec is None:
            self.rule_vec = 1; rule_sz = 1
        else:
            self.rule_vec = rule_vec; rule_sz = len(rule_vec)
        
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)
        
        if timing:
            self.timing.update(timing)
        
        self.abort = False

        self.choices = np.arange(3)

        name = {'fixation': 0, 'stimulus': range(1, 3), 'task': range(3, 3+rule_sz)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3+rule_sz,), dtype=np.float32, name=name)
        self.action_space = spaces.MultiDiscrete(np.repeat([3],1))

    def _new_trial(self, **kwargs):
        """
        Initialize a trial.
        Sets the following variables:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            stim: stimulus strenghts (evidence) for the trial
            obs: observation
        """
        # Trial info
        stim = self.rng.rand(2) - .5
        
        ground_truth = np.sum(stim) > 0
        
        trial = {
            'stim': stim,
            'ground_truth': ground_truth
        }
        trial.update(kwargs)

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')
        
        # Task rule
        self.add_ob(self.rule_vec,['fixation', 'stimulus', 'delay', 'decision'],where='task')

        # Ground truth
        self.set_groundtruth(2*ground_truth-1, period='decision')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
    
    
class LinearClassificationCentOut(ngym.TrialEnv):
    """Two independent streams of evidence are presented for the same stimulus.
    The participant should solve several binary classification tasks at the same time. 
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, dist = 'gauss', **kwargs):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        self.dist = dist
        
        # Divide plane in evenly spaced classification problems
        dphi = np.pi/n_task
        phis = np.arange(n_task)*dphi + dphi/2
        self.alphas = [tan(phi) for phi in phis]
        
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(3)

        name = {'fixation': 0, 'stimulus': range(1, 3), 'task': range(3, 3)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.MultiDiscrete(np.repeat([3],self.n_task))

    def _new_trial(self, **kwargs):
        """
        Initialize a trial.
        Sets the following variables:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            stim: stimulus strenghts (evidence) for the trial
            obs: observation
        """
        # Trial info
        stim = self.rng.rand(2) - .5
        ground_truth = np.zeros(self.n_task)
        for i, alpha in enumerate(self.alphas):
            ground_truth[i] = stim[1] > alpha*stim[0]
        
        trial = {
            'stim': stim,
            'ground_truth': ground_truth
        }
        trial.update(kwargs)

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        
        if self.dist == 'gauss':
            self.add_ob(stim, 'stimulus', where='stimulus')
            self.add_randn(0, self.sigma, 'stimulus', where='stimulus')
        elif self.dist == 'gumbel':
            noise = np.random.gumbel(0,self.sigma,(int(self.timing['stimulus']/self.dt),2))
            self.add_ob(stim+noise, 'stimulus', where='stimulus')
        
        # Ground truth
        self.set_groundtruth(2*ground_truth-1, period='decision')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
    


class LinearClassificationAnisot(ngym.TrialEnv):
    """
    LinearClassificationCentOut with anisotropic noise
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, **kwargs):
        super().__init__(dt=dt)
        
        self.sigma_1 = sigma / np.sqrt(self.dt)  # Input noise
        self.sigma_2 = self.sigma_1 * 2
        self.n_task = n_task
        
        # Divide plane in evenly spaced classification problems
        dphi = np.pi/n_task
        phis = np.arange(n_task)*dphi + dphi/2
        self.alphas = [tan(phi) for phi in phis]
        
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(3)

        name = {'fixation': 0, 'stimulus_1': range(1, 2), 'stimulus_2': range(2, 3), 'task': range(3, 3)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.MultiDiscrete(np.repeat([3],self.n_task))

    def _new_trial(self, **kwargs):
        """
        Initialize a trial.
        Sets the following variables:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            stim: stimulus strenghts (evidence) for the trial
            obs: observation
        """
        # Trial info
        stim = self.rng.rand(2) - .5
        ground_truth = np.zeros(self.n_task)
        for i, alpha in enumerate(self.alphas):
            ground_truth[i] = stim[1] > alpha*stim[0]
        
        trial = {
            'stim': stim,
            'ground_truth': ground_truth
        }
        trial.update(kwargs)

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        self.add_ob(stim[0], 'stimulus', where='stimulus_1')
        self.add_ob(stim[1], 'stimulus', where='stimulus_2')
        self.add_randn(0, self.sigma_1, 'stimulus', where='stimulus_1')
        self.add_randn(0, self.sigma_2, 'stimulus', where='stimulus_2')
        
        # Ground truth
        self.set_groundtruth(2*ground_truth-1, period='decision')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}



class LinearClassificationAutocorr(ngym.TrialEnv):
    """
    LinearClassificationCentOut with autocorrelated noise
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, ar_coef = 0.7, **kwargs):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        self.ar_coef = ar_coef
        
        # Divide plane in evenly spaced classification problems
        dphi = np.pi/n_task
        phis = np.arange(n_task)*dphi + dphi/2
        self.alphas = [tan(phi) for phi in phis]
        
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(3)

        name = {'fixation': 0, 'stimulus': range(1, 3), 'task': range(3, 3)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.MultiDiscrete(np.repeat([3],self.n_task))
        
        
    def generate_ar_noise(self, size, sigma):
        """Generate autocorrelated noise using an AR(1) process"""
        noise = np.zeros((size,2))
        noise[0,:] = self.rng.normal(0, sigma, 2)
        for i in range(2):
            for t in range(1, size):
                noise[t,i] = self.ar_coef * noise[t-1,i] + self.rng.normal(0, sigma * np.sqrt(1 - self.ar_coef**2))
        return noise
    

    def _new_trial(self, **kwargs):
        """
        Initialize a trial.
        Sets the following variables:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            stim: stimulus strenghts (evidence) for the trial
            obs: observation
        """
        # Trial info
        stim = self.rng.rand(2) - .5
        ground_truth = np.zeros(self.n_task)
        for i, alpha in enumerate(self.alphas):
            ground_truth[i] = stim[1] > alpha*stim[0]
        
        trial = {
            'stim': stim,
            'ground_truth': ground_truth
        }
        trial.update(kwargs)
        
        # autocorrelated noise
        noise = self.generate_ar_noise(int(self.timing['stimulus']/self.dt), self.sigma)

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        self.add_ob(stim+noise, 'stimulus', where='stimulus')

        
        # Ground truth
        self.set_groundtruth(2*ground_truth-1, period='decision')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
    

    
class LinearClassificationCorr(ngym.TrialEnv):
    """Extension of linear multitask classification for correlated evidence streams.

    Inputs:
        sigma: float, input noise level
        n_task: int, number of classification tasks to be solved
        corr: float, correlation coefficient
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2,
                 corr = 0.7, **kwargs):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        self.cov_matrix = [[1, corr], [corr, 1]]
        
        # Divide plane in evenly spaced classification problems
        dphi = np.pi/n_task
        phis = np.arange(n_task)*dphi + dphi/2
        self.alphas = [tan(phi) for phi in phis]
        
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(3)

        name = {'fixation': 0, 'stimulus': range(1, 3), 'task': range(3, 3)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.MultiDiscrete(np.repeat([3],self.n_task))

    def _new_trial(self, **kwargs):
        """
        Initialize a trial.
        Sets the following variables:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            stim: stimulus strenghts (evidence) for the trial
            obs: observation
        """
        # Trial info
        stim = self.rng.multivariate_normal([0, 0], self.cov_matrix)
        stim = norm.cdf(stim) - .5
        ground_truth = np.zeros(self.n_task)
        for i, alpha in enumerate(self.alphas):
            ground_truth[i] = stim[1] > alpha*stim[0]
        
        trial = {
            'stim': stim,
            'ground_truth': ground_truth
        }
        trial.update(kwargs)

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')
        
        # Ground truth
        self.set_groundtruth(2*ground_truth-1, period='decision')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
    
    

class LinearClassificationHighDim(ngym.TrialEnv):
    """Extension of the multitask classification objective to arbitrary
    dimension of the input
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
        n_task: int, number of classification tasks to be solved
        n_dim: int, dimensionality of state space
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task=2, n_dim=3, **kwargs):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        self.n_dim = n_dim
        
        # Randomly sample normal vectors for n_dim-1 dimensional hyperplanes
        normals = np.random.randn(n_task, n_dim)
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        self.normals = normals / norms
        
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(3)

        name = {'fixation': 0, 'stimulus': range(1, n_dim+1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(n_dim+1,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.MultiDiscrete(np.repeat([3],self.n_task))

    def _new_trial(self, **kwargs):
        """
        Initialize a trial.
        Sets the following variables:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            stim: stimulus strenghts (evidence) for the trial
            obs: observation
        """
        # Trial info
        stim = self.rng.rand(self.n_dim) - .5
        ground_truth = np.zeros(self.n_task)
        for i, normal in enumerate(self.normals):
            ground_truth[i] = np.dot(stim,normal) > 0
        
        trial = {
            'stim': stim,
            'ground_truth': ground_truth
        }
        trial.update(kwargs)

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')
        
        # Ground truth
        self.set_groundtruth(2*ground_truth-1, period='decision')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


class DenoiseQuads(ngym.TrialEnv):
    """Recover ground truth from noisy observations

    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, 
                 quad_num = np.array([1,2,3,4]), **kwargs):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        self.quads = np.array([[0.5,0.5],[-0.5,0.5],[-0.5,-0.5],[0.5,-0.5]])[quad_num-1]
        
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(3)

        name = {'fixation': 0, 'stimulus': range(1, 3), 'task': range(3, 3)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape=(self.n_task,), dtype=np.float32)
        
    def _new_trial(self, **kwargs):
        """
        Initialize a trial.
        Sets the following variables:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            stim: stimulus strenghts (evidence) for the trial
            obs: observation
        """
        # Trial info
        quad = self.quads[np.random.choice(self.quads.shape[0])]
        stim = self.rng.rand(2)*quad
        ground_truth = stim
        
        trial = {
            'stim': stim,
            'ground_truth': ground_truth
        }
        trial.update(kwargs)

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')
        
        # Ground truth
        self.set_groundtruth(ground_truth, period='decision')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


class DenoiseQuadsHighDim(ngym.TrialEnv):
    """Extension of denoise task to arbitrary dimensions

    Inputs:
        sigma: float, input noise level
        n_task: int, number of classification tasks to be solved
        n_dim: int, dimensionality of state space
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, 
                 n_dim = 3, quad_num = None, **kwargs):
        super().__init__(dt=dt)
        
        if quad_num is None:
           quad_num = np.arange(2**n_dim).astype(int) + 1
           
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        self.n_dim = n_dim
        self.quads = np.array(list((product([-.5, .5], repeat=n_dim))))[quad_num-1]
        
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(n_dim+1)

        name = {'fixation': 0, 'stimulus': range(1, n_dim+1)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(n_dim+1,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, n_dim+1)}
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape=(self.n_dim,), dtype=np.float32)
        
    def _new_trial(self, **kwargs):
        """
        Initialize a trial.
        Sets the following variables:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            stim: stimulus strenghts (evidence) for the trial
            obs: observation
        """
        # Trial info
        quad = self.quads[np.random.choice(self.quads.shape[0])]
        stim = self.rng.rand(self.n_dim)*quad
        ground_truth = stim
        
        trial = {
            'stim': stim,
            'ground_truth': ground_truth
        }
        trial.update(kwargs)

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')
        
        # Ground truth
        self.set_groundtruth(ground_truth, period='decision')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


class DenoiseQuadsFreeRT(ngym.TrialEnv):
    """Same as Denoise but choose which quadrants to include, for free RT task 
        (information keeps coming in and decision is made at arbitrary point)

    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, 
                 quad_num = np.array([1,2,3,4]), **kwargs):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        self.quads = np.array([[0.5,0.5],[-0.5,0.5],[-0.5,-0.5],[0.5,-0.5]])[quad_num-1]
        
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(3)

        name = {'fixation': 0, 'stimulus': range(1, 3), 'task': range(3, 3)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.Box(
            -np.inf, np.inf, shape=(self.n_task,), dtype=np.float32)
        
    def _new_trial(self, **kwargs):
        """
        Initialize a trial.
        Sets the following variables:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            stim: stimulus strenghts (evidence) for the trial
            obs: observation
        """
        # Trial info
        quad = self.quads[np.random.choice(self.quads.shape[0])]
        stim = self.rng.rand(2)*quad
        ground_truth = stim
        
        trial = {
            'stim': stim,
            'ground_truth': ground_truth
        }
        trial.update(kwargs)

        # Periods
        self.add_period(['fixation', 'stimulus', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus'], where='fixation')
        self.add_ob(stim, period=['stimulus','decision'], where='stimulus')
        self.add_randn(0, self.sigma, period=['stimulus','decision'], where='stimulus')
        
        # Ground truth
        self.set_groundtruth(ground_truth, period='decision')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}


# Multiply classification for all 4 quadrants

class MultiplyClassificationFull(ngym.TrialEnv):
    """Two independent streams of evidence are presented for the same stimulus.
    The participant should multiply the amounts of evidence and decide whether
    the result exceeds given thresholds.
    
    For simplicity, the streams are modelled as constant inputs plus noise. 
    
    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, thres = None, **kwargs):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        
        # Thresholds for classification
        if thres is None:
            dthres = 1/(2*n_task)
            self.thres = np.linspace(dthres,1,int(n_task/4))*.2
        else:
            self.thres = thres
        
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(3)

        name = {'fixation': 0, 'stimulus': range(1, 3), 'task': range(3, 3)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.MultiDiscrete(np.repeat([3],self.n_task))

    def _new_trial(self, **kwargs):
        """
        Initialize a trial.
        Sets the following variables:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            stim: stimulus strenghts (evidence) for the trial
            obs: observation
        """
        # Trial info
        stim = self.rng.rand(2) - .5
        ground_truth = np.zeros(self.n_task)
        if stim[0] > 0:
            if stim[1] > 0:
                # 1st quadrant
                ground_truth[0:int(self.n_task/4)] = stim[1]*stim[0] > self.thres
            elif stim[1] < 0:
                # 4th quadrant
                ground_truth[-int(self.n_task/4):] = stim[1]*stim[0] < - self.thres
        elif stim[0] < 0:
            if stim[1] < 0:
                # 3rd quadrant
                ground_truth[int(self.n_task/2):int(3*self.n_task/4)] = stim[1]*stim[0] > self.thres
            elif stim[1] > 0:
                # 2nd quadrant
                ground_truth[int(self.n_task/4):int(self.n_task/2)] = stim[1]*stim[0] < - self.thres
        
        trial = {
            'stim': stim,
            'ground_truth': ground_truth
        }
        trial.update(kwargs)

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(1, period=['fixation', 'stimulus', 'delay'], where='fixation')
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')
        
        # Ground truth
        self.set_groundtruth(2*ground_truth-1, period='decision')

        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
    

# Terminate trial once boundary has been reached     

class LinearClassificationBound(ngym.TrialEnv):
    """Two independent streams of evidence are presented for the same stimulus.
    The participant should solve several binary classification tasks at the same time. 
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
        thres: threshold level
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, thres = 5, **kwargs):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        self.t_task = int(sum(timing.values())/self.dt)
        self.thres = thres
        
        # Divide plane in evenly spaced classification problems
        dphi = np.pi/n_task
        phis = np.arange(n_task)*dphi + dphi/2
        self.alphas = [tan(phi) for phi in phis]
        
        # Rewards
        self.rewards = {'abort': -0.1, 'correct': +1., 'fail': 0.}
        if rewards:
            self.rewards.update(rewards)

        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(3)

        name = {'stimulus': range(0, 2)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(2,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.MultiDiscrete(np.repeat([3],self.n_task))

    def _new_trial(self, **kwargs):
        """
        Initialize a trial.
        Sets the following variables:
            durations, which stores the duration of the different periods
            ground truth: correct response for the trial
            stim: stimulus strenghts (evidence) for the trial
            obs: observation
        """
        # Trial info
        stim = self.rng.rand(2) - .5
        ground_truth = np.zeros((self.t_task,self.n_task))

        # Periods
        self.add_period(['fixation', 'stimulus', 'delay', 'decision'])

        # Observations
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')
        
        # Ground truth
        for i, alpha in enumerate(self.alphas):
            ground_truth[:,i] = np.cumsum(self.ob[:,1] - alpha*self.ob[:,0])
            
            for j in range(self.t_task):
                if ground_truth[j,i] > self.thres:
                    ground_truth[j:,i] = self.thres
                elif ground_truth[j,i] < - self.thres:
                    ground_truth[j:,i] = - self.thres
        
        self.set_groundtruth(ground_truth[0:int(self.timing['stimulus']/100),:], period='stimulus')
        if self.timing['delay'] > 0:
            self.set_groundtruth(ground_truth[-int(self.timing['delay']/100):,:], period='delay')
        
        trial = {
            'stim': stim,
            'ground_truth': ground_truth
        }
        trial.update(kwargs)
        
        return trial

    def _step(self, action):
        """
        _step receives an action and returns:
            a new observation, obs
            reward associated with the action, reward
            a boolean variable indicating whether the experiment has end, done
            a dictionary with extra information:
                ground truth correct response, info['gt']
                boolean indicating the end of the trial, info['new_trial']
        """
        new_trial = False
        # rewards
        reward = 0
        gt = self.gt_now

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}