"""
Classes for tasks to train on.
"""

import numpy as np
import neurogym as ngym
from neurogym import spaces
from math import tan


class TwoAlternativeForcedChoice(ngym.TrialEnv):
    """Alternative forced choice task where two streams of evidence are 
    presented. The participant should decide which stream has provided the 
    greatest amount of evidence. 
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
    """
    
    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, rule_vec = None):
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
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.Discrete(3, name=name)
        
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
        stim = self.rng.rand(2)
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
        self.set_groundtruth(ground_truth, period='decision', where='choice')

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
    
    
    
class AttributeIntegration(ngym.TrialEnv):
    """Two independent streams of evidence are presented for the same stimulus.
    The participant should decide whether the total accumulated evidence exceeds
    a decision threshold. 
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, rule_vec = None):
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
        name = {'fixation': 0, 'choice': range(1, 3)}
        self.action_space = spaces.Discrete(3, name=name)

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
        stim = self.rng.rand(2)
        thres = 1
        ground_truth = thres < np.sum(stim)
        
        trial = {
            'stim': stim,
            'thres': thres,
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
        self.add_ob(thres*self.rule_vec,['fixation', 'stimulus', 'delay', 'decision'],where='task')

        # Ground truth
        self.set_groundtruth(ground_truth, period='decision', where='choice')

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
    
    
    
class LinearClassification(ngym.TrialEnv):
    """Two independent streams of evidence are presented for the same stimulus.
    The participant should solve several binary classification tasks at the same time. 
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, thres = None):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
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
        self.add_ob(stim, 'stimulus', where='stimulus')
        self.add_randn(0, self.sigma, 'stimulus', where='stimulus')
        
        # Ground truth
        self.set_groundtruth(ground_truth+1, period='decision')

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

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, thres = None):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
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
    

class MultiplyClassification(ngym.TrialEnv):
    """Two independent streams of evidence are presented for the same stimulus.
    The participant should multiply the amounts of evidence and decide whether
    the result exceeds given thresholds.
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        
        # Thresholds for classification
        dthres = 1/(2*n_task)
        self.thres = np.linspace(dthres,1-dthres,n_task)*.15
        
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
        stim = self.rng.rand(2)*.5
        ground_truth = np.zeros(self.n_task)
        for i, thres in enumerate(self.thres):
            ground_truth[i] = stim[1]*stim[0] > thres
        
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
        self.set_groundtruth(ground_truth+1, period='decision')

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
    
    

class DivideClassification(ngym.TrialEnv):
    """Two independent streams of evidence are presented for the same stimulus.
    The participant should divide the amounts of evidence and decide whether
    the result exceeds given thresholds.
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        
        # Thresholds for classification
        ratios = np.linspace(.5,1,int(n_task/2),endpoint=False)
        self.thres = np.concatenate((ratios,1/ratios))
        
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
        stim = self.rng.rand(2)
        ground_truth = np.zeros(self.n_task)
        for i, thres in enumerate(self.thres):
            ground_truth[i] = stim[1]/stim[0] > thres
        
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
        self.set_groundtruth(ground_truth+1, period='decision')

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
    
    
class CircularClassification(ngym.TrialEnv):
    """Two independent streams of evidence are presented for the same stimulus.
    The participant figure out if the amounts of evidence lie within a circle.
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        
        # Thresholds for classification
        dthres = 1/(4*n_task)
        thres = np.linspace(0,.5-dthres,n_task+1)*np.sqrt(2)
        self.thres = thres[1:]
         
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
        for i, thres in enumerate(self.thres):
            ground_truth[i] = stim[1]**2 + stim[0]**2 > thres**2
        
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
        self.set_groundtruth(ground_truth+1, period='decision')

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
    

class Denoise(ngym.TrialEnv):
    """Two independent streams of evidence are presented for the same stimulus.
    The participant should report the noise-free estimates of inputs.
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        
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
        stim = self.rng.rand(2) - .5
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
    
    

class DenoiseSameSign(ngym.TrialEnv):
    """Same as Denoise but stimuli have the same sign

    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        
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
        stim = self.rng.rand(2) - .5
        sign = np.sign(stim[0]*stim[1])
        stim[0] = sign*stim[0]
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
    

class DenoiseQuads(ngym.TrialEnv):
    """Same as Denoise but choose which quadrants to include

    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, 
                 quad_num = np.array([1,2,3,4])):
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
    
    
class DenoiseQuadOut(ngym.TrialEnv):
    """Same as Denoise leave one quadrant out

    Inputs:
        sigma: float, input noise level
        n_task: number of classification tasks to be solved
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2):
        super().__init__(dt=dt)
        
        self.sigma = sigma / np.sqrt(self.dt)  # Input noise
        self.n_task = n_task
        
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
        stim = self.rng.rand(2) - .5
        while stim[0] > 0 and stim[1] < 0:
            stim = self.rng.rand(2) - .5
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

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, thres = None):
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
        self.set_groundtruth(ground_truth+1, period='decision')

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

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, n_task = 2, thres = 5):
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
