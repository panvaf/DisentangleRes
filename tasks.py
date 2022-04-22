"""
Classes for tasks to train on.
"""

import numpy as np
import neurogym as ngym
from neurogym import spaces


class TwoAlternativeForcedChoice(ngym.TrialEnv):
    """Alternative forced choice task where two streams of evidence are 
    presented. The participant should decide which stream has provided the 
    greatest amount of evidence. 
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, 
                 rule_vec = None):
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

        self.timing = {
            'fixation': 100,
            'stimulus': 2000,
            'delay': 0,
            'decision': 100}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(3)

        name = {'fixation': 0, 'stimulus': range(1, 3)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3+rule_sz,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, 3), 'task': range(4, 4+rule_sz)}
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
        # observations
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}
    
    
    
class AttributeIntegration(ngym.TrialEnv):
    """Two independent streams of evidence are presented for the same stimulus.
    The participant should decide whether the total accumulated evidence exceeds
    a decision threshold. 
    
    For simplicity, the streams are modelled as constant inputs plus noise. 

    Inputs:
        sigma: float, input noise level
    """

    def __init__(self, dt=100, rewards=None, timing=None, sigma=1.0, 
                 rule_vec = None):
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

        self.timing = {
            'fixation': 100,
            'stimulus': 2000,
            'delay': 0,
            'decision': 100}
        if timing:
            self.timing.update(timing)

        self.abort = False

        self.choices = np.arange(3)

        name = {'fixation': 0, 'stimulus': range(1, 3)}
        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(3+rule_sz,), dtype=np.float32, name=name)
        name = {'fixation': 0, 'choice': range(1, 3), 'task': range(4, 4+rule_sz)}
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
        thres = 2*self.rng.rand()
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
        # observations
        if self.in_period('fixation'):
            if action != 0:  # action = 0 means fixating
                new_trial = self.abort
                reward += self.rewards['abort']
        elif self.in_period('decision'):
            if action != 0:
                new_trial = True
                if action == gt:
                    reward += self.rewards['correct']
                    self.performance = 1
                else:
                    reward += self.rewards['fail']

        return self.ob_now, reward, False, {'new_trial': new_trial, 'gt': gt}