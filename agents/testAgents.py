import numpy as np


class FoldAgent(object):
    ''' Always fold  agent
    '''

    def __init__(self, action_num):
        ''' Initilize agent
        
        Args:
            action_num (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.action_num = action_num

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.
        
        Args:
            state (dict): An dictionary that represents the current state
        
        Returns:
            action (int): The action predicted (Action.FOLD: 0) by the agent
        '''
        return 0

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
        
        Args:
            state (dict): An dictionary that represents the current state
        
        Returns:
            action (int): The action predicted by the agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.action_num)]
        probs[0] = 1
        return self.step(state), probs
 
class CallAgent(object):
    ''' Always call/check agent
    '''

    def __init__(self, action_num):
        ''' Initilize the agent
       
        Args:
            action_num (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.action_num = action_num

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.
        
        Args:
            state (dict): An dictionary that represents the current state
        
        Returns:
            action (int): The action predicted (Action.CHECK: 1, Action.CALL: 2) by the agent
        '''
        if 1 in state['legal_actions']:
            return 1
        elif 2 in state['legal_actions']:
            return 2

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
        
        Args:
            state (dict): An dictionary that represents the current state
        
        Returns:
            action (int): The action predicted by the agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.action_num)]
        if 1 in state['legal_actions']:
            probs[1] = 1
        elif 2 in state['legal_actions']:
            probs[2] = 1
        return self.step(state), probs
    
class RiseAgent(object):
    ''' Always rise/all-in agent
    '''

    def __init__(self, action_num):
        ''' Initilize the agent
        
        Args:
            action_num (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.action_num = action_num

    @staticmethod
    def step(state):
        ''' Predict the action given the curent state in gerenerating training data.
        
        Args:
            state (dict): An dictionary that represents the current state
        
        Returns:
            action (int): The action predicted (Action.RAISE_POT: 4, Action.ALL_IN: 5) by the agent
        '''
        if 4 in state['legal_actions']:
            return 4
        elif 5 in state['legal_actions']:
            return 5
        else:
            return 0

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
        
        Args:
            state (dict): An dictionary that represents the current state
        
        Returns:
            action (int): The action predicted by the agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.action_num)]
        if 4 in state['legal_actions']:
            probs[4] = 1
        elif 5 in state['legal_actions']:
           probs[5] = 1
        else:
            probs[0] = 1
        return self.step(state), probs