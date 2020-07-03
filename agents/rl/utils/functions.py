# -*- coding: utf-8 -*-

import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

def returns(rewards, dones, last_value, gamma = 0.95):
        
    returns = np.zeros_like(rewards)
    
    next_value = last_value
    for t in reversed(range(len(rewards))):
        returns[t] = rewards[t] + (1 - dones[t]) * gamma * next_value
        next_value = returns[t]
        
    return returns
    
def returns_est(rewards, dones, next_values, gamma = 0.95):
    
    returns = rewards + (1 - dones) * gamma * next_values
    
    return returns
  
def general_advantage_estimates(rewards, dones, values, next_values, lam, gamma = 0.95):
    ### GENERALIZED ADVANTAGE ESTIMATION
    # discount/bootstrap off value fn
    # We create mb_returns and mb_advantages
    # mb_returns will contain Advantage + value
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    lastgaelam = 0
    T = len(rewards)
    
    # From last step to first step
    for t in reversed(range(T)):
        # Delta = R(st) + gamma * V(t+1) * nextnonterminal  - V(st)
        delta = rewards[t] + (1- dones[t]) * gamma * next_values[t] - values[t]
        # Advantage = delta + gamma *  λ (lambda) * nextnonterminal  * lastgaelam
        advantages[t] = lastgaelam = delta + gamma * lam * (1- dones[t]) * lastgaelam
    # Returns
    returns = advantages + values
    
    return returns
    
def normalize(data):
    
    mean = np.mean(data)
    std = np.std(data)
    if std == 0: std = 1
    
    norm = (data - mean) / (std)
    
    return norm

def softmax(logits, legal_actions=None):
    
    _, num_actions = logits.get_shape().as_list()
    probs = tf.keras.activations.softmax(logits)
    probs = probs * tf.reduce_sum(tf.one_hot(legal_actions, num_actions), axis=0)
    probs = probs / tf.reduce_sum(probs, axis=1)
        
    return probs.numpy()

def argmax(logits, legal_actions=None):
    
    probs = softmax(logits, legal_actions)
    ts = tf.Variable(probs)
    argmax = tf.math.argmax(ts, axis=1)
    
    return argmax.numpy()