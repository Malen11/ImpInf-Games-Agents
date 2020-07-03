# -*- coding: utf-8 -*-

import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

from agents.rl.ddqn import DDQN

class DDQNAgent(object):

    def __init__(self,
                 replay_memory_size=10000,
                 replay_memory_init_size=100,
                 update_target_estimator_every=1000,
                 discount_factor=0.95,
                 epsilon=1,
                 min_epsilon=0.1,
                 epsilon_decay_coef=0.99,
                 batch_size=64,
                 action_num=2,
                 state_shape=None,
                 train_every=1,
                 mlp_layers=[4,512],
                 learning_rate=0.00005,
                 activation_func='tanh', 
                 kernel_initializer='glorot_uniform'):
    
        self.use_raw = False
        self.replay_memory_init_size = replay_memory_init_size
        self.update_target_estimator_every = update_target_estimator_every
        self.discount_factor = discount_factor
        self.batch_size = batch_size
   
        # Total timesteps
        self.total_t = 0

        # Total training step
        self.train_t = 0
        self.train_every = train_every

        # The epsilon decay scheduler
        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_coef = epsilon_decay_coef
        
        # Create estimators
        self.bot = DDQN(
            num_state_params=state_shape[0],
            num_actions=action_num,
            hidden_units=np.full((mlp_layers[0]), mlp_layers[1]), 
            gamma=discount_factor, 
            max_replay_num=replay_memory_size, 
            min_replay_num=replay_memory_init_size, 
            batch_size=batch_size, 
            learning_rate=learning_rate,
            activation_func=activation_func, 
            kernel_initializer=kernel_initializer
            )

    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the memory without training
            In stage 2, train the agent every several timesteps
        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        
        (state, action, reward, next_state, done) = tuple(ts)
        
        self.bot.feed(state['obs'], action, reward, next_state['obs'], done)
        self.total_t += 1
        
        tmp = self.total_t - self.replay_memory_init_size
        if tmp >= 0 and tmp % self.train_every == 0:
            self.train()
            
    def train(self):
        self.bot.train()
        self.train_t += 1
        # Update the target estimator
        if self.train_t % self.update_target_estimator_every == 0:
            self.bot.update_target_net()
            print("\nINFO - Copied model parameters to target network.")


    def step(self, state):
        
        action = self.bot.get_action(state['obs'], state['legal_actions'], self.epsilon)
        
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay_coef)
        
        return action

    def eval_step(self, state):
        
        logits = self.bot.predict(state['obs'])
        probs = self.bot.softmax(logits, state['legal_actions'])[0]
        best_action = self.bot.argmax(logits, state['legal_actions'])[0]
        
        return best_action, probs
    
    def save_model(self, path):
        self.bot.save_model(path)
        
    def load_model(self, path):
        self.bot.load_model(path)