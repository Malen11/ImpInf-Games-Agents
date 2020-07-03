# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

from agents.rl.a2c_v2_qpg import A2C
from agents.rl.utils.functions import softmax, argmax

class A2CQPGAgent(object):

    def __init__(self,
                 action_num=2,
                 state_shape=None,
                 trainble=True,
                 
                 critic_mlp_layers=[4,256],
                 critic_activation_func='tanh', 
                 critic_kernel_initializer='glorot_uniform',
                 critic_learning_rate=0.0001,
                 critic_bacth_size=128,
                 
                 actor_mlp_layers=[4,256],
                 actor_activation_func='tanh', 
                 actor_kernel_initializer='glorot_uniform',  
                 actor_learning_rate=0.0001,
                 actor_bacth_size=2048,
                 
                 discount_factor=0.99,
                 lam=0.5,
                 
                 entropy_coef=0.9,
                 entropy_decoy=1,
                 max_entropy_part=0.9,
                 
                 max_grad_norm = 0,
                 
                 shared_memory = None,
                 
                 min_reward=0,
                 max_reward=100):
        self.use_raw = False
        
        # Create estimators
        self.bot = A2C(
            num_state_params=state_shape[0],
            num_actions=action_num,
            trainble=trainble,
            
            critic_hidden_units=np.full((critic_mlp_layers[0]), critic_mlp_layers[1]), 
            critic_learning_rate=critic_learning_rate,
            critic_activation_func=critic_activation_func, 
            critic_kernel_initializer=critic_kernel_initializer,
            critic_bacth_size=critic_bacth_size,
            
            actor_hidden_units=np.full((actor_mlp_layers[0]), actor_mlp_layers[1]),
            actor_learning_rate=actor_learning_rate, 
            actor_activation_func=actor_activation_func, 
            actor_kernel_initializer=actor_kernel_initializer,  
            actor_bacth_size=actor_bacth_size,
            
            gamma=discount_factor, 
            lam = lam,
            
            entropy_coef=entropy_coef,
            entropy_decoy=entropy_decoy,
            max_entropy_part=max_entropy_part,
            
            max_grad_norm = max_grad_norm,
                 
            shared_memory = shared_memory,
            )
           
        # Total timesteps
        self.total_t = 0
        
        # Total training step
        self.train_t = 0
        
        #normalization
        self.min_reward = 0
        self.max_reward = 100
        
    def get_memory(self):
        return self.bot.get_memory()
        
    def feed_batch(self, batch):
        self.bot.feed_batch(batch)

    def feed(self, ts):
        (state, action, reward, next_state, done) = tuple(ts)
        self.bot.feed(
            state['obs'], 
            action,
            (reward-self.min_reward) / (self.max_reward-self.min_reward), 
            next_state['obs'], 
            done)
        
        self.total_t += 1
        
    def train(self):
        loss = self.bot.train()
        
        self.train_t += 1
        
        return loss

    def get_weights(self):
        return self.bot.get_weights()
        
    def set_weights(self, weights):
         self.bot.set_weights(weights)

    def step(self, state):
        return self.bot.get_action(state['obs'], state['legal_actions'])
    
    def eval_step(self, state):
        
        batch = [state['obs']]
        ts = tf.convert_to_tensor(batch)
        
        logits,_ = self.bot.predict(ts)
        probs = self.bot.softmax(logits, state['legal_actions'])[0]
        best_action = np.argmax(probs)
        return best_action, probs
    
    def save_model(self, path):
        self.bot.save_model(path)
        
    def load_model(self, path):
        self.bot.load_model(path)