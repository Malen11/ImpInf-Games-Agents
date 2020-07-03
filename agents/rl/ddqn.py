''' DQN agent
'''

import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

from agents.rl.utils.memory import ReplayMemory
from agents.rl.models.neural_network_models import SimpleNeuralNetworkModel
from pprint import pprint

class DDQN(object):

    def __init__(self,
                 num_state_params, 
                 num_actions, 
                 hidden_units, 
                 gamma=0.99, 
                 max_replay_num=10000, 
                 min_replay_num=100, 
                 batch_size=32, 
                 learning_rate=0.00005,
                 activation_func='tanh', 
                 kernel_initializer='RandomNormal',
                 #train_q_net_every=1,
                 #update_target_net_every=1000
                 ):
        '''
        DDQN алгоритм
        
        Parameters
        ----------
        num_state_params : int
            DESCRIPTION.
        num_actions : int
            DESCRIPTION.
        hidden_units : TYPE
            DESCRIPTION.
        gamma : float32, optional
            DESCRIPTION. The default is 0.99.
        max_replay_num : int, optional
            DESCRIPTION. The default is 10000.
        min_replay_num : int, optional
            DESCRIPTION. The default is 100.
        batch_size : int, optional
            DESCRIPTION. The default is 32.
        learning_rate : float32, optional
            DESCRIPTION. The default is 0.00005.
        train_every : int, optional
            DESCRIPTION. The default is 1.
        update_target_net_every : int, optional
            DESCRIPTION. The default is 1000.

        Returns
        -------
        None.

        '''
        
        #Параметры сети
        self.num_actions = num_actions
        self.num_state_params = num_state_params
        self.hidden_units = hidden_units
        
        #параметры обучения
        self.batch_size = batch_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)   
        #self.train_every = train_every
        #self.update_target_net_every = update_target_net_every
        self.train_step = 0
        
        #параметры предсказания
        self.gamma = gamma
        
        #две сети-обучаемая и таргет(для обучения)
        self.q_net = SimpleNeuralNetworkModel(
            num_state_params,
            hidden_units, 
            num_actions,
            activation_func=activation_func, 
            kernel_initializer=kernel_initializer)
        self.target_net = SimpleNeuralNetworkModel(
            num_state_params, 
            hidden_units, 
            num_actions,
            activation_func=activation_func, 
            kernel_initializer=kernel_initializer)
        
        #память
        self.replay_memory = ReplayMemory(max_replay_num=max_replay_num,
                                          min_replay_num=min_replay_num)
    
    def predict(self, inputs, training=False):
        return self.q_net(np.atleast_2d(inputs.astype('float32')))
    
    def _target_net_predict(self, inputs):
        '''
        Выполняет предсказание по состоянию используя target-nn

        Parameters
        ----------
        inputs : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        return self.target_net(np.atleast_2d(inputs.astype('float32')))
    
    def train(self):
        '''
        Натренировать сеть

        Parameters
        ----------
        TargetNet : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
        if self.replay_memory.size < self.replay_memory.min_replay_num:
            return 0
        
        #выбираем batch_size записей
        replays = self.replay_memory.get_random_samples(self.batch_size)
        states, actions, rewards, states_next, dones = replays.values()
        
        #расчитываем target_net значения
        value_next = np.max(self._target_net_predict(states_next), axis=1)
        target_values = np.where(dones, rewards, rewards + self.gamma * value_next)

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
                        
            #logits - вектор необработанных (ненормализованных) предсказаний, 
            #которые генерирует модель классификации
            predicted_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions), axis=1)
            
            # Compute the loss value for this minibatch.            
            #loss_values = tf.math.reduce_mean(tf.square(target_values - predicted_values))
            loss_values  = self.loss_func(target_values, predicted_values)
            
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        variables = self.q_net.trainable_weights
        gradients = tape.gradient(loss_values, variables)
            
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(gradients, variables))
        self.train_step += 1

        return loss_values.numpy()
    
    def get_action(self, state, legal_actions, random_action_probality=0.0):
        
        if np.random.random() < random_action_probality:
            return np.random.choice(legal_actions)
        
        else:
            logits = self.predict(np.atleast_2d(state))
            return self.argmax(logits, legal_actions)[0]
        
    def feed(self, state, action, reward, next_state, done):
        replay = {
            'state': state,
            'action': action, 
            'reward': reward, 
            'next_state': next_state, 
            'done': done
            }
        self.replay_memory.add_replay(replay)

    def update_target_net(self):
        variables1 = self.q_net.trainable_variables
        variables2 = self.target_net.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v2.assign(v1.numpy())
    
    def remove_illegal_actions(self, probs, legal_actions):
        legal_probs = np.zeros_like(probs)
        legal_probs[legal_actions] = probs[legal_actions]
        if sum(legal_probs) == 0:
            legal_probs[legal_actions] = 1 / legal_actions.size
        else:
            legal_probs /= sum(legal_probs)
        return legal_probs
    
    def softmax(self, logits, legal_actions=None):
        
        probs = tf.keras.activations.softmax(logits).numpy()
        
        legal_probs = []
        
        if legal_actions is not None:
            legal_probs = [self.remove_illegal_actions(row, legal_actions) for row in probs]
        else:
            legal_probs = probs
            
        return legal_probs
    
    def argmax(self, logits, legal_actions=None):
        
        probs = self.softmax(logits, legal_actions)
        
        ts = tf.Variable(probs)
        return tf.math.argmax(ts, axis=1).numpy()
    
    def loss_func(self, target_values, predicted_values):
        loss_values = tf.math.reduce_mean(tf.square(target_values - predicted_values))
        return loss_values
    
    def save_model(self, path):
        shape = (1, self.num_state_params)
        bag_fix = tf.random.normal(shape)
        self.q_net.predict(bag_fix)
        self.q_net.save(path, save_format="tf")
        
    def load_model(self, path):
       self.q_net = tf.keras.models.load_model(path)
       self.target_net = tf.keras.models.load_model(path)