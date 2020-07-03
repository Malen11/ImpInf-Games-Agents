# -*- coding: utf-8 -*-`

import random
import numpy as np
import tensorflow as tf
from collections import namedtuple

from agents.rl.models.neural_network_models import SimpleNeuralNetworkModel
from pprint import pprint

class PolicyBased(object):

    def __init__(self,
                 num_state_params, 
                 num_actions, 
                 hidden_units, 
                 gamma=0.99, 
                 learning_rate=0.00005,
                 activation_func='tanh', 
                 kernel_initializer='RandomNormal',
                 output_activation_func='tanh', 
                 output_kernel_initializer='RandomNormal',
                 ):
        
        #Параметры сети
        self.num_actions = num_actions
        self.num_state_params = num_state_params
        self.hidden_units = hidden_units
        
        #параметры обучения
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)   
        self.train_step = 0
        
        #параметры предсказания
        self.gamma = gamma
        
        #две сети-обучаемая и таргет(для обучения)
        self.p_net = SimpleNeuralNetworkModel(
            num_state_params,
            hidden_units, 
            num_actions,
            activation_func=activation_func, 
            kernel_initializer=kernel_initializer,
            output_activation_func=output_activation_func, 
            output_kernel_initializer=output_kernel_initializer)
        
        #memory
        self.states = []
        self.actions = []
        self.rewards = []
    
    def predict(self, inputs, training=False):
        tf_inputs = tf.convert_to_tensor(np.atleast_2d(inputs), dtype=tf.dtypes.float32)
        return self.p_net(tf_inputs)
    
    def feed(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        
    def clear_memory(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
    
    @tf.function
    def train(self):
        
        acc_rewards = tf.convert_to_tensor(
            self._calc_acc_rewards(self.rewards), dtype=tf.dtypes.float32)
        
        # standardise the rewards
        #acc_rewards -= np.mean(acc_rewards)
        #acc_rewards /= np.std(acc_rewards)
        
        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:
            
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
                        
            #logits - вектор необработанных (ненормализованных) предсказаний, 
            #которые генерирует модель классификации
            predicted_values = self.predict(self.states)
            
            # Compute the loss value for this minibatch.            
            #loss_values = tf.math.reduce_mean(tf.square(target_values - predicted_values))
            loss_values  = tf.math.reduce_mean(
                self._loss_func(predicted_values, self.actions)*acc_rewards)
            
        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        variables = self.p_net.trainable_weights
        gradients = tape.gradient(loss_values, variables)
        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        self.optimizer.apply_gradients(zip(gradients, variables))
        
        '''
        predicted_values = self.predict(self.states)                             
        loss_values  = tf.math.reduce_mean(
                self._loss_func(predicted_values, self.actions)*acc_rewards)
        
        variables = self.p_net.trainable_weights
        self.optimizer.minimize(loss_values, variables)
        '''
        self.train_step += 1

        return loss_values
    
    def get_action(self, state, legal_actions):
        
        logits = self.predict(np.atleast_2d(state))
        probs = self.softmax(logits, legal_actions)[0]
        selected_action = np.random.choice(self.num_actions, p=probs)
        
        '''
        self.last_state = state
        self.legal_actions = legal_actions
        '''
        return selected_action
    
    def remove_illegal_actions(self, probs, legal_actions):
        legal_probs = np.zeros_like(probs)
        legal_probs[legal_actions] = probs[legal_actions]
        if sum(legal_probs) == 0:
            legal_probs[legal_actions] = 1. / len(legal_actions)
        else:
            legal_probs /= sum(legal_probs)
        return legal_probs
    
    def softmax(self, logits, legal_actions=None):
        
        probs = tf.keras.activations.softmax(logits)
        probs = probs * tf.reduce_sum(tf.one_hot(legal_actions, self.num_actions), axis=0)
        probs = probs / tf.reduce_sum(probs, axis=1)
            
        return probs.numpy()
    
    def argmax(self, logits, legal_actions=None):
        
        probs = self.softmax(logits, legal_actions)
        
        ts = tf.Variable(probs)
        return tf.math.argmax(ts, axis=1).numpy()
    
    def _loss_func(self, logits, actions):
        loss_values = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions, 
            logits=logits)
        #cost = tf.reduce_mean(acc_rewards * loss_values)
        return loss_values

    def _calc_acc_rewards(self, rewards):
        
        discounted_rewards = np.zeros_like(rewards)
        
        reward_sum = 0
        ind = 0
        for r in reversed(rewards):  # reverse buffer r
            reward_sum = r + self.gamma * reward_sum
            discounted_rewards[ind] = reward_sum
            ind += 1
        
        mean = np.mean(discounted_rewards)
        std = np.std(discounted_rewards)
        if std == 0:
            std = 1
            
        discounted_rewards = (discounted_rewards - mean) / (std)
        
        return discounted_rewards
    
    def save_model(self, path):
        
        shape = (1, self.num_state_params)
        bag_fix = tf.random.normal(shape)
        
        #self.p_net.predict(bag_fix)
        self.p_net.save(path, save_format="tf")
        
    def load_model(self, path):
        
       self.p_net = tf.keras.models.load_model(path)