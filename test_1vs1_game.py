# -*- coding: utf-8 -*-
import tensorflow as tf
#для исправления бага с совместимостью
#tf.compat.v1.enable_eager_execution()
from tensorflow.python.client import device_lib

import os
import math
import sys
import argparse
from pprint import pprint

import rlcard
from rlcard.agents import RandomAgent
from rlcard.utils import set_global_seed, tournament
from rlcard.utils import Logger

from agents.DDQNAgent import DDQNAgent
from agents.A2CAgent import A2CAgent
from agents.A2CQPGAgent import A2CQPGAgent
from agents.A2CLSTMAgent import A2CLSTMAgent
from agents.A2CLSTMQPGAgent import A2CLSTMQPGAgent
from agents.testAgents import FoldAgent

def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-enn', '--env_name', default = 'no-limit-holdem')
    parser.add_argument('-evn', '--evaluate_num', default = 100000, type=int)
    parser.add_argument('-rs', '--random_seed', default = 0, type=int)
    
    parser.add_argument('-at0', '--agent_type0', default = None)
    parser.add_argument('-at1', '--agent_type1', default = None)
    
    parser.add_argument('-lm0', '--load_model0', default = None)
    parser.add_argument('-lm1', '--load_model1', default = None)
    
    return  parser

def getAgent(agent_type, env):
    
    agent = None
    
    if agent_type == 'RandomAgent':
        agent = RandomAgent(action_num=env.action_num)
        
    elif agent_type == 'DDQNAgent':
        agent = DDQNAgent(
                     action_num=env.action_num,
                     state_shape=env.state_shape,)
        
    elif agent_type == 'A2CLSTMAgent':
        agent = A2CLSTMAgent(
                     action_num=env.action_num,
                     state_shape=env.state_shape,
                     trainble=False,
                     
                     discount_factor=0.95,
                
                     critic_lstm_layers=[1,512],
                     critic_mlp_layers=[3,512],
                     critic_activation_func='tanh', 
                     critic_kernel_initializer='glorot_uniform',
                     critic_learning_rate=0.001,
                     critic_bacth_size=128,
                     
                     actor_lstm_layers=[1,512],
                     actor_mlp_layers=[3,512],
                     actor_activation_func='tanh', 
                     actor_kernel_initializer='glorot_uniform', 
                     actor_learning_rate=0.0001,
                     actor_bacth_size=512,
                     
                     entropy_coef=0.5,
                     
                     max_grad_norm = 1,) 
    
    elif agent_type == 'A2CQPGAgent':
        agent = A2CQPGAgent(
                     action_num=env.action_num,
                     state_shape=env.state_shape,
                     trainble=False,
                     
                     discount_factor=0.95,
                
                     critic_mlp_layers=[4,512],
                     critic_activation_func='tanh', 
                     critic_kernel_initializer='glorot_uniform',
                     critic_learning_rate=0.001,
                     critic_bacth_size=128,
                     
                     actor_mlp_layers=[4,512],
                     actor_activation_func='tanh', 
                     actor_kernel_initializer='glorot_uniform', 
                     actor_learning_rate=0.0001,
                     actor_bacth_size=512,
                     
                     entropy_coef=1,
                     
                     max_grad_norm = 1,)
        
    elif agent_type == 'A2CLSTMQPGAgent':
        agent = A2CLSTMQPGAgent(
                     action_num=env.action_num,
                     state_shape=env.state_shape,
                     trainable=False,
                     
                     discount_factor=0.95,
                
                     critic_lstm_layers=[1,512],
                     critic_mlp_layers=[3,512],
                     critic_activation_func='tanh', 
                     critic_kernel_initializer='glorot_uniform',
                     critic_learning_rate=0.001,
                     critic_bacth_size=128,
                     
                     actor_lstm_layers=[1,512],
                     actor_mlp_layers=[3,512],
                     actor_activation_func='tanh', 
                     actor_kernel_initializer='glorot_uniform', 
                     actor_learning_rate=0.0001,
                     actor_bacth_size=512,
                     
                     entropy_coef=0.5,
                     
                     max_grad_norm = 1,) 
        
    elif agent_type == 'A2CAgent':
        agent = A2CAgent(
                     action_num=env.action_num,
                     state_shape=env.state_shape,
                     
                     discount_factor=0.95,
                
                     critic_mlp_layers=[4,512],
                     critic_activation_func='tanh', 
                     critic_kernel_initializer='glorot_uniform',
                     critic_learning_rate=0.001,
                     critic_bacth_size=128,
                     
                     actor_mlp_layers=[4,512],
                     actor_activation_func='tanh', 
                     actor_kernel_initializer='glorot_uniform', 
                     actor_learning_rate=0.0001,
                     actor_bacth_size=512,
                     
                     entropy_coef=1,
                     
                     max_grad_norm = 1,)
    else:
        raise ValueError(str(agent_type)+' type not exist')
    
    return  agent

def main():
    
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    
    #random seed
    random_seed = namespace.random_seed
    #names
    env_name = namespace.env_name
    # Set the iterations numbers and how frequently we evaluate/save plot
    evaluate_num = namespace.evaluate_num
    
    # Make environment
    eval_env = rlcard.make(env_name, config={'seed': random_seed})
    
    # Set a global seed
    set_global_seed(random_seed)
    
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    
    # Set up the agents
    agent0 = getAgent(namespace.agent_type0, eval_env) 
    agent1 = getAgent(namespace.agent_type1, eval_env)
    
    
    if namespace.load_model0 is not None:
        agent0.load_model(namespace.load_model0)
    if namespace.load_model1 is not None:
        agent1.load_model(namespace.load_model1)
    
  
    eval_env.set_agents([agent0, agent1])

    # Evaluate the performance. Play with random agents.
    rewards = tournament(eval_env, evaluate_num)
    print('Average reward for agent0 against agent1: ', rewards[0])
        
if __name__ == '__main__':
    main()