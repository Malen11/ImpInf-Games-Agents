# -*- coding: utf-8 -*-
import tensorflow as tf
#для исправления бага с совместимостью
#tf.compat.v1.enable_eager_execution()
from tensorflow.python.client import device_lib

import os
import math
import sys
import argparse

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
    parser.add_argument('-tn', '--test_name', default = 'test_r')
    parser.add_argument('-ee', '--evaluate_every', default = 100000, type=int)
    parser.add_argument('-epn', '--episode_num', default = 1000000, type=int)
    parser.add_argument('-evn', '--evaluate_num', default = 10000, type=int)
    parser.add_argument('-te', '--train_every', default = 2500, type=int)
    parser.add_argument('-se', '--save_every', default = 100000, type=int)
    parser.add_argument('-rs', '--random_seed', default = 0, type=int)
    
    parser.add_argument('-lm', '--load_model', default = None)
    
    return  parser
    
def main():
    
    parser = createParser()
    namespace = parser.parse_args(sys.argv[1:])
    
    #random seed
    random_seed = namespace.random_seed
    #names
    env_name = namespace.env_name
    env_num = 1
    test_name = namespace.test_name
    dir_name = str(env_name)+'_a2c_'+str(test_name)+str(random_seed)
    # Set the iterations numbers and how frequently we evaluate/save plot
    evaluate_every = namespace.evaluate_every
    evaluate_num = namespace.evaluate_num
    episode_num = namespace.episode_num
    # Train the agent every X steps
    train_every = namespace.train_every
    save_every = namespace.save_every
    
    
    # Make environment
    env_rand = rlcard.make(env_name, config={'seed': random_seed})
    eval_env = rlcard.make(env_name, config={'seed': random_seed})
        
    # The paths for saving the logs and learning curves
    log_dir = './experiments/rl/'+dir_name+'_result'
    
    # Save model
    save_dir = 'models/rl/'+dir_name+'_result'
    
    # Set a global seed
    set_global_seed(random_seed)
    
    # Initialize a global step
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Set up the agents
    
    agent_rand = RandomAgent(action_num=eval_env.action_num)    
    
    agent_test = A2CQPGAgent(
                     action_num=eval_env.action_num,
                     state_shape=eval_env.state_shape,
                     
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
                     entropy_decoy=math.pow(0.05/1, 1.0/(episode_num//train_every)),
                     
                     max_grad_norm = 1,)
    
    if namespace.load_model is not None:
        agent_test.load_model(namespace.load_model)
    
    env_rand.set_agents([agent_test, agent_rand])
    
    eval_env.set_agents([agent_test, agent_rand])

    # Init a Logger to plot the learning curve
    logger = Logger(log_dir+'/'+test_name)
    
    envs = [env_rand, 
            ]
    
    env_num = len(envs)
    for episode in range(episode_num // env_num):

        # Generate data from the
        for env in envs:
            trajectories, _ = env.run(is_training=True)

            # Feed transitions into agent memory, and train the agent
            for ts in trajectories[0]:
                agent_test.feed(ts)
            
        if episode % (train_every // env_num) == 0:
            agent_test.train()
        
        if episode % (save_every // env_num) == 0 :
            # Save model
            if not os.path.exists(save_dir+'/'+test_name+str(episode*env_num)):
                os.makedirs(save_dir+'/'+test_name+str(episode*env_num))
            agent_test.save_model(save_dir+'/'+test_name+str(episode*env_num))
            
        # Evaluate the performance. Play with random agents.
        if episode % (evaluate_every // env_num) == 0:
            print('episode: ', episode*env_num)
            logger.log_performance(episode*env_num, tournament(eval_env, evaluate_num)[0])


    # Close files in the logger
    logger.close_files()

    # Plot the learning curve
    logger.plot(dir_name)
         
    # Save model
    if not os.path.exists(save_dir+'/'+test_name+str(episode_num)):
        os.makedirs(save_dir+'/'+test_name+str(episode_num))
    agent_test.save_model(save_dir+'/'+test_name+str(episode_num))
    
if __name__ == '__main__':
    main()