# -*- coding: utf-8 -*-
"""
Created on Tue May 30 14:38:55 2023

@author: zyimi
"""
import argparse
from env_s import env

def to_one_hot(number, num_actions):
    one_hot = [0] * num_actions
    one_hot[number] = 1
    return one_hot

def get_largest_indices(lst):
    max_val = max(lst)
    indices = [i for i, val in enumerate(lst) if val == max_val]
    return indices

def simple_algorithm(agent_index, partial_obs):
    action = 0
    longest_q = max(partial_obs)
    indices = [i for i, val in enumerate(partial_obs) if val == longest_q]
    if agent_index in indices:
        if len(indices) == 1:
            action = 1
        elif np.random.random() < 0.5:
            action = 1
    return action
    
    
def parse_args(parser):
    parser.add_argument('--scenario', type=str,
                     default='conflict_graph', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int,
                        default=20, help="number of players")
    
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="random seed",
    )
    parser.add_argument(
        "--N",
        type=int,
        default=20,
        help="number of links in marl",
    )
    parser.add_argument(
        "--M",
        type=int,
        default=1,
        help="number of subbands",
    )
    parser.add_argument(
        "--max_duration",
        type=int,
        default=1000,
        help="max duration of env in one episode, should be equal the args.episode_length parameter",
    )
    parser.add_argument(
        "--data_rates",
        type=list,
        default=[0.15] * 20,
        help="the arrival rate of packets for agents, the length should be equal to the number of agents",
    )
    parser.add_argument(
        "--max_queue",
        type=int,
        default=15,
        help="the threshold of queue length to be considered as unstable",
    )
                        
    all_args = parser.parse_args()  # Parse the arguments

    return all_args

parser = argparse.ArgumentParser()  # Create an empty parser
all_args = parse_args(parser)  # Parse the arguments using the parser

env = env(all_args)
#if test, set seed to make sure each time the env is the same
obs = env.reset()
#print(env.poisson_process[1][:15])
import numpy as np
info_ep_list = []

for i in range(2):
    done = False
    for step in range(env.max_duration):
        #print(f"Step {step + 1}")
        
        actions = [0 for i in range(env.N)]
        actions_one_hot = [[] for i in range(env.N)]
        
        for n in range(env.N):
            actions[n] = simple_algorithm(n, obs[n])
            actions_one_hot[n] = to_one_hot(actions[n],env.action_dim)
        #print(actions)
        #print('obs is{}'.format(obs))
        obs, reward, ternimated, info = env.step(actions_one_hot)
        
        done = (any(ternimated) == True)

        if done or step == env.max_duration - 1:
            #print("episode_end!", "reward=", reward)
            info_ep = env.get_info_ep()
            info_ep_list.append(info_ep)
            print(info_ep)
            break

    
def save_data(file_path, data):
    np.savez(file_path, *data)
    print('data saved in {}'.format(file_path))
    
data = info_ep_list  # Get the data dictionary
save_data('./benchmark_ch{}_{}nodes_dr{}.npz'.format(all_args.M,all_args.N,all_args.data_rates[0]), data) 