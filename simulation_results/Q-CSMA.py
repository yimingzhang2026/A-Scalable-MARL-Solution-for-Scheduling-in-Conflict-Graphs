# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:52:19 2023

@author: zyimi
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import random
import copy
import math
import argparse
import pandas as pd


non_zero_divison = 1e-9
class conflict_graph(object):
    """
  
    # env def
    """
    def __init__(self, N, M, max_duration, data_rates, seed):
        self.N = N
        self.M = 1
        self.action_dim = 2
        self.max_duration = max_duration
        self.data_rates = data_rates
        self.seed = seed
        self.throughput = np.zeros(self.N)
        np.random.seed(self.seed)
        if self.N != 24: #reprocdue Q-CSMA results in paper
          ### 6 node graph, conflict graph in book
          if self.N == 6:
    
              # Define the adjacency matrix as a numpy array
              A = np.array([[0, 1, 0, 1, 1, 0],
                    [1, 0, 1, 1, 0, 0],
                    [0, 1, 0, 1, 0, 1],
                    [1, 1, 1, 0, 1, 0],
                    [1, 0, 0, 1, 0, 1],
                    [0, 0, 1, 0, 1, 0]])
              pos = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (0.5, 0.5), 4: (1,0), 5: (1.5, 0.5)}
              G = nx.from_numpy_array(A)
              labels = {i: i for i in range(self.N)}
              nx.draw_networkx(G, with_labels=True,pos=pos)
              #nx.draw_networkx_labels(G, pos=nx.spring_layout(G), labels=labels, font_size=10, font_color='r')
              plt.show()
          else:
              ### random size node graph, custom
              G = nx.random_regular_graph(3, self.N, seed = 1) # 3 here is the degree for nodes
              # Generate random positions for the vertices using the force-directed layout algorithm
              pos = nx.spring_layout(G, seed=1)
              nx.draw_networkx(G, with_labels=True, pos=pos)
              nx.draw_networkx_edges(G, pos,
              edgelist=G.edges, edge_color='black',
              style='solid',alpha=1, arrows=True, width=1,
              arrowsize=14, arrowstyle='<->',
              node_size=1500,
              )
              plt.show()
    
          self.vertices = list(G.nodes())
          self.edges = set(G.edges())
          self.neighbors = [[] for i in range(self.N)]
          for node in self.vertices:
              self.neighbors[node] = list(G.neighbors(node))
        else:
          G = nx.Graph()
          for i in range(4):
              for j in range(4):
                  G.add_node((i, j))
          label = 0
          for i in range(4):
              for j in range(3):
                  G.add_edge((i, j), (i, j+1), label=label)
                  label += 1
              if i < 3:
                  for j in range(4):
                      G.add_edge((i, j), (i+1, j), label=label)
                      label += 1
    
          plt.figure(figsize=(8,8))
          pos = dict((node, node) for node in G.nodes())
          nx.draw_networkx(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=15, font_weight="bold")
          edge_labels = nx.get_edge_attributes(G, 'label')
          nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
          plt.title("4x4 Grid Graph")
          plt.show()
          self.neighbors = self.compute_edge_neighbors(G)
          #print(self.neighbors)
          neighbored_edge_set = set()
    
          # Iterate over each edge and its neighbors
          for edge_label, neighbors in enumerate(self.neighbors):
              for neighbor in neighbors:
                  # Sort the pair before adding to ensure uniqueness
                  pair = tuple(sorted([edge_label, neighbor]))
                  neighbored_edge_set.add(pair)
    
          self.edges = neighbored_edge_set
          
    def reset(self):
        self.t = 0
        self.create_traffic()
        self.load_traffic()
        observation = self.get_queue()
        obs = []
        for i in range(self.N):
            obs.append(observation[i])
        return obs
    
    def step(self, actions):
        obs = []
        for n in range(self.N):
            tmp = actions[n]
            if tmp > 0 and len(self.packets[n]) > 0:
                self.processed_packets_t[n].append(self.t - self.packets_t[n][0])
                del(self.packets[n][0])
                del(self.packets_t[n][0])
            
            
        self.load_traffic()
        observations = self.get_queue()
    
        self.t += 1
        for i in range(self.N):
            obs.append(observations[i])
        return obs
    
    def create_traffic(self):
        """
        # create traffic for all the links
        """
        np.random.seed(self.seed)
        self.packets = [[] for i in range(self.N)]
        self.packets_t = [[] for i in range(self.N)]
        self.poisson_process = [[] for i in range(self.N)]
        for i in range(self.N):
            # self.poisson_process[i] = np.random.poisson(self.data_rates[i], self.max_duration + 10)
            self.poisson_process[i] = np.random.choice([1, 0], size=self.max_duration + 10, p=[self.data_rates[i], 1-self.data_rates[i]])
    
        self.processed_packets_t = [[] for i in range(self.N)]
    
    def load_traffic(self):
        """
        # load traffic at each time slot(part of env transition)
        """
        for n in range(self.N):
            num_incoming = int(self.poisson_process[n][self.t])
            while num_incoming != 0:
                self.packets[n].append(1)
                self.packets_t[n].append(self.t)
                num_incoming -= 1
    
    
    def get_queue(self):
        """
        # each link has the queue length information of itself and its neighbors
        """
        states = [[] for i in range(self.N)]
        queues = [0 for i in range(self.N)]
        for i in range(self.N):
            if len(self.packets[i]) > 0:
                queues[i] = len(self.packets[i])
        for i in range(self.N):
            nei = copy.copy(self.neighbors[i])
            nei.append(i)
            states[i] = np.zeros_like(queues)
            for j in nei:
                states[i][j] = queues[j]
        return states
    
    
    def get_info_ep(self):
        """
        # compute the average delay and througput, and record the queue length at the end of each episode (get the custom metric)
        """
        queue_length = [len(self.packets[i]) for i in range(self.N)]
        #process_packets_delay = [self.processed_packets_t[i] for i in range(self.N)]
        throughput = [len(self.processed_packets_t[i]) for i in range(self.N)]
        d = []
        for n in range(self.N):
            a = np.sum(self.processed_packets_t[n])
            b = len(self.processed_packets_t[n])
            d.append(np.array(round(a / (b + non_zero_divison),2)))
    
        return {"queue_length": queue_length,
                "ave_delay":d,
                #'processed_delay' : process_packets_delay,
                'throughput': throughput}
    
    def compute_edge_neighbors(self, G):
      edge_neighbors = {}
      for edge in G.edges():
          edge_label = G.get_edge_data(*edge)['label']
          edge_neighbors[edge_label] = []
          
          # For each node connected by the edge, find all connecting edges (neighbors)
          for node in edge:
              for neighbor_edge in G.edges(node):
                  neighbor_edge_label = G.get_edge_data(*neighbor_edge)['label']
                  if neighbor_edge_label != edge_label:  # Exclude the original edge
                      edge_neighbors[edge_label].append(neighbor_edge_label)
      
      # Sort neighbor lists and convert to a list of lists
      edge_neighbors_list = [sorted(edge_neighbors[label]) for label in sorted(edge_neighbors.keys())]
      return edge_neighbors_list
  
    
    
def simple_algorithm(link_index, obs):
    """
    # decide action for each link, 0 : no transimission, 1: transmission
    """
    action = 0
    longest_q = max(obs)
    indices = [i for i, val in enumerate(obs) if val == longest_q]
    if link_index in indices:
        if len(indices) == 1:
            action = 1
        elif np.random.random() < 0.5:
            action = 1
    return action



def q_csma_modified(env,previous_actions,W):
    actions = copy.copy(previous_actions)
    back_off_times = np.zeros_like(actions)
    neighbors = copy.copy(env.neighbors)
    decision_set = set()

    # Step 1: Calculate wl for each link
    queue_length = [len(env.packets[i]) for i in range(env.N)]
    #w = [math.log(1 + q) for q in queue_length]
    w = [math.log(10 * q) if q!=0 else -float('inf') for q in queue_length]
    # Step 2: Choose random backoff time in [0, W-1]
    for i in range(len(actions)):
      #back_off_times[i] = random.randint(0, W - 1)
      back_off_times[i] = random.randint(0, W - 1) if queue_length[i] > 0 else W + 10
     
    # Step 3: Check for control message collisions
    intents = {i : set() for i in range(W + 1)}
    
    for mini_slot in range(W):
        # Identify links transmitting an INTENT message in this mini_slot.
        transmitting_links = [i for i, back_off_time in enumerate(back_off_times) if back_off_time == mini_slot]
        intents[mini_slot + 1] = copy.copy(intents[mini_slot])
        # Check for collisions in this mini_slot and update intents and decision_set.
        for link in transmitting_links:
            if any(neighbor in intents[mini_slot] for neighbor in neighbors[link]):
                continue  # If collision, skip this link.
            intents[mini_slot + 1].add(link)  # Add link to intents.
    intent_transmitting_links = intents[W]
    decision_set = []
    #print(decision_set)
    for link in intent_transmitting_links:
        # Check if any of its neighbors are also in the nodes list
        #has_neighbors_in_list = any(neighbor in intent_transmitting_links for neighbor in neighbors[link])
        has_neighbors_in_list = any(neighbor in decision_set for neighbor in neighbors[link])
        
        # If the node does not have any neighbors in the list, add it to the remaining_nodes list
        if not has_neighbors_in_list:
            decision_set.append(link)
        

    #print(decision_set)

    # Step 4: Process links in decision set
    for link in decision_set:
        ewl = math.exp(w[link])
        p = ewl / (1 + ewl)
        # If no neighbors in decision_set were active in the previous slot, decide state with probability p.
        if all(previous_actions[neighbor] == 0 for neighbor in env.neighbors[link]):
            actions[link] = 1 if random.random() < p else 0
        else:
            actions[link] = 0  
    
    for i in range(len(actions)):
        if queue_length[i] == 0:
            actions[i] = 0 
    return actions

seed = 1
N = 20
M = 1
W = 2 * N
data_rate = 0.15
max_duration = 1000
# dr_M = np.array([[1, 3, 8, 10, 15, 17, 22, 24],
#     [4, 5, 6, 7, 18, 19, 20, 21],
#      [1, 3, 9, 11, 14, 16, 22, 24],
#      [2, 4, 7, 12, 13, 18, 21, 23]  
# ])

# dr_M -= 1
# print(dr_M)
# dr_matrix = np.zeros((4, 24))
# for i in range(4):
#     dr_matrix[i, dr_M[i]] = 1
# print(dr_matrix)
# c = np.array([0.2, 0.3, 0.2, 0.3])
# # print(dr_matrix.T * c)
# max_data_rates = np.sum(dr_matrix.T * c, axis=1).tolist()
# print(max_data_rates)

data_rates = [data_rate] * N
env = conflict_graph(N, M, max_duration, data_rates, seed)
W = 2 * N
info_ep_list = []
# traffic_density = np.arange(0.3, 0.9, 0.1).tolist()
# for p in traffic_density: 
for i in range(2):
    obs = env.reset()
    previous_actions = [0 for i in range(env.N)]
    print(f'W = {W}')
    print(data_rates)
    for step in range(env.max_duration):
        actions = q_csma_modified(env,previous_actions,W) #you can uncomment print in step 8 in qcsma function to see the behaviour of links
        obs = env.step(actions)
        previous_actions = actions
        if step == env.max_duration - 1:
            info_ep_qcsma = env.get_info_ep()
            info_ep_list.append(info_ep_qcsma)
            print(info_ep_qcsma)
            break
  
def save_data(file_path, data):
    np.savez(file_path, *data)
    print('data saved in {}'.format(file_path))
    
data = info_ep_list  # Get the data dictionary
save_data('./QCSMA_ch{}_{}nodes_dr{}.npz'.format(M,N,data_rates[0]), data) 
