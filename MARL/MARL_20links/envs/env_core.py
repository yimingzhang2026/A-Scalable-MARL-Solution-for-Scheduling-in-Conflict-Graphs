import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import random
import copy
import math

non_zero_divison = 1e-7

class EnvCore(object):
    """
    # env agent def
    """

    def __init__(self,args):
        self.agent_num = self.N = args.N  
        self.obs_dim = args.N  
        self.M = args.M  
        self.action_dim = 2
        self.unstable_th = args.max_queue
        self.max_duration = args.max_duration
        self.data_rates = args.data_rates
        self.seed = args.seed
        
        self.throughput = np.zeros(self.N)
        np.random.seed(self.seed)
        # Define the adjacency matrix as a numpy array
        # A = np.array([[0, 1, 0, 1, 1, 0],
        #       [1, 0, 1, 1, 0, 0],
        #       [0, 1, 0, 1, 0, 1],
        #       [1, 1, 1, 0, 1, 0],
        #       [1, 0, 0, 1, 0, 1],
        #       [0, 0, 1, 0, 1, 0]])
        # pos = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (0.5, 0.5), 4: (1,0), 5: (1.5, 0.5)}
        # # A = np.array([[0, 1, 0, 1],
        # #       [1, 0, 1, 0],
        # #       [0, 1, 0, 1],
        # #       [1, 0, 1, 0]])
        # # pos = {0: (0, 1), 1: (0, 0), 2: (1, 0), 3: (1, 1)}
        # # Generate the graph from the adjacency matrix
        # G = nx.from_numpy_array(A)
            ### random size node graph, custom
        G = nx.random_regular_graph(3, self.N, seed = 1) # 3 here is the degree for nodes
        # Generate random positions for the vertices using the force-directed layout algorithm
        pos = nx.spring_layout(G, seed=1)
        #Plot the graph
        # nx.draw_networkx(G, with_labels=True, pos=pos)
        # plt.show()
        # Extract the vertices and edges from the graph
        self.vertices = list(G.nodes())
        self.edges = set(G.edges())

       

        self.neighbors = [[] for i in range(self.N)]
        for node in self.vertices:
            self.neighbors[node] = list(G.neighbors(node))

    def reset(self):
        """
        # The return value is a list, each list contains a shape = (self.obs_dim, ) observation data
        """
        self.t = 0
        self.create_traffic()
        self.load_traffic()  
        observation = self.get_state_traffic()
        
        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_agent_obs.append(observation[i])
        return sub_agent_obs

    def step(self, actions):
        """
        # The input of actions is a n-dimensional list, each list contains a shape = (self.action_dim, ) action data
        """
        sub_agent_obs = []
        sub_agent_reward = []
        sub_agent_done = [False for i in range(self.N)]
        sub_agent_info = []
        
        actions = [np.argmax(actions[n]) for n in range(len(actions))]
        allocated_spectrum = np.reshape(actions, (self.N, 1))
        #print("The permission is: {}".format(allocated_spectrum))
        assert allocated_spectrum.shape == (self.N,self.M) , "check joint action shape"
        
        
        for n in range(self.N):
            if len(self.packets[n]) == 0:
                allocated_spectrum[n,:] = 0
        
        #print("The transmission before collide is: {}".format(allocated_spectrum))

        #compute if different links collide, if collide throughput is 0, otherwise it is set to be 1
        for m in range(self.M):
            if np.sum(allocated_spectrum[:,m]) > 1:
                #collide may happen
                l = np.where(allocated_spectrum[:, m])[0]
                #print(l)
                combinations = list(itertools.combinations(l, 2))
                #print(combinations)
                for (i,j) in combinations:
                    if (i,j) in self.edges:
                        allocated_spectrum[i,m] = 0
                        allocated_spectrum[j,m] = 0
        #print("The transmission after collide is: {}".format(allocated_spectrum))
        for n in range(self.N):
            tmp = np.sum(allocated_spectrum[n,:])
            tmp_init = tmp
            while tmp > 0 and len(self.packets[n]) > 0:
                if tmp >= self.packets[n][0]:
                    tmp -= self.packets[n][0]
                    self.processed_packets_t[n].append(self.t - self.packets_t[n][0])
                    del(self.packets[n][0])
                    del(self.packets_t[n][0])
                else:
                    self.packets[n][0] -= tmp
                    tmp = 0
            
            self.throughput[n] = tmp_init - tmp
            if len(self.packets[n]) > self.unstable_th:
                sub_agent_done = [True for i in range(self.N)]
                
        self.load_traffic()
        rewards = self.get_reward_traffic()
        observations = self.get_state_traffic()
        
        self.t += 1
        
        if self.t >= self.max_duration:
            done = sub_agent_done = [True for i in range(self.N)]
        
        for i in range(self.agent_num):
            sub_agent_obs.append(observations[i])
            sub_agent_reward.append([rewards[i]])
            sub_agent_info.append({})

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info]
    def create_traffic(self):
        """
        # create traffic for all the agents
        """
        # if self.seed:
        #     np.random.seed(self.seed)
        self.packets = [[] for i in range(self.N)]
        self.packets_t = [[] for i in range(self.N)]
        
        self.poisson_process = [[] for i in range(self.N)]
        for i in range(self.N):
            self.poisson_process[i] = np.random.poisson(self.data_rates[i], self.max_duration + 10)
            
        self.processed_packets_t = [[] for i in range(self.N)]
    
    def load_traffic(self):
        """
        # load traffic at each time slot(part of env transition)
        """
        for n in range(self.N):
            num_incoming = int(self.poisson_process[n][self.t])
            if num_incoming != 0:
                #self.packets[n].append(self.packet_size)
                self.packets[n].append(1)
                self.packets_t[n].append(self.t)
            
            
    def get_state_traffic(self):
        """
        # each agent has the queue length information of itself and its neighbors
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
                states[i][j] = min(queues[j],self.unstable_th)
        return states
    
    def get_reward_traffic(self):     
        """
        # each agent get reward from its own utility and its neighbors(consider interaction between agents)
        """
        reward = np.zeros(self.N)
        for n in range(self.N):
            this_reward = - len(self.packets[n])
            nei = self.neighbors[n]
            for ne in nei:
                this_reward -= len(self.packets[ne]) 
            reward[n] = this_reward    
        return reward

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
            if a > 0:
                a += 1
            b = len(self.processed_packets_t[n])
            
            d.append(np.array(round(a / (b + non_zero_divison),2)))
            
        return {"queue_length": queue_length,
                "ave_delay":d,
                #'processed_delay' : process_packets_delay,
                'throughput': throughput}