# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:50:07 2022

@author: Fontanesi
"""

import numpy as np
import random
import networkx as nx
# The gym library is a collection of test problems — environments — 
# that you can use to work out your reinforcement learning algorithms. 
# These environments have a shared interface, allowing you to write general algorithms
import gym
import matplotlib.pyplot as plt



def create_nsfnet_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 12), (5, 13),
         (6, 7), (7, 10), (8, 9), (8, 11), (9, 10), (9, 12), (10, 11), (10, 13), (11, 12)])

    return Gbase

def generate_nx_graph(topology):
    """
    Generate graphs for training with the same topology.
    """
   
    G = create_nsfnet_graph()
    nx.draw(G, with_labels=True)
    plt.show()
    plt.clf()

    # Node id counter
    incId = 1
    # Put all distance weights into edge attributes.
    for i, j in G.edges():
        G.get_edge_data(i, j)['edgeId'] = incId
        G.get_edge_data(i, j)['betweenness'] = 0
        G.get_edge_data(i, j)['numsp'] = 0  # Indicates the number of shortest paths going through the link
        # We set the edges capacities to 200
        G.get_edge_data(i, j)["capacity"] = float(200)
        G.get_edge_data(i, j)['bw_allocated'] = 0
        incId = incId + 1

    return G

class Env1(gym.Env):
    """
    Description:
    The self.graph_state stores the relevant features for the GNN model

    self.graph_state[:][0] = CAPACITY
    self.graph_state[:][1] = BW_ALLOCATED
    self.graph_state[:][2] = THROUGHPUT
    self.graph_state[:][3] = LOSS
    self.graph_state[:][4] = DELAY
  """
    def __init__(self):
        self.graph = None
        self.initial_state = None
        self.source = None
        self.destination = None
        self.demand = None
        self.graph_state = None
        self.diameter = None

        # Nx Graph where the nodes have features. Betweenness is allways normalized.
        # The other features are "raw" and are being normalized before prediction
        self.first = None
        self.firstTrueSize = None
        self.second = None
        self.between_feature = None

        # Mean and standard deviation of link betweenness
        self.mu_bet = None
        self.std_bet = None

        self.max_demand = 0
        self.K = 4
        self.listofDemands = None
        self.nodes = None
        self.ordered_edges = None
        self.edgesDict = None
        self.numNodes = None
        self.numEdges = None

        self.state = None
        self.episode_over = True
        self.reward = 0
        self.allPaths = dict()
# def __init__(self):
#         self.graph = None
#         self.initial_state = None
#         self.source = None
#         self.destination = None
#         self.demand = None
#         self.graph_state = None
#         self.allPaths = dict() #crea un dizionario in python
        
        
    def data_collectionOVS(self):
        # msg = of.ofp_stats_request(body=of.ofp_flow_stats_request())
        # retrieve throughput for each link, delay and  packet loss
        # to calculate throughput and packet loss walk through all distinct paths and select both last and first switch
        # First step is to have the series of the links
        # If you have edges and node we can first compute the links
        self.diameter = nx.diameter(self.graph)
        # Iterate over all node1,node2 pairs from the graph
        for n1 in self.graph:
                for n2 in self.graph:
                    if (n1 != n2):
                        # Check if we added the element of the matrix
                        if str(n1)+':'+str(n2) not in self.allPaths:
                            self.allPaths[str(n1)+':'+str(n2)] = []
                        
                        # First we compute the shortest paths taking into account the diameter
                        # This is because large topologies might take too long to compute all shortest paths 
                        [self.allPaths[str(n1)+':'+str(n2)].append(p) for p in nx.all_simple_paths(self.graph, source=n1, target=n2, cutoff=self.diameter*2)]
                        
                        # We take all the paths from n1 to n2 and we order them according to the path length
                        self.allPaths[str(n1)+':'+str(n2)] = sorted(self.allPaths[str(n1)+':'+str(n2)], key=lambda item: (len(item), item))

    def measureDelay():
        for p in monitored_paths: #Walk through all distinct paths
				
				ip_pck = pkt.ipv4(protocol=253, #use for experiment and testing
								srcip = IPAddr(p.__hash__()),
								dstip = IPAddr("224.0.0.255"))
			
				
				pl = Payload(id(p), time.time())
					
				ip_pck.set_payload(repr(pl))
						
				eth_packet = pkt.ethernet(type=pkt.ethernet.IP_TYPE) #use something that does not interfer with regular traffic
				eth_packet.src = struct.pack("!Q", p.src)[2:] #convert dpid to EthAddr
				eth_packet.dst = struct.pack("!Q", p.dst)[2:]
				eth_packet.set_payload(ip_pck)
				
				msg = of.ofp_packet_out()
				msg.actions.append(of.ofp_action_output(port = p.first_port))
				msg.data = eth_packet.pack()
				switches[p.src].connection.send(msg)
                
                eth_packet = pkt.ethernet(type=pkt.ethernet.IP_TYPE)
				eth_packet.src = struct.pack("!Q", p.src)[2:]
				eth_packet.dst = struct.pack("!Q", p.src)[2:]
				eth_packet.set_payload(ip_pck)
				
				msg = of.ofp_packet_out()
				msg.actions.append(of.ofp_action_output(port = of.OFPP_CONTROLLER))
				msg.data = eth_packet.pack()				
				switches[p.src].connection.send(msg)
                
    def LastSwitch():
			switchRead = {}
			for dpid in switches:
				switchRead[dpid] = False
				
			for p in monitored_paths: #Walk through all distinct paths and select both last and first switch to calculate throughput and packet loss.
				if switchRead[p.dst] == False:
					switchRead[p.dst] = True
					msg = of.ofp_stats_request(body=of.ofp_flow_stats_request())
					switches[p.dst].connection.send(msg)
				
				if switchRead[p.src] == False:
					switchRead[p.src] = True
					msg = of.ofp_stats_request(body=of.ofp_flow_stats_request())
					switches[p.src].connection.send(msg)
    
    def generate_environment(self, topology, listofdemands):
            # The nx graph will only be used to convert graph from edges to nodes
            self.graph = generate_nx_graph(topology)
            # import pdb; pdb.set_trace()
            # G = generate_nx_graph(topology)
            # subax1 = plt.subplot(121)
            # nx.draw(G, with_labels=True, font_weight='bold')
            # subax2 = plt.subplot(122)
            # nx.draw_shell(G, nlist=[range(5, 10), range(5)], with_labels=True, font_weight='bold')
            
            self.listofDemands = listofdemands
    
            self.max_demand = np.amax(self.listofDemands)
    
            # Compute number of shortest paths per link. This will be used for the betweenness
            self.num_shortest_path(topology)
    
            # Compute the betweenness value for each link
            # self.mu_bet, self.std_bet = compute_link_betweenness(self.graph, self.K)
    
            self.edgesDict = dict()
    
            some_edges_1 = [tuple(sorted(edge)) for edge in self.graph.edges()]
            self.ordered_edges = sorted(some_edges_1)
    
            self.numNodes = len(self.graph.nodes())
            self.numEdges = len(self.graph.edges())
    
            self.graph_state = np.zeros((self.numEdges, 2))  #change two with the number of features
            self.between_feature = np.zeros(self.numEdges)
    
            position = 0
            for edge in self.ordered_edges:
                i = edge[0]
                j = edge[1]
                self.edgesDict[str(i)+':'+str(j)] = position
                self.edgesDict[str(j)+':'+str(i)] = position
                # betweenness = (self.graph.get_edge_data(i, j)['betweenness'] - self.mu_bet) / self.std_bet
                # self.graph.get_edge_data(i, j)['betweenness'] = betweenness
                
                ####### here insert DATA COLLECTION
                self.graph_state[position][0] = self.graph.get_edge_data(i, j)["capacity"]
                # self.between_featusre[position] = self.graph.get_edge_data(i, j)['betweenness']
                position = position + 1
    
            self.initial_state = np.copy(self.graph_state)
    
            self._first_second_between()
    
            self.firstTrueSize = len(self.first)
    
            # We create the list of nodes ids to pick randomly from them
            self.nodes = list(range(0,self.numNodes))
    
    
    
    def reset(self):
            """
            Reset environment and setup for new episode. Generate new demand and pair source, destination.
    
            Returns:
                initial state of reset environment, a new demand and a source and destination node
            """
            self.graph_state = np.copy(self.initial_state)   #set to 0??
            # self.state = np.copy(self.initial_state)
            self.demand = random.choice(self.listofDemands)
            self.source = random.choice(self.nodes)
    
            # We pick a pair of SOURCE,DESTINATION different nodes
            while True:
                self.destination = random.choice(self.nodes)
                if self.destination != self.source:
                    break
    
            return self.graph_state, self.demand, self.source, self.destination
