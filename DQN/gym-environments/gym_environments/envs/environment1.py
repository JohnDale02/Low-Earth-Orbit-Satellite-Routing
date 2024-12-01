# Copyright (c) 2021, Paul Almasan [^1]
#
# [^1]: Universitat Politècnica de Catalunya, Computer Architecture
#     department, Barcelona, Spain. Email: felician.paul.almasan@upc.edu

import gym
import numpy as np
import networkx as nx
import random
from gym import error, spaces, utils
from random import choice
import pylab
import json 
import gc
import matplotlib.pyplot as plt

def create_geant2_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (1, 3), (1, 6), (1, 9), (2, 3), (2, 4), (3, 6), (4, 7), (5, 3),
         (5, 8), (6, 9), (6, 8), (7, 11), (7, 8), (8, 11), (8, 20), (8, 17), (8, 18), (8, 12),
         (9, 10), (9, 13), (9, 12), (10, 13), (11, 20), (11, 14), (12, 13), (12,19), (12,21),
         (14, 15), (15, 16), (16, 17), (17,18), (18,21), (19, 23), (21,22), (22, 23)])

    return Gbase

def create_nsfnet_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 12), (5, 13),
         (6, 7), (7, 10), (8, 9), (8, 11), (9, 10), (9, 12), (10, 11), (10, 13), (11, 12)])

    return Gbase

def create_small_top():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])
    Gbase.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 7), (2, 5), (3, 8), (3, 4), (4, 5), (4, 6), (5, 0),
         (6, 7), (6, 8), (7, 8), (8, 0), (8, 6), (3, 2), (5, 3)])

    return Gbase

def create_gbn_graph():
    Gbase = nx.Graph()
    Gbase.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16])
    Gbase.add_edges_from(
        [(0, 2), (0, 8), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4), (3, 9), (4, 8), (4, 10), (4, 9),
         (5, 6), (5, 8), (6, 7), (7, 8), (7, 10), (9, 10), (9, 12), (10, 11), (10, 12), (11, 13),
         (12, 14), (12, 16), (13, 14), (14, 15), (15, 16)])

    return Gbase


def create_iridium_graph():
    # Initialize graph
    G = nx.Graph()
    
    # Parameters
    num_planes = 2 # Number of orbital planes (6)
    sats_per_plane = 5  # Number of satellites per orbital plane (11)
    total_sats = num_planes * sats_per_plane
    edges = []
    
    # Add nodes for each satellite
    G.add_nodes_from(range(total_sats))
    
    # Create edges
    for plane in range(num_planes):
        for sat in range(sats_per_plane):
            # Current satellite index
            current_sat = plane * sats_per_plane + sat
            
            # Intra-plane links
            next_sat_in_plane = plane * sats_per_plane + (sat + 1) % sats_per_plane
            edges.append((current_sat, next_sat_in_plane))
           
            # Inter-plane links (to adjacent planes)
            next_plane = (plane + 1) % num_planes
            prev_plane = (plane - 1) % num_planes  # Wrap-around for the previous plane
            corresponding_next = next_plane * sats_per_plane + sat
            corresponding_prev = prev_plane * sats_per_plane + sat

            edges.append((current_sat, corresponding_next))
            edges.append((current_sat, corresponding_prev))
    
    print(edges)
    G.add_edges_from(edges)

    return G


def generate_nx_graph(topology):
    """
    Generate graphs for training with the same topology.
    """
    if topology == 0:
        G = create_nsfnet_graph()
    elif topology == 1:
        G = create_geant2_graph()
    elif topology == 2:
        G = create_small_top()
    elif topology == 3:
        G = create_gbn_graph()
    else:  # == 4 
        G = create_iridium_graph()  # OUR CUSTOM GRAPH FOR THE IRIDIUM SATELLITE CONSTELATION

    # nx.draw(G, with_labels=True)
    # plt.show()
    # plt.clf()

    # Node id counter
    incId = 1
    # Put all distance weights into edge attributes.
    for i, j in G.edges():
        G.get_edge_data(i, j)['edgeId'] = incId
        G.get_edge_data(i, j)['betweenness'] = 0
        G.get_edge_data(i, j)['numsp'] = 0  # Indicates the number of shortest paths going through the link
        # We set the edges capacities to 200
        G.get_edge_data(i, j)["capacity"] = float(200)     # BANDWIDTH FOR ALL EDGES SET TO 200 
        G.get_edge_data(i, j)['bw_allocated'] = 0
        incId = incId + 1

    print("RETURNING G")
    return G


def compute_link_betweenness(g, k):
    n = len(g.nodes())
    betw = []
    for i, j in g.edges():
        # we add a very small number to avoid division by zero
        b_link = g.get_edge_data(i, j)['numsp'] / ((2.0 * n * (n - 1) * k) + 0.00000001)
        g.get_edge_data(i, j)['betweenness'] = b_link
        betw.append(b_link)

    mu_bet = np.mean(betw)
    std_bet = np.std(betw)
    return mu_bet, std_bet

class Env1(gym.Env):
    """
    Description:
    The self.graph_state stores the relevant features for the GNN model

    self.graph_state[:][0] = CAPACITY
    self.graph_state[:][1] = BW_ALLOCATED
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
        self.subset_nodes = []

        self.state = None
        self.episode_over = True
        self.reward = 0
        self.bw = 0
        self.delay = 0 
        self.allPaths = dict()



    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def num_shortest_path(self, topology):
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

                    path = 0
                    while path < self.K and path < len(self.allPaths[str(n1)+':'+str(n2)]):
                        currentPath = self.allPaths[str(n1)+':'+str(n2)][path]
                        i = 0
                        j = 1

                        # Iterate over pairs of nodes increase the number of sp
                        while (j < len(currentPath)):
                            self.graph.get_edge_data(currentPath[i], currentPath[j])['numsp'] = \
                                self.graph.get_edge_data(currentPath[i], currentPath[j])['numsp'] + 1
                            i = i + 1
                            j = j + 1

                        path = path + 1

                    # Remove paths not needed
                    del self.allPaths[str(n1)+':'+str(n2)][path:len(self.allPaths[str(n1)+':'+str(n2)])]
                    gc.collect()

    def _first_second_between(self):
        self.first = list()
        self.second = list()

        # For each edge we iterate over all neighbour edges
        for i, j in self.ordered_edges:
            neighbour_edges = self.graph.edges(i)

            for m, n in neighbour_edges:
                if ((i != m or j != n) and (i != n or j != m)):
                    self.first.append(self.edgesDict[str(i) +':'+ str(j)])
                    self.second.append(self.edgesDict[str(m) +':'+ str(n)])

            neighbour_edges = self.graph.edges(j)
            for m, n in neighbour_edges:
                if ((i != m or j != n) and (i != n or j != m)):
                    self.first.append(self.edgesDict[str(i) +':'+ str(j)])
                    self.second.append(self.edgesDict[str(m) +':'+ str(n)])


    def generate_environment(self, topology, listofdemands):
        # The nx graph will only be used to convert graph from edges to nodes
        self.graph = generate_nx_graph(topology)

        self.listofDemands = listofdemands

        self.max_demand = np.amax(self.listofDemands)

        # Compute number of shortest paths per link. This will be used for the betweenness
        self.num_shortest_path(topology)

        # Compute the betweenness value for each link
        self.mu_bet, self.std_bet = compute_link_betweenness(self.graph, self.K)

        self.edgesDict = dict()

        some_edges_1 = [tuple(sorted(edge)) for edge in self.graph.edges()]
        self.ordered_edges = sorted(some_edges_1)

        self.numNodes = len(self.graph.nodes())
        self.numEdges = len(self.graph.edges())

        self.graph_state = np.zeros((self.numEdges, 2))
        self.between_feature = np.zeros(self.numEdges)

        position = 0
        print("DOWN HERE")
        for edge in self.ordered_edges:
            # adding all the edge state (betweeness, capacity to graph_state)
            # creating an itital state
            i = edge[0]
            j = edge[1]
            self.edgesDict[str(i)+':'+str(j)] = position
            self.edgesDict[str(j)+':'+str(i)] = position
            betweenness = (self.graph.get_edge_data(i, j)['betweenness'] - self.mu_bet) / self.std_bet
            self.graph.get_edge_data(i, j)['betweenness'] = betweenness
            self.graph_state[position][0] = self.graph.get_edge_data(i, j)["capacity"]
            self.between_feature[position] = self.graph.get_edge_data(i, j)['betweenness']
            position = position + 1


        self.initial_state = np.copy(self.graph_state)

        self._first_second_between()

        self.firstTrueSize = len(self.first)

        # We create the list of nodes ids to pick randomly from them
        # FOR OUR EXPERIMENT WE WILL PICK 6 RANDOMLY
        self.nodes = list(range(0,self.numNodes))

    def utility_function(self, bw, delay, alpha=0.9, lambda_weight=1.0):
        U_bw = (bw ** (1 - alpha)) / (1 - alpha)
        U_delay = (delay ** (1 - alpha)) / (1 - alpha)
        return U_bw - lambda_weight * U_delay

    def make_step(self, state, action, demand, source, destination):
        """
        Perform a step in the environment by selecting a path and allocating the required bandwidth.

        Args:
            state: Current state of the graph.
            action: The index of the selected path.
            demand: Bandwidth required for the transmission.
            source: Source node for the transmission.
            destination: Destination node for the transmission.

        Returns:
            Updated graph state, reward for the step, whether the episode is over, 
            new demand, new source, and new destination.
        """
        self.graph_state = np.copy(state)
        self.episode_over = True
        self.reward = 0

        i = 0
        j = 1
        currentPath = self.allPaths[str(source) +':'+ str(destination)][action]

        # Once we pick the action, we decrease the total edge capacity from the edges
        # from the allocated path (action path)
        while (j < len(currentPath)):
            self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] -= demand
            if self.graph_state[self.edgesDict[str(currentPath[i]) + ':' + str(currentPath[j])]][0] < 0:
                # FINISH IF LINKS CAPACITY <0
                return self.graph_state, self.reward, self.episode_over, self.demand, self.source, self.destination, self.bw, self.delay
            i = i + 1
            j = j + 1

        # Leave the bw_allocated back to 0 --> send all requests 
        self.graph_state[:,1] = 0

        # Reward is the allocated demand or 0 otherwise (end of episode)
        # We normalize the demand to don't have extremely large values
         # Reward calculation based on the utility function
        self.bw = demand / self.max_demand  # Normalize bandwidth 
        self.delay = len(currentPath) / (self.numNodes // 2)  # Normalize delay
       
        self.reward = self.utility_function(self.bw, self.delay)
        self.episode_over = False

        self.demand = random.choice(self.listofDemands)

        if self.subset_nodes != []:
            self.source = random.choice(self.subset_nodes)

            # We pick a pair of SOURCE,DESTINATION different nodes
            while True:
                self.destination = random.choice(self.subset_nodes)
                if self.destination != self.source:
                    break
        else:
            self.demand = 0
            self.source = 0
            self.destination = 0

        return self.graph_state, self.reward, self.episode_over, self.demand, self.source, self.destination, self.bw, self.delay

    def reset(self):
        """
        Reset environment and setup for new episode. Generate new demand and pair source, destination.

        Returns:
            initial state of reset environment, a new demand and a source and destination node
        """
        self.bw = 0
        self.delay = 0
        self.graph_state = np.copy(self.initial_state)
        self.subset_nodes = random.sample(self.nodes, 6)

        self.source = random.choice(self.subset_nodes)

        # We pick a pair of SOURCE,DESTINATION different nodes
        while True:
            self.destination = random.choice(self.subset_nodes)
            if self.destination != self.source:
                break

        return self.graph_state, self.demand, self.source, self.destination
    
    def eval_sap_reset(self, demand, source, destination):
        """
        Reset environment and setup for new episode. This function is used in the "evaluate_DQN.py" script.
        """
        self.bw = 0
        self.delay = 0
        self.graph_state = np.copy(self.initial_state)

        self.demand = demand
        self.source = source
        self.destination = destination

        return self.graph_state