'''
Author = João Pedro Campos
some functions are taken or adapted from the work by Itallo Machado https://github.com/ItalloMachado/Mestrado
'''
import random
#import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import csv
from pgmpy.readwrite import BIFReader
import time
import statistics
import bnlearn as bn
from pgmpy.metrics import structure_score
from pgmpy.models import BayesianNetwork
import gc
from utils import *
from loaders import *
import matplotlib.pyplot as plt

class Individual():
    def __init__(self, genes, nodes):
        self.genes = genes
        self.nodes = nodes
        self.bic = None

    def init_random(self, num_nodes, sparsity=0.5):
        G = nx.gnp_random_graph(num_nodes, sparsity, directed=True)
        genes = nx.adjacency_matrix(G).todense().flatten().tolist()
        self.genes = genes

    def init_from_genes(self, genes):
        self.genes = genes

    def bit_flip_mutation(self, feasible_only=False):
        #select random bit
        bit = random.randint(0, len(self.genes)-1)
        #flip bit
        self.genes[bit] = 1 - self.genes[bit]

        if feasible_only:
            if not self.is_dag():
                self.repair_dag()
            if not self.is_connected():
                self.repair_connectivity()
    
    def uniform_mutation(self, prob, feasible_only=False):
        for i in range(len(self.genes)):
            if random.random() < prob:
                self.genes[i] = 1 - self.genes[i]
        
        if feasible_only:
            if not self.is_dag():
                self.repair_dag()
            if not self.is_connected():
                self.repair_connectivity()

    def reverse_edge_mutation(self, feasible_only=False):
        adj_matrix = self.compute_adjacency_matrix()
        adj_matrix = reverse_random_edge(adj_matrix)
        self.genes = adj_matrix.flatten().tolist()

        if feasible_only:
            if not self.is_dag():
                self.repair_dag()
            if not self.is_connected():
                self.repair_connectivity()

    def compute_adjacency_matrix(self):
        n = len(self.nodes)
        return np.array(self.genes).reshape(n, n)
    
    def individual_to_digraph(self):
        n = len(self.nodes)
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        adj = self.compute_adjacency_matrix()
        for i in range(n):
            for j in range(n):
                if adj[i, j] == 1:
                    G.add_edge(self.nodes[i], self.nodes[j])
        return G

    def is_dag(self):
        G = self.individual_to_digraph()
        return nx.is_directed_acyclic_graph(G)
    
    def is_connected(self):
        G = self.individual_to_digraph()
        return nx.is_weakly_connected(G)
    
    def repair_connectivity(self):
        G = self.individual_to_digraph()
        if not self.is_connected():
            G = fix_disconnected_graph(G)
            adj = nx.to_numpy_array(G)
            self.genes = adj.flatten().tolist()
    
    def compute_bic(self, data):
        G = self.individual_to_digraph()
        score = BIC(G, data)
        self.bic = score
        return score
    
    def repair_dag(self):
        G = self.individual_to_digraph()
        G = break_cycles(G)
        adj = nx.to_numpy_array(G)
        self.genes = adj.flatten().tolist()
    
    def __str__(self):
        return str(self.genes) + ' - '
    

def mutation(individual, feasible_only=True):
    '''
    Mutates an individual in the digraph representation, by randomly adding, removing or reversing an edge.

    Parameters:
    -----------
    individual : Individual
        The individual to be mutated.
    feasible_only : bool
        Whether to create only feasible individuals.

    Returns:
    --------
    Individual
        The mutated individual.
    '''
    nodes = individual.nodes
    digraph = individual.individual_to_digraph()

    # Randomly select two nodes
    node1 = random.choice(nodes)
    node2 = node1
    while node2 == node1:
        node2 = random.choice(nodes)
    
    rand = random.random()
    if digraph.has_edge(node1, node2):
        if rand < 0.5:
            # Reverse an existing edge
            digraph.remove_edge(node1, node2)
            digraph.add_edge(node2, node1)
            if feasible_only:
                digraph = break_cycles(digraph)
                #digraph = search_dag(digraph, node1, node2)

        else:
            # Remove an existing edge
            digraph.remove_edge(node1, node2)
    elif digraph.has_edge(node2, node1):
        if rand < 0.5:
            # Reverse an existing edge
            digraph.remove_edge(node2, node1)
            digraph.add_edge(node1, node2)
            if feasible_only:
                digraph = break_cycles(digraph)
                #digraph = search_dag(digraph, node1, node2)
        else:
            # Remove an existing edge
            digraph.remove_edge(node2, node1)
    else:
        if rand < 0.5:
            # Add a new edge
            digraph.add_edge(node1, node2)
            if feasible_only:
                digraph = break_cycles(digraph)
                #digraph = search_dag(digraph, node1, node2)
        else:
            # Add a new edge
            digraph.add_edge(node2, node1)
            if feasible_only:
                digraph = break_cycles(digraph)
                #digraph = search_dag(digraph, node1, node2)

    if not nx.is_weakly_connected(digraph):
        digraph = fix_disconnected_graph(digraph)
    
    # Update individual genes
    individual.genes = nx.adjacency_matrix(digraph).todense().flatten().tolist()


    return individual

def reverse_random_edge(adj_matrix):
    """
    Reverse the direction of a randomly selected existing edge in the adjacency matrix.
    
    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix representing the directed graph.
    
    Returns:
        numpy.ndarray: Updated adjacency matrix with a randomly selected edge reversed.
    """
    # Find indices of existing edges
    edges = np.argwhere(adj_matrix == 1)
    
    if len(edges) == 0:
        print("No edges to reverse.")
        return adj_matrix
    
    # Randomly select an edge
    edge_idx = random.choice(range(len(edges)))
    u, v = edges[edge_idx]
    
    # Copy the adjacency matrix to avoid modifying the original
    reversed_matrix = np.copy(adj_matrix)
    
    # Reverse the direction of the selected edge
    reversed_matrix[u][v] = 0
    reversed_matrix[v][u] = 1
    
    return reversed_matrix


def adjacency_matrix_to_individual(adjacency_matrix, nodes):
    return Individual(adjacency_matrix.flatten().tolist(), nodes)

def single_point_crossover(parent1, parent2, feasible_only=False):
    '''
    Single point cross over.

    Parameters:
    -----------
    parent1 : Individual
        The first parent.
    parent2 : Individual
        The second parent.

    Returns:
    --------
    Individual
        The child.
    '''
    point = random.randint(0, len(parent1.genes))
    child1 = Individual(parent1.genes[:point] + parent2.genes[point:], parent1.nodes)
    child2 = Individual(parent2.genes[:point] + parent1.genes[point:], parent1.nodes)

    if feasible_only:
        if not child1.is_dag():
            child1.repair_dag()
        if not child1.is_connected():
            child1.repair_connectivity()
        if not child2.is_dag():
            child2.repair_dag()
        if not child2.is_connected():
            child2.repair_connectivity()

    #return best child
    if child1.bic < child2.bic:
        return child1
    else:
        return child2

def bnc_pso_crossover(parent1, parent2, feasible_only=False):
    child1 = Individual([], parent1.nodes)
    child2 = Individual([], parent1.nodes)
    for i in range(len(parent1.genes)):
        if parent1.genes[i] == parent2.genes[i]:
            child1.genes.append(parent1.genes[i])
            child2.genes.append(parent1.genes[i])
        else:
            child1.genes.append(random.choice([parent1.genes[i], parent2.genes[i]]))
            child2.genes.append(random.choice([parent1.genes[i], parent2.genes[i]]))
    if feasible_only:
        if not child1.is_dag():
            child1.repair_dag()
        if not child1.is_connected():
            child1.repair_connectivity()
        if not child2.is_dag():
            child2.repair_dag()
        if not child2.is_connected():
            child2.repair_connectivity()

    return child1, child2

def uniform_crossover(parent1, parent2, feasible_only=False):
    '''
    Uniform cross over.

    Parameters:
    -----------
    parent1 : Individual
        The first parent.
    parent2 : Individual
        The second parent.

    Returns:
    --------
    Individual
        The child.
    '''
    child1 = Individual([], parent1.nodes)
    child2 = Individual([], parent1.nodes)
    for i in range(len(parent1.genes)):
        child1.genes.append(random.choice([parent1.genes[i], parent2.genes[i]]))
        child2.genes.append(random.choice([parent1.genes[i], parent2.genes[i]]))
    if feasible_only:
        if not child1.is_dag():
            child1.repair_dag()
        if not child1.is_connected():
            child1.repair_connectivity()
        if not child2.is_dag():
            child2.repair_dag()
        if not child2.is_connected():
            child2.repair_connectivity()
            
    return child1, child2

def create_population(pop_size, nodes, data, feasible_only=True):
    '''
    Creates a population of individuals.

    Parameters:
    -----------
    pop_size : int
        The size of the population.
    nodes : list
        The list of nodes.
    feasible_only : bool
        Whether to create only feasible individuals.

    Returns:
    --------
    list
        The population.
    '''
    pop = []
    for _ in range(pop_size):
        individual = Individual([], nodes)
        sparsity = 0.1
        #print("initializing individual randomly")
        individual.init_random(num_nodes = len(nodes), sparsity=sparsity)
        #print("individual created, fixing connectivity and cycles")
        if feasible_only:
            #print("FEASIBLE ONLY, LETS CHECK")
            if not individual.is_dag():
                #print("repairing cycles")
                individual.repair_dag()
            if not individual.is_connected():
                #print("repairing connectivity")
                individual.repair_connectivity()
        #print("individual fixed, computing BIC")
        individual.compute_bic(data)
        pop.append(individual)
    return pop

    
    




# Main
if __name__ == "__main__":
    # Parameters
    gen_max = 100
    p_cruzamento = 0.7
    p_mutacao = 0.2
    pop_size = 100
    times = 1
    av_max = 100
    
    data = load_asia_data(sample_size=1000)
    pop = create_population(pop_size, data.columns, data, feasible_only=False)
