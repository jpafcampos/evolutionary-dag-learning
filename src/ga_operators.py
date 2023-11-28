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

def fix_graph(G):
    '''
    Fixes the graph so that it becomes connected and avoiding cycles.

    Parameters:
    G (nx.DiGraph): The graph to be fixed.

    Returns:
    nx.DiGraph: The fixed graph.
    '''
    if nx.is_weakly_connected(G):
        return G

    # Find connected components
    components = list(nx.weakly_connected_components(G))

    # Add edges to connect components
    for i in range(len(components)-1):
        nodes1 = list(components[i])
        nodes2 = list(components[i+1])
        node1 = random.choice(nodes1)
        node2 = random.choice(nodes2)
        G.add_edge(node1, node2)

    return G

def crossover(ind1, ind2, nodes, feasible_only = False):
    '''
    Perform crossover between two individuals (graphs) by combining their edges.
    
    Parameters:
    -----------
    ind1 : networkx.DiGraph
        First individual (graph) to be crossed over.
    ind2 : networkx.DiGraph
        Second individual (graph) to be crossed over.
    nodes : list
        List of nodes that should be present in the resulting graphs.
    feasible_only : bool, optional
        If True, only feasible graphs (DAGs) will be returned. Default is False.

    Returns:
    --------
    networkx.DiGraph
        Resulting graph after crossover.
    float
        BIC score of the resulting graph.
    dict
        Updated dictionary containing auxiliary information for the BIC scoring function.
    '''
    a = ind1.copy()
    a.update(ind2.copy())
    f1 = nx.DiGraph()
    f2 = nx.DiGraph()
    for i in nodes:
        if i not in f1.nodes():
            f1.add_node(i)
        if i not in f2.nodes():
            f2.add_node(i)
    for i in ind1.edges():
        if i in ind2.edges():
            f1.add_edge(i[0], i[1])
            f2.add_edge(i[0], i[1])
    for i in a.edges():
        if not f1.has_edge(i[0], i[1]):
            if f1.has_edge(i[1], i[0]):
                r = random.random()
                if r <= 0.5:
                    f1.remove_edge(i[1], i[0])
                    f1.add_edge(i[0], i[1])
                    if feasible_only:
                        search_dag(f1, i[0], i[1])
                else:
                    if f2.has_edge(i[1], i[0]):
                        f2.remove_edge(i[1], i[0])
                        f2.add_edge(i[0], i[1])
                    if feasible_only:
                        search_dag(f2, i[0], i[1])
            else:
                r = random.random()
                if r <= 0.5:
                    f1.add_edge(i[0], i[1])
                    if feasible_only:
                        search_dag(f1, i[0], i[1])
                else:
                    f2.add_edge(i[0], i[1])
                    if feasible_only:
                        search_dag(f2, i[0], i[1])

    # Check if new individuals are connected, and fix them if they are not
    if not nx.is_weakly_connected(f1):
        f1 = fix_graph(f1)
    if not nx.is_weakly_connected(f2):
        f2 = fix_graph(f2)

    return f1, f2


def mutation(ind1, nodes, feasible_only=False):
    '''
    Applies a mutation operator to the given individual by randomly adding, removing, or reversing an edge between two nodes in the graph.

    Parameters:
    ind1 (nx.DiGraph): The individual to be mutated.
    nodes (list): A list of nodes in the graph.
    feasible_only (bool): If True, the mutation operator will only produce feasible individuals (i.e. DAGs).

    Returns:
    tuple: A tuple containing the mutated individual and a list of the nodes and action taken during the mutation.
    '''
    R = nx.DiGraph()
    R = ind1.copy()
    action="NNN"
    for a in nodes:
        if a not in R.nodes():
            R.add_node(a)

    node1= random.randint(0,len(nodes)-1)
    node2= node1
    while(node1 == node2):
        node2= random.randint(0,len(nodes)-1)
    rand = random.random()
    if ind1.has_edge(nodes[node1],nodes[node2]):
        if rand <=0.5:
            R.remove_edge(nodes[node1],nodes[node2])
            R.add_edge(nodes[node2],nodes[node1])
            if feasible_only:
                search_dag(R,node1,node2)
            action="right"
        else:
            R.remove_edge(nodes[node1],nodes[node2])
            action="remove"
    elif ind1.has_edge(nodes[node2],nodes[node1]):
        if rand <=0.5:
            R.remove_edge(nodes[node2],nodes[node1])
            R.add_edge(nodes[node1],nodes[node2])
            if feasible_only:
                search_dag(R,node1,node2)
            action="left"
        else:
            R.remove_edge(nodes[node2],nodes[node1])
            action="remove"
    else:
        if rand <=0.5:
            R.add_edge(nodes[node1],nodes[node2])
            if feasible_only:
                search_dag(R,node1,node2)
            action="right"
        else:
            R.add_edge(nodes[node2],nodes[node1])
            if feasible_only:
                search_dag(R,node1,node2)
            action="left"
    ns = [nodes[node2],nodes[node1],action]

    # Check if new individual is connected, and fix it if it is not
    if not nx.is_weakly_connected(R):
        R = fix_graph(R)
    
    return R #, ns


def tournament(Agents, popSize):
    k=0.75
    individuo1=round((popSize-1) * random.random())
    individuo2=round((popSize-1) * random.random())
    r = random.random()
    if r<=k:
        if Agents[individuo1].score<=Agents[individuo2].score:
            return individuo1
        else:
            return individuo2
    else:
        if Agents[individuo1].score>=Agents[individuo2].score:
            return individuo1
        else:
            return individuo2


def elitism(all_agents,popSize):
    prox_ind=[]
    fit=[]
    melhorfit=all_agents[0].score
    melhorind=all_agents[0].variable
    for j in range(len(all_agents)):
        fit.append(all_agents[j].score)
        if all_agents[j].score<melhorfit:
            melhorfit=all_agents[j].score
            melhorind=all_agents[j].variable
            
    indice=[i[0] for i in sorted(enumerate(fit), key=lambda x:x[1])] # sort
    i=0
    while len(prox_ind)<popSize:
        prox_ind.append(all_agents[indice[i]])
        i=i+1
    return prox_ind,melhorfit,melhorind


def createPopulation(popSize, nodes, density_factor = 0.3, feasible_only = False):
    '''
    Creates the initial population of individuals (graphs).
    
    Parameters:
    -----------
    popSize : int
        The size of the population.
    nodes : list
        A list of nodes in the graph.
    
    Returns:
    --------
    list
        A list of individuals (graphs).
    '''
    population = []
    for i in range(popSize):
        ind = nx.DiGraph()
        for j in nodes:
            ind.add_node(j)
        for j in nodes:
            for k in nodes:
                if j != k:
                    r = random.random()
                    if r <= density_factor:
                        if random.random() <= 0.5:
                            ind.add_edge(j, k)
                        if feasible_only:
                            search_dag(ind, j, k)
        if not nx.is_weakly_connected(ind):
            ind = fix_graph(ind)
        # Save an image of the created graph on a different figure
        #plt.figure()
        #nx.draw(ind, with_labels=True)
        #plt.savefig('graph' + str(i) + '.png')
 
        population.append(ind)

    return population


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
    pop = createPopulation(pop_size, data.columns, data, feasible_only=False)
