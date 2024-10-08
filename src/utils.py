import random
#import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import csv
from pgmpy.readwrite import BIFReader
import time
import scipy.linalg as slin
from scipy.linalg import det
from scipy import linalg
import statistics
import bnlearn as bn
from pgmpy.metrics import structure_score
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ScoreCache
from pgmpy.estimators import BicScore, K2Score, BDeuScore
import gc
from matplotlib import pyplot as plt



def BIC(G, data):
    """
    Compute the BIC score of a graph G given a dataset.
    
    Parameters:
    -----------
    G : networkx.DiGraph
        Graph whose score will be computed.
    data : pandas.DataFrame
        Dataset used to compute the score.
    
    Returns:
    --------
    float
        BIC score of the graph.
    int
        Number of times the metric was calculated

    """
    # If G is not a DAG, return inf
    if not nx.is_directed_acyclic_graph(G):
        print("Error: G is not a DAG")
        return float('inf')

    # G is converted to a BayesianNetwork Model in pgmpy
    # the bic is computed using the pgmpy function structure_score
    scoring_method = ScoreCache(BicScore(data), data).local_score
    score = 0
    for node in G.nodes():
        parents = list(G.predecessors(node))
        score += scoring_method(node, parents)

    score = -score
    
    return score

def least_squares(G, data):
    '''
    Compute the least squares score of a graph G given a dataset.
    The score is computed as proposed in NOTEARS paper (Zheng et al., 2018).

    Parameters:
    -----------
    G : networkx.DiGraph
        Graph whose score will be computed.
    data : pandas.DataFrame
        Dataset used to compute the score.
    auxn: counts how many times the metric was calculated

    Returns:
    --------
    float
        Least squares score of the graph.
    numpy.ndarray
        Gradient of the least squares score of the graph.
    int
        Number of times the metric was calculated

    References:
    -----------
    Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018).
    DAGs with NO TEARS: Continuous optimization for structure learning.
    
    '''
    # W is the adjacency matrix of G
    W = nx.adjacency_matrix(G).todense()
    W = np.array(W)
    X = data.values
    # least squares score
    M = X @ W
    R = X - M
 
    loss = 0.5 / X.shape[0] * np.sum(R**2)
    
    return loss

def h(W):
    '''
    Compute the 'dagness' of a matrix W.
    The dagness is computed as proposed in NOTEARS paper (Zheng et al., 2018).

    Parameters:
    -----------
    W : numpy.ndarray
        Matrix whose dagness will be computed.

    Returns:
    --------
    float
        Dagness of the matrix.
    '''
    # dagness
    E = slin.expm(W * W)  # (Zheng et al. 2018)
    d = W.shape[0]
    h = np.trace(E) - d
 
    return h

def dagness(G):
    '''
    Compute the 'dagness' of a graph G.
    The dagness is computed as proposed in NOTEARS paper (Zheng et al., 2018).

    Parameters:
    -----------
    G : networkx.DiGraph
        Graph whose dagness will be computed.

    Returns:
    --------
    float
        Dagness of the graph.
    '''
    # Convert graph to matrix
    W = nx.to_numpy_array(G)
    W = np.array(W)
    # dagness
    E = slin.expm(W * W)  # (Zheng et al. 2018)
    d = W.shape[0]
    h = np.trace(E) - d
 
    return h

def lagrangian(G, data, mu=4):
    '''
    Compute the lagrangian of a graph G given a dataset.

    Parameters:
    -----------
    G : networkx.Graph
        Graph whose lagrangian will be computed.
    data : pandas.DataFrame
        Dataset used to compute the lagrangian.
    mu : float, optional
        Regularization parameter.

    Returns:
    --------
    float
        Lagrangian of the graph.
    '''
    # Convert graph to matrix
    W = nx.to_numpy_array(G)
    W = np.array(W)
    # lagrangian
    loss = least_squares(G, data)
    h_W = h(W)
    lagrangian = loss + mu * h_W

    return lagrangian



def break_cyclessss(G):
    '''
    Function that repairs the cyclic graph G to a DAG.
    This function selects a random edge from the cycle and removes it.
    '''
    if nx.is_directed_acyclic_graph(G):
        return G
    
    while not nx.is_directed_acyclic_graph(G):
        cycle = nx.find_cycle(G)
        edge = random.choice(cycle)
        G.remove_edge(edge[0], edge[1])
        #if random.random() <= 0.5:
        #    G.add_edge(edge[1], edge[0])
    
    return G

def break_cycles(G):
    '''
    Function that repairs the cyclic graph G to a DAG.
    This function selects a random edge from the cycle and removes it.
    '''
    if nx.is_directed_acyclic_graph(G):
        return G
    
    while not nx.is_directed_acyclic_graph(G):
        cycles = list(nx.simple_cycles(G))
        
        if not cycles:
            break
        
        # Select a random cycle
        #print("selecting a random cycle")
        cycle = random.choice(cycles)
        
        #print("removing edge from cycle")
        random_node = random.choice(cycle)
        child_node = cycle[(cycle.index(random_node) + 1) % len(cycle)]

        G.remove_edge(random_node, child_node)
    
    return G

def fix_disconnected_graph(G):
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

def make_dag_mfas(G):
    '''
    Function that solves the Minimum Feedback Arc Set problem to transform G into a DAG.
    '''
    # Find the minimum feedback arc set
    
    mfas = nx.minimum_feedback_arc_set(G)  #infelizmente não existe

    # Remove the edges in the minimum feedback arc set
    G.remove_edges_from(mfas)

    return G


# ALTERNATIVE: USE NETWORKX FUNCTION "FIND CYCLE" TO FIND CYCLES IN THE GRAPH AND BREAK THEM


def search_dag(G,edge_a,edge_b):
   
    '''
    Function to make sure that the resulting graph is a DAG.
    '''
    no_dag=list(nx.simple_cycles(G))
    while no_dag != []:
        no_dag=no_dag[0]
        if len(no_dag)>2:
            rand_i=random.randint(0,len(no_dag)-1)
            if rand_i == 0:
                rand_aux=rand_i+1
            elif rand_i == len(no_dag)-1:
                rand_aux=rand_i-1
            else:
                rand_aux=random.random()
                if rand_aux<=0.5:
                    rand_aux=rand_i+1
                else:
                    rand_aux=rand_i-1
            aux=0
            while (no_dag[rand_i]==edge_a and no_dag[rand_aux]==edge_b) and aux<10:
                aux+=1
                if rand_i==0:
                    rand_i=rand_aux+1
                elif rand_aux == len(no_dag)-1:
                    rand_aux=rand_i-1
                else:
                    if random.random()<0.5:
                        rand_i=rand_aux+1
                    else:
                        rand_aux=rand_i-1
            if aux<10:     
                if rand_i<rand_aux:
                    if random.random()<=0.5:
                      G.remove_edge(no_dag[rand_i], no_dag[rand_aux]) 
                    else:
                      G.remove_edge(no_dag[rand_i], no_dag[rand_aux])
                      G.add_edge(no_dag[rand_aux],no_dag[rand_i])
                else:
                    if random.random()<=0.5:
                      G.remove_edge(no_dag[rand_aux],no_dag[rand_i]) 
                    else:
                      G.remove_edge(no_dag[rand_aux], no_dag[rand_i])
                      G.add_edge(no_dag[rand_i],no_dag[rand_aux])
            else:
               G.remove_edge(no_dag[rand_i], no_dag[rand_aux])  
        else:
            G.remove_edge(edge_b,edge_a)
        no_dag=list(nx.simple_cycles(G))     
    return G


def compute_mean_score(score_history):
    '''
    Computes the mean score for each iteration.

    Parameters:
    -----------
    score_history : list
        List containing the score history for each iteration.

    Returns:
    --------
    list
        List containing the mean score for each iteration.
    '''
    mean_score = []
    for i in range(len(score_history)):
        mean_score.append(statistics.mean(score_history[i]))
    return mean_score

def compute_std_score(score_history):
    '''
    Computes the standard deviation of the score for each iteration.

    Parameters:
    -----------
    score_history : list
        List containing the score history for each iteration.

    Returns:
    --------
    list
        List containing the standard deviation of the score for each iteration.
    '''
    std_score = []
    for i in range(len(score_history)):
        std_score.append(statistics.stdev(score_history[i]))
    return std_score

def compute_min_score(score_history):
    '''
    Computes the minimum (best) score for each iteration.

    Parameters:
    -----------
    score_history : list
        List containing the score history for each iteration.

    Returns:
    --------
    list
        List containing the minimum score for each iteration.
    '''
    min_score = []
    for i in range(len(score_history)):
        min_score.append(min(score_history[i]))
    return min_score

def save_feasible_only_plots(PATH, bic_hist, ls_hist):
    best_score_hist_bic = compute_min_score(bic_hist)
    mean_score_hist_bic = compute_mean_score(bic_hist)
    #std_score_hist_bic = compute_std_score(bic_hist)
    best_score_hist_ls = compute_min_score(ls_hist)
    mean_score_hist_ls = compute_mean_score(ls_hist)
    #std_score_hist_ls = compute_std_score(ls_hist)
    # Save plots
    fig, ax1 = plt.subplots()

    # Plotting BIC scores on the first y-axis
    bic_line, = ax1.plot(best_score_hist_bic, color='blue', label='BIC')
    ax1.set_ylabel('BIC', color='blue')

    # Creating a secondary y-axis for Least Squares scores
    ax2 = ax1.twinx()
    ls_line, = ax2.plot(best_score_hist_ls, color='red', linestyle = '--', label='Least Squares')
    ax2.set_ylabel('Least Squares', color='red')

    # Displaying legends for both plots
    lines = [bic_line, ls_line]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')
    plt.savefig(PATH + 'best_score_history.png')
    
    plt.figure()
    fig2, ax1 = plt.subplots()

    # Plotting BIC scores on the first y-axis
    bic_line, = ax1.plot(mean_score_hist_bic, color='blue', label='BIC (mean)')
    ax1.set_ylabel('BIC', color='blue')

    # Creating a secondary y-axis for Least Squares scores
    ax2 = ax1.twinx()
    ls_line, = ax2.plot(mean_score_hist_ls, color='red', linestyle = '--', label='Least Squares (mean)')
    ax2.set_ylabel('Least Squares', color='red')

    # Displaying legends for both plots
    lines = [bic_line, ls_line]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')
    plt.savefig(PATH + 'mean_score_history.png')





def save_infeasible_plots(PATH, ls_hist, dagness_hist, bic_hist):
    best_score_hist = compute_min_score(ls_hist)
    mean_score_hist = compute_mean_score(ls_hist)
    #std_score_hist = compute_std_score(ls_hist)
    # Save plots
    fig, ax1 = plt.subplots()

    # Creating a secondary y-axis for BIC scores
    ax2 = ax1.twinx()

    # Plotting Least Squares scores on the left y-axis
    ls_line, = ax1.plot(best_score_hist, color='blue', label='Least Squares')
    ax1.set_ylabel('Least Squares', color='blue')

    # Plotting BIC scores on the right y-axis
    bic_line, = ax2.plot(bic_hist, color='red', linestyle='--', label='BIC')
    ax2.set_ylabel('BIC', color='red')

    # Displaying legends for both plots
    lines = [ls_line, bic_line]
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc='upper right')
    plt.savefig(PATH + 'ls_bic_history.png')

    plt.figure()
    plt.plot(dagness_hist)
    plt.xlabel('Iteration')
    plt.ylabel('Dagness')
    plt.title('Dagness History')
    plt.savefig(PATH + 'dagness_history.png')

    # plot the mean LS score history
    plt.figure()
    plt.plot(mean_score_hist)
    plt.xlabel('Iteration')
    plt.ylabel('Mean LS score')
    plt.title('Mean LS score History')
    plt.savefig(PATH + 'mean_ls_history.png')
