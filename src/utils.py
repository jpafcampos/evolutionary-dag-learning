import random
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
from matplotlib import pyplot as plt


def break_cycles(G):
    '''
    Repairs the cyclic graph G into a DAG by repeatedly removing a random edge
    from a detected cycle.

    Behaviour is identical to the original (random edge from find_cycle removed
    until acyclic). It is faster because it drives the loop off find_cycle alone
    instead of also calling is_directed_acyclic_graph every iteration: find_cycle
    raises NetworkXNoCycle exactly when the graph is a DAG, so one traversal per
    iteration does the work of the previous two.
    '''
    while True:
        try:
            cycle = nx.find_cycle(G)
        except nx.NetworkXNoCycle:
            return G
        edge = random.choice(cycle)
        G.remove_edge(edge[0], edge[1])


def fix_disconnected_graph(G):
    '''
    Fixes the graph so that it becomes connected while avoiding cycles.

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
    for i in range(len(components) - 1):
        nodes1 = list(components[i])
        nodes2 = list(components[i + 1])
        node1 = random.choice(nodes1)
        node2 = random.choice(nodes2)
        G.add_edge(node1, node2)

    return G


def search_dag(G, edge_a, edge_b):
    '''
    Function to make sure that the resulting graph is a DAG.
    (Unused in the current mutation path; kept for reference.)
    '''
    no_dag = list(nx.simple_cycles(G))
    while no_dag != []:
        no_dag = no_dag[0]
        if len(no_dag) > 2:
            rand_i = random.randint(0, len(no_dag) - 1)
            if rand_i == 0:
                rand_aux = rand_i + 1
            elif rand_i == len(no_dag) - 1:
                rand_aux = rand_i - 1
            else:
                rand_aux = random.random()
                if rand_aux <= 0.5:
                    rand_aux = rand_i + 1
                else:
                    rand_aux = rand_i - 1
            aux = 0
            while (no_dag[rand_i] == edge_a and no_dag[rand_aux] == edge_b) and aux < 10:
                aux += 1
                if rand_i == 0:
                    rand_i = rand_aux + 1
                elif rand_aux == len(no_dag) - 1:
                    rand_aux = rand_i - 1
                else:
                    if random.random() < 0.5:
                        rand_i = rand_aux + 1
                    else:
                        rand_aux = rand_i - 1
            if aux < 10:
                if rand_i < rand_aux:
                    if random.random() <= 0.5:
                        G.remove_edge(no_dag[rand_i], no_dag[rand_aux])
                    else:
                        G.remove_edge(no_dag[rand_i], no_dag[rand_aux])
                        G.add_edge(no_dag[rand_aux], no_dag[rand_i])
                else:
                    if random.random() <= 0.5:
                        G.remove_edge(no_dag[rand_aux], no_dag[rand_i])
                    else:
                        G.remove_edge(no_dag[rand_aux], no_dag[rand_i])
                        G.add_edge(no_dag[rand_i], no_dag[rand_aux])
            else:
                G.remove_edge(no_dag[rand_i], no_dag[rand_aux])
        else:
            G.remove_edge(edge_b, edge_a)
        no_dag = list(nx.simple_cycles(G))
    return G


def compute_mean_score(score_history):
    '''
    Computes the mean score for each iteration.
    '''
    return [statistics.mean(scores) for scores in score_history]


def compute_std_score(score_history):
    '''
    Computes the standard deviation of the score for each iteration.
    '''
    return [statistics.stdev(scores) for scores in score_history]


def compute_min_score(score_history):
    '''
    Computes the minimum (best) score for each iteration.
    '''
    return [min(scores) for scores in score_history]
