# -*- coding: utf-8 -*-
"""
Created on may 2024

@author: João Pedro Campos
"""

import random
#import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import csv
import time
import statistics
import bnlearn as bn
from pgmpy.metrics import structure_score
from pgmpy.models import BayesianNetwork
import gc
from utils import *
from loaders import *
import matplotlib.pyplot as plt
import argparse
import copy

import math
from itertools import permutations

from scipy.stats import multivariate_normal
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.models import BayesianModel

def bic_score(dag, data):
    bic = BicScore(data)
    if len(dag.edges()) == 0:
        return float('inf')
    model = BayesianNetwork(list(dag.edges()))
    score = np.abs(bic.score(model))
    return score

def get_connected_nodes(graph):
    # Initialize an empty set to store nodes with at least one edge
    nodes_with_edges = set()

    # Iterate through the edges to collect nodes
    for edge in graph.edges():
        nodes_with_edges.update(edge)

    # Convert the set to a list (if needed)
    nodes_with_edges_list = list(nodes_with_edges) 

    return nodes_with_edges_list

def FES(graph, data, nodes, bics, goalscore):
    bic = BicScore(data)
    again=True
    #bic = BIC(graph, data)
    score = float('inf')
    count = 0
    edges_and_deltas = {}
    best_delta = 0
    while(again):
        # get set of candidate edges
        potential_new_edges = (set(permutations(nodes, 2)) - set(graph.edges()) - set([(y, x) for x, y in graph.edges()]))

        for x, y in potential_new_edges:
            if not nx.has_path(graph, y, x):
                old_parents = list(graph.predecessors(y))
                new_parents = old_parents + [x]
                score_delta = bic.local_score(y, new_parents) - bic.local_score(y, old_parents)
                edges_and_deltas[(x, y)] = score_delta
                if score_delta < best_delta:
                    best_delta = score_delta
                    best_edge = (x, y)
        print(best_edge)
        #print(graph.edges())
        # if edge already exists, break
        if best_edge in graph.edges():
            break
        if best_delta < 0:
            graph.add_edge(best_edge[0], best_edge[1])
            bics.append(bic.score(graph))
        
    return graph, 0, 0


def BES(graph, data, nodes, bics, goalscore):
    bic = BicScore(data)
    again=True
    #bic = BIC(graph, data)
    score = float('inf')
    count = 0
    best_delta = 0

    while(again):

        for x, y in graph.edges():
            old_parents = list(graph.predecessors(y))
            new_parents = [var for var in old_parents if var != x]
            score_delta = bic.local_score(y, new_parents) - bic.local_score(y, old_parents)
            if score_delta < best_delta:
                best_delta = score_delta
                best_edge = (x, y)
            
        if best_delta < 0:
            graph.remove_edge(best_edge[0], best_edge[1])
            bics.append(bic.score(graph))
        else:
            break

    return graph, 0, 0
  

def GES(nodes, data, goalscore):
    graph = nx.DiGraph()
    for a in nodes:
        graph.add_node(a)
    bics = []
    graph, bic, bics = FES(graph, data, nodes, bics, goalscore)
    graph, bic, bics = BES(graph, data, nodes, bics, goalscore)
    return graph, bic, bics


def compute_metrics():
    best_score_bic = BIC(best_graph, data)
    best_score_ls = lagrangian(best_graph, data)

    # Hamming distance
    SLF, TLF = learning_factors(ground_truth, best_graph)
    # Print Results
    print('Best graph:', best_graph.edges())
    print('Best score BIC:', best_score_bic)
    print('Best score LS:', best_score_ls)
    print('Structure Learning Factor:', SLF)
    print('Topology Learning Factor:', TLF)

    # Save results
    print('Saving results')

    filename = PATH +f'results_{args.data}.csv'

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.data, args.sample_size, best_score_bic, best_score_ls, SLF, TLF])

    # Draw best graph
    plt.figure()
    nx.draw(best_graph, with_labels=True)
    plt.savefig(PATH + 'best_graph_' + '.png')


if __name__ == '__main__':

    # Load arguments
    parser = argparse.ArgumentParser(description='GES Algorithm for BN structural learning.')
    parser.add_argument('--sample_size', type=int, help='Number of samples to be used.')
    parser.add_argument('--data', type=str, help='Name of the dataset.')
    parser.add_argument('--num_runs', type=int, help='Number of runs', default=1)
    args = parser.parse_args()

    #PATH = define_path_to_save(args.data, args.feasible_only)
    PATH = '/home/joao/Desktop/UFMG/PhD/code/EA-DAG/results/GES/' + args.data + '/' 

    # Parameters
    n_runs = args.num_runs

    time_vector = []
    for i in range(n_runs):

        # Load data
        if args.data == 'asia':
            data = load_asia_data(sample_size=args.sample_size)
            ground_truth = load_gt_network_asia()
            print('Asia data and ground truth loaded')
        elif args.data == 'child':
            data = load_child_data(sample_size=args.sample_size)
            ground_truth = load_gt_network_child()
            print('Child data and ground truth loaded')
        else:
            # dataset not available
            print('Dataset not available.')
            exit(1)

        goal_bic = BIC(ground_truth, data)
        print('Goal BIC:', goal_bic)

        # measure time
        start = time.time()

        # Create list of nodes
        nodes = data.columns
        nodes = list(nodes)
        print(nodes)

        # Run GES
        print("Running GES")
        best_graph, bic, bics = GES(nodes, data, goal_bic)

        print("best graph returned BIC")
        print(BIC(best_graph, data[get_connected_nodes(best_graph)]))

        # Compute metrics
        compute_metrics()

        print("Algorithm ended. Computing Results")
        end = time.time()
        time_vector.append(end-start)


    print("Mean time:", statistics.mean(time_vector))