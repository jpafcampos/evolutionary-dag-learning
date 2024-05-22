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



def FES(graph, data, nodes, bics, goalscore):
    again=True
    bic = BIC(graph, data)

    while(again):
        aux_graph=copy.deepcopy(graph)
        new_graph=copy.deepcopy(graph)
        best_bic=bic
        no_add=True

        if best_bic<(goalscore+0.00000001):
            break;
        for i in nodes:
            for j in nodes:
                if i !=j:
                    if not aux_graph.has_edge(i,j):                
                        aux_graph.add_edge(i,j)
                        if nx.is_directed_acyclic_graph(aux_graph):
                            [bic,aux_bic]=BIC(aux_graph, data) 
                            bic=abs(bic)
                            if best_bic>bic:
                                no_add=False
                                new_graph=copy.deepcopy(aux_graph)
                                best_bic=bic
                                bics.append(best_bic)
                            else:
                                bics.append(best_bic)
                        aux_graph.remove_edge(i,j)
        graph=copy.deepcopy(new_graph)
        bic=best_bic
        if no_add:
            again=False
    return new_graph,best_bic,aux_bic,bics

def BES(graph, data, nodes, bics, goalscore):
    again = True
    bic = BIC(graph, data)
    while(again):
        aux_graph = copy.deepcopy(graph)
        new_graph = copy.deepcopy(graph)
        bic = abs(bic)
        best_bic = bic
        if best_bic<(goalscore+0.00000001):
            break;            
        no_add = True
        for i in nodes:
            for j in nodes:
                if i !=j:
                    if aux_graph.has_edge(i,j):                
                        aux_graph.remove_edge(i,j)
                        if nx.is_directed_acyclic_graph(aux_graph):
                            bic = BIC(aux_graph, data) 
                            bic = abs(bic)
                            #print(bic)
                            if best_bic>bic:
                                no_add = False
                                new_graph = copy.deepcopy(aux_graph)
                                best_bic=bic
                                bics.append(best_bic)
                            else:
                                bics.append(best_bic)
                        aux_graph.add_edge(i,j)
        graph = copy.deepcopy(new_graph)
        bic = best_bic
        if no_add:
            again = False
    return new_graph, best_bic, bics

def GES(nodes, data, goalscore):
    aux_bic = 0
    graph = nx.DiGraph()
    for a in nodes:
        graph.add_node(a)
    bics = []
    [graph, bic, aux_bic, bics] = FES(graph, data, nodes, bics, goalscore)
    [graph, bic, aux_bic, bics] = BES(graph, data, nodes, bics, goalscore)
    return graph, bic, aux_bic, bics


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

        # Run GES
        print("Running GES")
        best_graph = GES(nodes, data, goal_bic)

        print("best graph returned BIC")
        print(best_graph.bic)

        # Compute metrics
        compute_metrics()

        print("Algorithm ended. Computing Results")
        end = time.time()
        time_vector.append(end-start)


    print("Mean time:", statistics.mean(time_vector))