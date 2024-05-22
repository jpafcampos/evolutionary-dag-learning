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
from ga_operators import *
from loaders import *
import matplotlib.pyplot as plt
import argparse
import copy



def MAGA(population, max_iter, Pm_min, Pm_max, Po, Pc_min, Pc_max, t_max, goal_bic, sL, sPm, sGen, feasible_only, verbose=False):

    best_bic = float('inf')
    best_graph = None
    best_pos = 0
    iteration = 0
    t = 0
    while (best_bic > goal_bic + 0.00000001) and (iteration < max_iter):
        if verbose:
            print('Iteration:', iteration)
        Pc=(Pc_min-t*(Pc_min-Pc_max)/t_max)
        Pm=(Pm_min-t*(Pm_min-Pm_max)/t_max)
        t += 1
        best_ind = 0
        for agent_idx in range(len(population)):
            aux_best_bic = population[agent_idx].bic if population[agent_idx].bic != None else population[agent_idx].compute_bic(data)
            aux_best_idx = agent_idx
            if random.uniform(0,1) < Pc:
                best_neighbor = find_best_neighbor(population, agent_idx)
                if population[agent_idx].bic > population[best_neighbor].bic:
                    child1, child2 = bnc_pso_crossover(population[agent_idx], population[best_neighbor], feasible_only)
                    child1.compute_bic(data)
                    child2.compute_bic(data)
                    # current agent receives best child's fenotype
                    best_child = child1 if child1.bic < child2.bic else child2
                    population[agent_idx].update_fenotype(best_child)
                    aux_best_bic = population[agent_idx].bic
            if aux_best_bic < population[best_ind].bic:
                best_ind = aux_best_idx
        total_m = round(len(population) * len(population[0].nodes) * Pm)
        aux_m = 0
        while aux_m < total_m:
            aux_rand = random.randint(0, len(population)-1)
            if aux_rand != best_ind:
                aux_m += 1
                agent_before_mutation = copy.deepcopy(population[aux_rand])
                new_agent = mutation(agent_before_mutation, feasible_only)
                new_agent.compute_bic(data)
                if new_agent.bic < agent_before_mutation.bic:
                    population[aux_rand].update_fenotype(new_agent)
                    aux_m += 1
                elif random.uniform(0,1) < Po:
                    population[aux_rand].update_fenotype(new_agent)
                    aux_m += 1
        for agent in population:
            if agent.bic < best_bic:
                best_bic = agent.bic
                best_graph = agent
                best_pos = agent.pos
        sBest = self_learning(sL, best_graph, sPm, Po, sGen, data, feasible_only)
        population[best_pos[0]] = sBest
        iteration += 1

    return best_graph, best_graph.bic, population

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
        writer.writerow([args.data, args.sample_size, args.max_iter, best_score_bic, best_score_ls, SLF, TLF])

    # Draw best graph
    plt.figure()
    nx.draw(best_graph, with_labels=True)
    plt.savefig(PATH + 'best_graph_' + '.png')


if __name__ == '__main__':

    # Load arguments
    parser = argparse.ArgumentParser(description='Evolutionary Algorithm for BN structural learning.')
    parser.add_argument('--sample_size', type=int, help='Number of samples to be used.')
    parser.add_argument('--data', type=str, help='Name of the dataset.')
    parser.add_argument('--max_iter', type=int, help='Maximum number of iterations.')
    parser.add_argument('--sGen', type=int, help='Maximum number of iterations in self learning.')
    parser.add_argument('--sPm', type=float, help='Mutation probability inside self learning.')
    parser.add_argument('--Po', type=float, help='Prob of keeping worse individual.')
    parser.add_argument('--Pm_min', type=float, help='Min probability of mutation.')
    parser.add_argument('--Pm_max', type=float, help='Max probability of mutation.')
    parser.add_argument('--Pc_min', type=float, help='Min probability of crossover.')
    parser.add_argument('--Pc_max', type=float, help='Max probability of crossover.')
    parser.add_argument('--L_size', type=int, help='Grid size.')
    parser.add_argument('--sL', type=int, help='Small Grid size.')
    parser.add_argument('--mu', type=float, help='Lagrangian multiplier.')
    parser.add_argument('--feasible_only', action='store_true')
    parser.add_argument('--no-feasible_only', dest='feasible_only', action='store_false')
    parser.add_argument('--feasible_only_init_pop', action='store_true')
    parser.add_argument('--no-feasible_only_init_pop', dest='feasible_only_init_pop', action='store_false')
    parser.add_argument('--num_runs', type=int, help='Number of runs', default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    args = parser.parse_args()

    #PATH = define_path_to_save(args.data, args.feasible_only)
    PATH = '/home/joao/Desktop/UFMG/PhD/code/EA-DAG/results/MAGA/' + args.data + '/' 

    # Parameters
    max_iter = args.max_iter
    Pm_min = args.Pm_min
    Pm_max = args.Pm_max
    Po = args.Po
    Pc_min = args.Pc_min
    Pc_max = args.Pc_max
    feasible_only = args.feasible_only
    mu = args.mu
    n_runs = args.num_runs
    L_size = args.L_size
    sL = args.sL

    t_max = 10


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

        # Create initial population
        nodes = data.columns
        nodes = list(nodes)
        print("Creating initial population")
        population = create_MAGA_population(L_size, nodes, data, feasible_only=args.feasible_only_init_pop)

        # Evolve population
        print("Evolving population")
        #def MAGA(population, max_iter, Pm_min, Pm_max, Po, Pc_min, Pc_max, t_max, goal_bic, sL, sPm, sGen, feasible_only, verbose=False):
        best_graph, _, population = MAGA(population, max_iter, Pm_min, Pm_max, Po, Pc_min,
                                                   Pc_max, t_max, goal_bic, sL, args.sPm, args.sGen, feasible_only, verbose=args.verbose)
        print("best graph returned BIC")
        print(best_graph.bic)
        print("first ind bic")
        print(population[0].bic)
        print("bic of all inds")
        for ind in population:
            print(ind.compute_bic(data))
        
        best_graph = best_graph.individual_to_digraph()
        # Compute metrics
        compute_metrics()

        print("Algorithm ended. Computing Results")
        end = time.time()
        time_vector.append(end-start)


    print("Mean time:", statistics.mean(time_vector))