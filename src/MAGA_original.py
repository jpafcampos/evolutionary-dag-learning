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

def select_best(population, n, data):
    '''
    Select the n best individuals from a population.

    Parameters:
    -----------
    population : list
        List containing the population of DAGs.
    n : int
        Number of individuals to be selected.
    score_function : function
        Function used to compute the score of each individual.
    data : pandas.DataFrame
        Dataset used to compute the score.
    
    Returns:
    --------
    list
        List containing the selected individuals.
    '''
    # Select best individuals
    scores = [i.compute_bic(data) if i.bic == None else i.bic for i in population]
    rank = np.argsort(scores)
    best_n = rank[:n]
    return list(population[i] for i in best_n)

def select_parents_by_rank(population, num_parents, selection_pressure, data):
    '''
    Select parents from a population using rank-based selection.

    Parameters:
    -----------
    population : list
        List containing the population of DAGs.
    num_parents : int
        Number of parents to be selected.
    data : pandas.DataFrame
        Dataset used to compute the score.
    
    Returns:
    --------
    list
        List containing the selected parents.
    '''
    # Rank population
    population = select_best(population, len(population), data)
    # Compute rank probabilities
    s = selection_pressure # Selection pressure between 1 and 2 as described in Eiben and Smith (2003)
    mu = len(population)
    rank_probs = [(2-s)/mu + 2*(i)*(s-1)/(mu*(mu-1)) for i in range(1, mu+1)]
    rank_probs = rank_probs[::-1]
    # Select parents
    parents = random.choices(population, weights=rank_probs, k=num_parents)
    return parents

def select_parents_by_tournament(population, num_parents, data):
    '''
    Select parents from a population using tournament selection.

    Parameters:
    -----------
    population : list
        List containing the population of DAGs.
    num_parents : int
        Number of parents to be selected.
    data : pandas.DataFrame
        Dataset used to compute the score.
    
    Returns:
    --------
    list
        List containing the selected parents.
    '''
    parents = []
    for _ in range(num_parents):
        # Select random individuals
        tournament = random.sample(population, 2)
        # Select best individual
        best = select_best(tournament, 1, data)[0]
        parents.append(best)
    return parents


def findNeighbors(i, j, L):
    if i == 0:
        i1 = L - 1
    else:
        i1 = i - 1
    if i == L - 1:
        i2 = 0
    else:
        i2 = i + 1
    if j == 0:
        j1 = L - 1
    else:
        j1 = j - 1
    if j == L - 1:
        j2 = 0
    else:
        j2 = j + 1
    return [i1, j1, i2, j2]

def MAGA(population, max_iter, mutation_rate, crossover_rate, patience, selection_pressure, goal_bic, crossover_function, feasible_only, verbose=False):

    pass



def compute_metrics(run_both = False, alg = 'bic'):
    best_score_bic = BIC(best_graph, data)
    best_score_ls = lagrangian(best_graph, data)
    cross = args.crossover_function

    # Hamming distance
    SLF, TLF = learning_factors(ground_truth, best_graph)

    # Save results
    print('Saving results')

    if run_both:
        filename = PATH +f'results_{args.data}_both_{cross}.csv'
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([args.data, args.sample_size, args.max_iter, args.mutation_rate, 
                            args.crossover_rate, args.popSize, args.patience, alg, 
                            args.feasible_only_init_pop, best_score_bic, best_score_ls, SLF, TLF])
    
    else:
        filename = PATH +f'results_{args.data}_{"feasible" if args.feasible_only else "infeasible"}_{cross}.csv'
    
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([args.data, args.sample_size, args.max_iter, args.mutation_rate, 
                            args.crossover_rate, args.popSize, args.patience, args.feasible_only, 
                            args.feasible_only_init_pop, args.selection_pressure, best_score_bic, best_score_ls, SLF, TLF])


    # Print Results
    print('Best graph:', best_graph.edges())
    print('Best score BIC:', best_score_bic)
    print('Best score LS:', best_score_ls)
    print('Structure Learning Factor:', SLF)
    print('Topology Learning Factor:', TLF)

    # Draw best graph
    plt.figure()
    nx.draw(best_graph, with_labels=True)
    plt.savefig(PATH + 'best_graph_' + alg + '.png')


if __name__ == '__main__':

    # Load arguments
    parser = argparse.ArgumentParser(description='Evolutionary Algorithm for BN structural learning.')
    parser.add_argument('--sample_size', type=int, help='Number of samples to be used.')
    parser.add_argument('--data', type=str, help='Name of the dataset.')
    parser.add_argument('--max_iter', type=int, help='Maximum number of iterations.')
    parser.add_argument('--mutation_rate', type=float, help='Probability of mutation.')
    parser.add_argument('--crossover_rate', type=float, help='Probability of crossover.')
    parser.add_argument('--popSize', type=int, help='Population size.')
    parser.add_argument('--patience', type=int, help='Max number of iterations without improvement.')
    parser.add_argument('--mu', type=float, help='Lagrangian multiplier.')
    parser.add_argument('--selection_pressure', type=float, help='Selection pressure for rank selection.')
    parser.add_argument('--feasible_only', action='store_true')
    parser.add_argument('--no-feasible_only', dest='feasible_only', action='store_false')
    parser.add_argument('--feasible_only_init_pop', action='store_true')
    parser.add_argument('--no-feasible_only_init_pop', dest='feasible_only_init_pop', action='store_false')
    parser.add_argument('--run_both', action='store_true')
    parser.add_argument('--num_runs', type=int, help='Number of runs', default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.add_argument('--crossover_function', type=str, help='Crossover function to be used.', default='bnc_pso')
    args = parser.parse_args()

    PATH = define_path_to_save(args.data, args.feasible_only)



    print("Using crossover function:", args.crossover_function)

    # Parameters
    max_iter = args.max_iter
    mutation_rate = args.mutation_rate
    crossover_rate = args.crossover_rate
    pop_size = args.popSize
    patience = args.patience
    feasible_only = args.feasible_only
    mu = args.mu
    selction_pressure = args.selection_pressure
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

        # Create initial population
        nodes = data.columns
        nodes = list(nodes)
        print("Creating initial population")
        population = create_population(pop_size, nodes, data, feasible_only=args.feasible_only_init_pop)

        # Evolve population
        print("Evolving population")
        best_graph, bic_history, population = MAGA(population, max_iter, mutation_rate, 
                                                          crossover_rate, patience, selction_pressure, goal_bic, args.crossover_function, feasible_only, verbose=args.verbose)
        print("best graph returned BIC")
        print(best_graph.bic)
        print("first ind bic")
        print(population[0].bic)
        print("bic of all inds")
        for ind in population:
            print(ind.compute_bic(data))
        
        best_graph = best_graph.individual_to_digraph()
        # Compute metrics
        compute_metrics(run_both=args.run_both, alg='bic')

        print("Algorithm ended. Computing Results")
        end = time.time()
        time_vector.append(end-start)


    print("Mean time:", statistics.mean(time_vector))