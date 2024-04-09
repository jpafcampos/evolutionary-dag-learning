# -*- coding: utf-8 -*-
"""
Created on nov 2023

@author: João Pedro Campos
"""

import random
#import matplotlib.pyplot as plt
import networkx as nx
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
from ga_operators import *
from loaders import *
import matplotlib.pyplot as plt
import argparse


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
    scores = [i.compute_bic(data) for i in population]
    rank = np.argsort(scores)
    best_n = rank[:n]
    return list(population[i] for i in best_n)

def select_parents_by_rank(population, num_parents, data):
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
    rank_probs = [1/(i+1) for i in range(len(population))]
    rank_probs = [p/sum(rank_probs) for p in rank_probs]
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
        best = select_best(tournament, 1, BIC, data)[0]
        parents.append(best)
    return parents



def evolve_DAGs (population, max_iter, mutation_rate, crossover_rate, patience, feasible_only=True, verbose=False):
    '''
    Evolve a population of DAGs using a genetic algorithm.
    
    Parameters:
    -----------
    population : list
        List containing the initial population of DAGs.
    max_iter : int
        Maximum number of iterations.
    mutation_rate : float
        Probability of mutation.
    crossover_rate : float
        Probability of crossover.
    nodes: list
        List of nodes that should be present in the resulting graphs.
    patience: int
        Number of iterations without improvement before stopping.
    feasible_only : bool, optional
        If True, only feasible graphs (DAGs) will be returned. Default is True.
    
    Returns:
    --------
    list
        List containing the final population of DAGs.
    '''
    # Initialize auxiliary variables
    best_score = float('inf')
    best_graph = None
    bic_history = []
    count_patience = 0
    pop_size = len(population)
    # Iterate
    for i in range(max_iter):
        # Select parents
        if verbose:
            print('Iteration:', i)
        parents = select_parents_by_rank(population, len(population)//2, data)
        # Crossover
        if verbose:
            print('Crossover')
        children = []
        for j in range(0, len(parents), 2):
            child = bnc_pso_crossover(parents[j], parents[j+1], feasible_only)
            children.append(child)
        # Mutation
        if verbose:
            print('Mutation')
        for child in children:
            if random.random() < mutation_rate:
                child.bit_flip_mutation(feasible_only)
        # Evaluate children
        if verbose:
            print('Evaluating children')
        for child in children:
            child.compute_bic(data)
        # Select survivors
        if verbose:
            print('Selecting survivors')
        population = select_best(population + children, pop_size, data)
        # Update best graph
        if verbose:
            print('Updating best graph')
        best = select_best(population, 1, data)[0]
        if best.compute_bic(data) < best_score:
            best_score = best.compute_bic(data)
            best_graph = best
            count_patience = 0
        else:
            if verbose:
                print('No improvement for iteration', i)
            count_patience += 1
        bic_history.append(best_score)
        # Check patience
        if count_patience >= patience:
            break

    return best_graph, bic_history, population


def compute_metrics(run_both = False, alg = 'bic'):
    best_score_bic = BIC(best_graph, data)
    best_score_ls = lagrangian(best_graph, data)

    # Hamming distance
    SLF, TLF = learning_factors(ground_truth, best_graph)

    # Save results
    print('Saving results')

    if run_both:
        filename = PATH +f'results_{args.data}_both.csv'
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([args.data, args.sample_size, args.max_iter, args.mutation_rate, 
                            args.crossover_rate, args.popSize, args.patience, args.density_factor, alg, 
                            args.feasible_only_init_pop, best_score_bic, best_score_ls, SLF, TLF])
    
    else:
        filename = PATH +f'results_{args.data}_{"feasible" if args.feasible_only else "infeasible"}.csv'
    
        with open(filename, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([args.data, args.sample_size, args.max_iter, args.mutation_rate, 
                            args.crossover_rate, args.popSize, args.patience, args.density_factor, args.feasible_only, 
                            args.feasible_only_init_pop, best_score_bic, best_score_ls, SLF, TLF])


    # Save best graph
    #nx.write_graphml(best_graph, 'best_graph.graphml')

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
    parser.add_argument('--feasible_only', action='store_true')
    parser.add_argument('--no-feasible_only', dest='feasible_only', action='store_false')
    parser.add_argument('--feasible_only_init_pop', action='store_true')
    parser.add_argument('--no-feasible_only_init_pop', dest='feasible_only_init_pop', action='store_false')
    parser.add_argument('--run_both', action='store_true')
    parser.add_argument('--num_runs', type=int, help='Number of runs', default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    args = parser.parse_args()

    PATH = define_path_to_save(args.data, args.feasible_only)

    if args.data == 'asia':
        data = load_asia_data(sample_size=args.sample_size)
        ground_truth = load_gt_network_asia()
        print('Asia data and ground truth loaded')
    elif args.data == 'adult':
        data = load_adult_data(sample_size=args.sample_size)
        ground_truth = load_gt_adult()
        print('Adult data and grount truth loaded')
    else:
        # dataset not available
        print('Dataset not available.')
        exit(1)

    print(data.head())
    # Parameters
    max_iter = args.max_iter
    mutation_rate = args.mutation_rate
    crossover_rate = args.crossover_rate
    pop_size = args.popSize
    patience = args.patience
    feasible_only = args.feasible_only
    mu = args.mu


    n_runs = args.num_runs

    time_vector = []
    for i in range(n_runs):
        # measure time
        start = time.time()
        # Create initial population
        nodes = data.columns
        nodes = list(nodes)
        population = create_population(pop_size, nodes, data, feasible_only=args.feasible_only_init_pop)
        # Evolve population
        best_graph, bic_history, population = evolve_DAGs(population, max_iter, mutation_rate, crossover_rate, patience, feasible_only, verbose=args.verbose)

        # Draw BIC history
        plt.figure()
        plt.plot(bic_history)
        plt.savefig(PATH + 'bic_history.png')

        # Draw best graph
        plt.figure()
        nx.draw(best_graph.individual_to_digraph(), with_labels=True)
        plt.savefig(PATH + 'best_graph.png')


        print("Algorithm ended. Computing Results")
        end = time.time()
        time_vector.append(end-start)


    print("Mean time:", statistics.mean(time_vector))