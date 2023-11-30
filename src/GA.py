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


def select_best(population, n, score_function, data):
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
    scores = [score_function(i, data) for i in population]
    rank = np.argsort(scores)
    best_n = rank[:n]
    return list(population[i] for i in best_n)


def evolve_DAGs (population, max_iter, mutation_rate, crossover_rate, nodes, patience, feasible_only=True):
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
    ls_history = []
    count_patience = 0
    # Iterate
    for i in range(max_iter):
        # Select parents
        parents = select_best(population, len(population)//2, BIC, data)
        # Crossover
        children = []
        for j in range(0, len(parents), 2):
            if random.random() < crossover_rate:
                child1, child2 = crossover(parents[j], parents[j+1], nodes, feasible_only)
                children.append(child1)
                children.append(child2)
            else:
                children.append(parents[j])
                children.append(parents[j+1])
        # Mutation
        for j in range(len(children)):
            if random.random() < mutation_rate:
                children[j] = mutation(children[j], nodes, feasible_only)
        # Selection
        population = parents + children
        # Update best graph
        best_current_score = float('inf')
        score_vector_bic = []
        score_vector_ls = []
        for j in population:
            score = BIC(j, data)
            score_vector_bic.append(score)
            score_ls = lagrangian(j, data, args.mu)
            score_vector_ls.append(score_ls)
            if score < best_score:
                best_current_score = score
                best_graph = j

        if best_current_score < best_score:
            best_score = best_current_score
            count_patience = 0
        else:
            count_patience += 1

        if count_patience >= patience:
            break
        
        # Update BIC history
        bic_history.append(score_vector_bic)
        ls_history.append(score_vector_ls)
        print('Iteration', i, 'score', best_score)
    

    return best_graph, bic_history, ls_history, population

def evolve_infeasible(population, max_iter, mutation_rate, crossover_rate, nodes, patience, mu, feasible_only=False):
    '''
    Evolve a population of graphs, allowing infeasible graphs.
    
    Parameters:
    -----------
    population : list
        List containing the initial population of graphs.
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
        If True, only feasible graphs (DAGs) will be returned. Default is False.
    
    Returns:
    --------
    list
        List containing the final population of graphs.   
    '''
    # Initialize auxiliary variables
    best_score = float('inf')
    best_graph = None
    least_squares_history = []
    dagness_history = []
    bic_history = []
    # Iterate
    for i in range(max_iter):
        # Select parents
        def lagrangian_(G, data):
            W = nx.to_numpy_array(G)
            W = np.array(W)
            loss = least_squares(G, data)
            h_W = h(W)
            lagrangian = loss + mu * h_W
            return lagrangian
        
        parents = select_best(population, len(population)//2, lagrangian_, data)
        # Crossover
        children = []
        for j in range(0, len(parents), 2):
            if random.random() < crossover_rate:
                child1, child2 = crossover(parents[j], parents[j+1], nodes, feasible_only)
                children.append(child1)
                children.append(child2)
            else:
                children.append(parents[j])
                children.append(parents[j+1])
        # Mutation
        for j in range(len(children)):
            if random.random() < mutation_rate:
                children[j] = mutation(children[j], nodes, feasible_only)
        # Selection
        population = parents + children
        # Update best graph
        best_current_score = float('inf')
        score_vector = []
        for j in population:
            score = lagrangian(j, data)
            score_vector.append(score)
            dagness_of_best = dagness(j)
            if score < best_score:
                best_current_score = score
                best_graph = j

        if best_current_score < best_score:
            best_score = best_current_score
            count_patience = 0
        else:
            count_patience += 1

        if count_patience >= patience:
            break
        
        # Update history
        least_squares_history.append(score_vector)
        dagness_history.append(dagness_of_best)
        if dagness_of_best == 0:
            bic_history.append([BIC(best_graph, data)])

        print('Iteration', i, 'score', best_score)

    return best_graph, least_squares_history, dagness_history, bic_history, population



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
    parser.add_argument('--density_factor', type=float, help='Density factor.')
    parser.add_argument('--mu', type=float, help='Lagrangian multiplier.')
    parser.add_argument('--feasible_only', action='store_true')
    parser.add_argument('--no-feasible_only', dest='feasible_only', action='store_false')
    parser.add_argument('--feasible_only_init_pop', action='store_true')
    parser.add_argument('--no-feasible_only_init_pop', dest='feasible_only_init_pop', action='store_false')
    args = parser.parse_args()

    PATH = define_path_to_save(args.data, args.feasible_only)

    if args.data == 'asia':
        data = load_asia_data(sample_size=args.sample_size)
        ground_truth = load_gt_asia()
        print('Asia data loaded')
    elif args.data == 'adult':
        data = load_adult_data(sample_size=args.sample_size)
        ground_truth = load_gt_adult()
        print('Adult data loaded')
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
    density_factor = args.density_factor
    feasible_only = args.feasible_only
    mu = args.mu


    n_runs = 1
    time_vector = []
    for i in range(n_runs):
        # measure time
        start = time.time()
        # Create initial population
        nodes = data.columns
        nodes = list(nodes)
        population = createPopulation(pop_size, nodes, density_factor, args.feasible_only_init_pop)

        # Evolve population
        if feasible_only:
            print('Feasible only')
            best_graph, bic_hist, ls_hist, evolved_pop = evolve_DAGs(population, max_iter, mutation_rate, crossover_rate, nodes, patience, feasible_only)
            # Save plots
            save_feasible_only_plots(PATH, bic_hist, ls_hist)

        else:
            print('Infeasible allowed')
            best_graph, ls_hist, dagness_hist, bic_hist, evolved_pop = evolve_infeasible(population, max_iter, mutation_rate, crossover_rate, nodes, patience, mu, feasible_only)
            # Save plots
            save_infeasible_plots(PATH, ls_hist, dagness_hist, bic_hist)


        print("Algorithm ended. Computing Results")
        end = time.time()
        time_vector.append(end-start)

        # Compute metrics

        best_score_bic = BIC(best_graph, data)
        best_score_ls = lagrangian(best_graph, data)

        # Hamming distance
        SLF, TLF = learning_factors(ground_truth, best_graph)

        # Save results
        print('Saving results')
        
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
        plt.savefig(PATH + 'best_graph.png')


    print("Mean time:", statistics.mean(time_vector))