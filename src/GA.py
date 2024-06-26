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
import time
import statistics
import bnlearn as bn
from pgmpy.metrics import structure_score
from pgmpy.models import BayesianNetwork
import gc
from utils import *
from eval_metrics import *
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



def evolve_DAGs (population, max_bic_eval, mutation_rate, crossover_rate, selection_pressure, goal_bic, file, feasible_only=True, verbose=False):
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
    num_bic_eval = 0
    pop_size = len(population)
    reached_goal = False
    local_minimum = False

    crossover_operator = bnc_pso_crossover

    # Iterate
    while num_bic_eval < max_bic_eval:
        # Select parents
        if verbose:
            print('Iteration:', i)
        parents = select_parents_by_rank(population, len(population), selection_pressure, data)
        # Crossover
        if verbose:
            print('Crossover')
        children = []
        for j in range(0, len(parents), 2):
            if random.random() < crossover_rate:
                c1, c2 = crossover_operator(parents[j], parents[j+1], feasible_only)
        
            else:
                c1 = copy.deepcopy(parents[j])
                c2 = copy.deepcopy(parents[j+1])
            children.append(c1)
            children.append(c2)

        # Mutation
        new_children = []
        for child in children:
            if random.random() < mutation_rate:
                child = mutation(child)
            new_children.append(child)
        children = new_children

        # Evaluate children
        if verbose:
            print('Evaluating children')
        for child in children:
            child.compute_bic(data)
            num_bic_eval += 1

        # Select survivors by tournament or rank
        if verbose:
            print('Selecting survivors')
        population = select_best(population + children, pop_size, data)

        # Print standard deviation of population scores
        if verbose:
            print('Standard deviation of population scores:', statistics.stdev([i.bic for i in population]))
        # Update best graph
        if verbose:
            print('Updating best graph')
        
        # Sort population to take the best
        #population = select_best(population, len(population), data)
        best = population[0]
        #bic_history.append(best.bic)
        write_metrics_history(file, best, goal_bic, ground_truth, data)

        # Check if goal is reached
        eps = 1e-6
        if best.bic < goal_bic + eps:
            best_graph = best
            reached_goal = True
            break
        
        if best.bic < best_score:
            best_score = best.bic
            best_graph = best
        else:
            if verbose:
                print('No improvement for iteration', i)

        

    return best_graph, population, reached_goal, num_bic_eval


def write_metrics_history(file, individual, goal_bic, ground_truth, data):
    bic = individual.compute_bic(data) if individual.bic == None else individual.bic
    f1score, accuracy, precision, recall, SHD, SLF, TLF = individual.compute_accuracy_metrics(ground_truth)

    with open(file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([bic-goal_bic, bic, f1score, accuracy, precision, recall, SHD, SLF, TLF])

def compute_metrics():
    best_score_bic = BIC(best_graph, data)
    best_score_ls = lagrangian(best_graph, data)

    # Hamming distance
    SLF, TLF = learning_factors(best_graph, ground_truth)
    # Print Results
    print('Best graph:', best_graph.edges())
    print('Best score BIC:', best_score_bic)
    print('Best score LS:', best_score_ls)
    print('Structure Learning Factor:', SLF)
    print('Topology Learning Factor:', TLF)

    # Save results
    print('Saving results')

    filename = PATH +f'GA_all_runs_results_{args.data}.csv'

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.data, args.type_exp, goal_bic, args.mutation_rate, 
                            args.crossover_rate, args.popSize, args.feasible_only, 
                            args.feasible_only_init_pop, args.selection_pressure, reached_goal, num_bic_eval, end-start, best_score_bic, best_score_ls, SLF, TLF])

if __name__ == '__main__':

    # Load arguments
    parser = argparse.ArgumentParser(description='Evolutionary Algorithm for BN structural learning.')
    parser.add_argument('--data', type=str, help='Name of the dataset.')
    parser.add_argument('--mutation_rate', type=float, help='Probability of mutation.')
    parser.add_argument('--crossover_rate', type=float, help='Probability of crossover.')
    parser.add_argument('--popSize', type=int, help='Population size.')
    parser.add_argument('--mu', type=float, help='Lagrangian multiplier.')
    parser.add_argument('--selection_pressure', type=float, help='Selection pressure for rank selection.')
    parser.add_argument('--feasible_only', action='store_true')
    parser.add_argument('--no-feasible_only', dest='feasible_only', action='store_false')
    parser.add_argument('--feasible_only_init_pop', action='store_true')
    parser.add_argument('--no-feasible_only_init_pop', dest='feasible_only_init_pop', action='store_false')
    parser.add_argument('--num_runs', type=int, help='Number of runs', default=1)
    parser.add_argument('--random', type=int, help='Wether the sample is random or not', default=1)
    parser.add_argument('--type_exp', type=int, help='Type of experiment (ratio parameters)', default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    args = parser.parse_args()

    #PATH = '/home/joao/Desktop/UFMG/PhD/code/EA-DAG/results/GA/' + args.data + '/' 
    PATH = '/home/joaocampos/phd_code/evolutionary-dag-learning/results/GA/' + args.data + '/'
    # Parameters
    mutation_rate = args.mutation_rate
    crossover_rate = args.crossover_rate
    pop_size = args.popSize
    feasible_only = args.feasible_only
    mu = args.mu
    selection_pressure = args.selection_pressure
    n_runs = args.num_runs

    randomized = True if args.random == 1 else False
    
    sample_size, max_bic_eval = load_samplesize_num_evals(args.data, args.type_exp)
    
    

    time_vector = []
    for i in range(n_runs):
        print(f'Run {i+1}')

        # Create new numbered csv file for each run
        filename_run = PATH +f'run_{i+1}_results_{args.data}.csv'

        # Load data
        if args.data == 'asia':
            data = load_asia_data(sample_size=sample_size, randomized=randomized)
            ground_truth = load_gt_network_asia()
            print('Asia data and ground truth loaded')
        elif args.data == 'child':
            data = load_child_data(sample_size=sample_size, randomized=randomized)
            ground_truth = load_gt_network_child()
            print('Child data and ground truth loaded')
        elif args.data == 'insurance':
            data = load_insurance_data(sample_size=sample_size, randomized=randomized)
            ground_truth = load_gt_network_insurance()
            print('Insurance data and ground truth loaded')
        else:
            print('Data not recognized')
            break


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
        best_graph, population, reached_goal, num_bic_eval = evolve_DAGs(population, max_bic_eval, mutation_rate, 
                                                          crossover_rate, selection_pressure, goal_bic, filename_run, feasible_only, verbose=args.verbose)
        print("best graph returned BIC")
        print(best_graph.bic)
        print("first ind bic")
        print(population[0].bic)
        print("bic of all inds")
        for ind in population:
            print(ind.compute_bic(data))

        print("Algorithm ended. Computing Results")
        end = time.time()
        time_vector.append(end-start)
        best_graph = best_graph.individual_to_digraph()

        #save best graph
        nx.write_gml(best_graph, PATH + f'best_graph_{args.data}_run_{i+1}.gml')

        compute_metrics()

    print("Mean time:", statistics.mean(time_vector))