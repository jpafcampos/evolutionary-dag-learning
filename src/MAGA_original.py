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
from eval_metrics import *
from ga_operators import *
from loaders import *
import matplotlib.pyplot as plt
import argparse
import copy
import os


def MAGA(population, max_eval_bic, Pm_min, Pm_max, Po, Pc_min, Pc_max, t_max, goal_bic, sL, sPm, sGen, file, self_learn, feasible_only, verbose=False):

    best_bic = float('inf')
    best_graph = None
    best_pos = 0
    iteration = 0
    t = 0
    num_eval_bic = 0
    reached_goal = False
    while (best_bic > goal_bic + 0.00000001) and (num_eval_bic < max_eval_bic):
        if verbose:
            print('Iteration:', iteration)
        Pc=(Pc_min-t*(Pc_min-Pc_max)/t_max)
        Pm=(Pm_min-t*(Pm_min-Pm_max)/t_max)
        #Pc = Pc_min
        #Pm = Pm_max
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
                    num_eval_bic += 2
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
                num_eval_bic += 1
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

        if self_learn:
            best_graph, num_eval_bic_sl = self_learning(sL, best_graph, sPm, Po, sGen, data, feasible_only)
            num_eval_bic += num_eval_bic_sl

        population[best_pos] = best_graph
        best_bic = best_graph.bic

        write_metrics_history(file, best_graph, goal_bic, ground_truth, data)
        iteration += 1



    if best_bic <= goal_bic + 0.00000001:
        reached_goal = True

    return best_graph, num_eval_bic, population, reached_goal

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

    filename = PATH +f'MAGA_all_runs_results_{args.data}.csv'

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.data, args.type_exp, args.Pm_min, args.Pm_max, args.Pc_min, args.Pc_max, args.sGen, args.sPm,
                          end-start, goal_bic, num_eval_bic, self_learn, reached_goal, best_score_bic, best_score_ls, SLF, TLF])


if __name__ == '__main__':

    # Load arguments
    parser = argparse.ArgumentParser(description='Evolutionary Algorithm for BN structural learning.')
    parser.add_argument('--data', type=str, help='Name of the dataset.')
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
    parser.add_argument('--self_learn', type=int, help='Wether to use self learning or not.', default=1)
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
    
    PATH = '/home/joao/Desktop/UFMG/PhD/code/EA-DAG/results/MAGA/' + args.data + '/'
    # if PATH does not exist, change it to the path in the server
    if not os.path.exists(PATH):
        PATH = '/home/joaocampos/phd_code/evolutionary-dag-learning/results/MAGA/' + args.data + '/'
    if not os.path.exists(PATH):
        PATH = '/home/bessani/phd_code/evolutionary-dag-learning/results/MAGA/' + args.data + '/'

    # Parameters
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
    self_learn = True if args.self_learn == 1 else False

    t_max = 10

    randomized = True if args.random == 1 else False
    
    sample_size, max_bic_eval = load_samplesize_num_evals(args.data, args.type_exp)
    
    time_vector = []
    for i in range(n_runs):
        print(f'Run {i+1}/{n_runs}')
        
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

        # Create file to save results
        file = PATH +f'run_{i+1}_results_{args.data}.csv'
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['BIC - Goal', 'BIC', 'F1 Score', 'Accuracy', 'Precision', 'Recall', 'SHD', 'SLF', 'TLF'])


        # Create initial population
        nodes = data.columns
        nodes = list(nodes)
        print("Creating initial population")
        population = create_MAGA_population(L_size, nodes, data, feasible_only=args.feasible_only_init_pop)

        # measure time
        start = time.time()
        # Evolve population
        print("Evolving population")
        #def MAGA(population, max_iter, Pm_min, Pm_max, Po, Pc_min, Pc_max, t_max, goal_bic, sL, sPm, sGen, feasible_only, verbose=False):
        best_graph, num_eval_bic, population, reached_goal = MAGA(population, max_bic_eval, Pm_min, Pm_max, Po, Pc_min,
                                                   Pc_max, t_max, goal_bic, sL, args.sPm, args.sGen, file, self_learn, feasible_only, verbose=args.verbose)
        print("best graph returned BIC")
        print(best_graph.bic)
        

        print("Algorithm ended. Computing Results")
        end = time.time()
        time_vector.append(end-start)

        best_graph = best_graph.individual_to_digraph()
        #save best graph
        nx.write_gml(best_graph, PATH + f'best_graph_{args.data}_run_{i+1}.gml')
        # Compute metrics
        compute_metrics()

    print("Mean time:", statistics.mean(time_vector))