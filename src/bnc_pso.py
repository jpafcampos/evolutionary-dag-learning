# -*- coding: utf-8 -*-
"""
Created on may 2024

@author: João Pedro Campos
adapted from Itallo Machado https://github.com/ItalloMachado/Mestrado/tree/main
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

def BNC_PSO(particles, goalscore, data, evaluations, w_start, w_end, c1_start, c1_end, c2_start, c2_end, max_bic_eval, feasible_only=True):
    
    best_score = particles[0].bic if particles[0].bic != None else particles[0].compute_bic(data)
    media_score = 0
    best_particle = particles[0]
    for i in range(len(particles)):  
        if particles[i].bic < best_score:
            best_score = particles[i].bic if particles[i].bic != None else particles[i].compute_bic(data)
            best_particle = particles[i]

    ev = 0
    bests = []
    medias = []
    iter = 0
    while  (iter < max_iter) and best_score >(goalscore+0.00000001):
        print("iter: {}".format(iter))
        #print("ev: {}".format(ev))
        #print("aux_bic: {}".format(aux_bic))
        w = w_start - ((w_start-w_end)/evaluations)*ev
        c1 = c1_start - ((c1_start - c1_end)/evaluations)*ev
        c2 = c2_start - ((c2_start - c2_end)/evaluations)*ev
        new_particles = []
        
        for i in range(len(particles)):
            
            w_rand = random.random()
            particle_before_mutation = copy.deepcopy(particles[i])
            
            if w_rand < w:
                aux_particle = mutation(particles[i], feasible_only=feasible_only)
                aux_particle.compute_bic(data)               
            else:
                aux_particle = particle_before_mutation

            c1_rand = random.random()
            
            if c1_rand<c1:
                rand_index = random.randint(0,len(particles)-1)
                while rand_index  ==  i:
                    rand_index = random.randint(0,len(particles)-1)
                    rand_particle = particles[rand_index]
                    child1, child2 = bnc_pso_crossover(aux_particle, rand_particle, feasible_only)
                    child1.compute_bic(data)
                    child2.compute_bic(data)
                    aux_particle = child1 if child1.bic < child2.bic else child2

            c2_rand = random.random()
            
            if c2_rand < c2:
                if best_particle.genes !=  aux_particle.genes:
                    child1, child2 = bnc_pso_crossover(aux_particle, best_particle, feasible_only)
                    child1.compute_bic(data)
                    child2.compute_bic(data)
                    aux_particle = child1 if child1.bic < child2.bic else child2

            new_particles.append(aux_particle)
        media_score = 0

        for i in range(len(new_particles)):
            if new_particles[i].bic < best_score:
                best_score = copy.deepcopy(new_particles[i].bic)
                best_particle = copy.deepcopy(new_particles[i])

        for i in range(len(particles)):
           if new_particles[i].bic <= particles[i].bic:
                particles[i] = copy.deepcopy(new_particles[i])
    
        for i in range(len(particles)):
            media_score += particles[i].bic
        bests.append(best_score)
        medias.append(media_score/len(particles))
        ev += 1
        iter += 1
    return best_particle, particles, num_bic_eval, reached_goal

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
                            args.feasible_only_init_pop, reached_goal, num_bic_eval, end-start, best_score_bic, best_score_ls, SLF, TLF])


if __name__ == '__main__':

    # Load arguments
    parser = argparse.ArgumentParser(description='Evolutionary Algorithm for BN structural learning.')
    parser.add_argument('--data', type=str, help='Name of the dataset.')
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
    parser.add_argument('--random', type=int, help='Wether the sample is random or not', default=1)
    parser.add_argument('--type_exp', type=int, help='Type of experiment (ratio parameters)', default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.add_argument('--crossover_function', type=str, help='Crossover function to be used.', default='bnc_pso')
    args = parser.parse_args()

    #PATH = '/home/joao/Desktop/UFMG/PhD/code/EA-DAG/results/GA/' + args.data + '/' 
    PATH = '/home/joaocampos/phd_code/evolutionary-dag-learning/results/GA/' + args.data + '/'



    print("Using crossover function:", args.crossover_function)

    # Parameters
    mutation_rate = args.mutation_rate
    crossover_rate = args.crossover_rate
    pop_size = args.popSize
    patience = args.patience
    feasible_only = args.feasible_only
    mu = args.mu
    n_runs = args.num_runs

    randomized = True if args.random == 1 else False
    
    sample_size, max_bic_eval = load_samplesize_num_evals(args.data, args.type_exp)

    time_vector = []
    for i in range(n_runs):

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

        evaluations = 200
        w_start = 0.95
        w_end = 0.4
        c1_start = 0.82
        c1_end = 0.5
        c2_start = 0.4
        c2_end = 0.83
        max_iter = 200

        #def BNC_PSO(particles, goalscore, data, evaluations, w_start, w_end, c1_start, c1_end, c2_start, c2_end, max_iter, feasible_only=True):
        best_particle, particles, num_bic_eval, reached_goal = BNC_PSO(population, goal_bic, data, evaluations, w_start, w_end, c1_start, c1_end, c2_start, c2_end, max_bic_eval, feasible_only=feasible_only)
        print("best graph returned BIC")
        print(best_graph.bic)

        best_graph = best_graph.individual_to_digraph()
        # Compute metrics
        compute_metrics(run_both=args.run_both, alg='bic')

        print("Algorithm ended. Computing Results")
        end = time.time()
        time_vector.append(end-start)


    print("Mean time:", statistics.mean(time_vector))