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

def BNC_PSO(particles, goalscore, data, evaluations, w_start, w_end, c1_start, c1_end, c2_start, c2_end, max_iter, feasible_only=True):
    
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
    return best_particle, best_score, ev, bests, medias


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

    PATH = '/home/joao/Desktop/UFMG/PhD/code/EA-DAG/results/BNCPSO/' + args.data + '/' 



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

        evaluations = 200
        w_start = 0.95
        w_end = 0.4
        c1_start = 0.82
        c1_end = 0.5
        c2_start = 0.4
        c2_end = 0.83
        max_iter = 200

        #def BNC_PSO(particles, goalscore, data, evaluations, w_start, w_end, c1_start, c1_end, c2_start, c2_end, max_iter, feasible_only=True):
        best_graph, best_score, ev, bests, medias = BNC_PSO(population, goal_bic, data, evaluations, w_start, w_end, c1_start, c1_end, c2_start, c2_end, max_iter, feasible_only=feasible_only)
        print("best graph returned BIC")
        print(best_graph.bic)

        best_graph = best_graph.individual_to_digraph()
        # Compute metrics
        compute_metrics(run_both=args.run_both, alg='bic')

        print("Algorithm ended. Computing Results")
        end = time.time()
        time_vector.append(end-start)


    print("Mean time:", statistics.mean(time_vector))