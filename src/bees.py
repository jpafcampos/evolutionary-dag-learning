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
from ga_operators import *
from loaders import *
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy
from pgmpy.estimators import BDeuScore, K2Score, BicScore
from pgmpy.models import BayesianModel

def bic_score(dag, data):
    nodes = data.columns
    connected_nodes = get_connected_nodes(dag)
    missing_nodes = len(nodes) - len(connected_nodes)
    data = data[connected_nodes]
    bic = BicScore(data)
    if len(dag.edges()) == 0:
        return float('inf')
    model = BayesianNetwork(list(dag.edges()))
    score = np.abs(bic.score(model))
    return score + missing_nodes * 1000

def get_connected_nodes(graph):
    # Initialize an empty set to store nodes with at least one edge
    nodes_with_edges = set()

    # Iterate through the edges to collect nodes
    for edge in graph.edges():
        nodes_with_edges.update(edge)

    # Convert the set to a list (if needed)
    nodes_with_edges_list = list(nodes_with_edges) 

    return nodes_with_edges_list

def roleta(prob):
    """ Função que escolhe um indivíduo através de uma roleta.
    entrada: prob =  vetor de probabilidades de  cada indivíduo
    Saída: escolhido =  indivíduo escolhido."""   
    soma_prob = sum(prob)
    prob = [prob[i]/soma_prob for i in range(len(prob))]
    rand_num = random.random()
    aux_num = 0
    i = 0
    while aux_num < rand_num:
        aux_num += prob[i]
        i += 1
    escolhido = i-1    
    return escolhido


def Probabilistic_transition_rule_v2_new(pheromones_matrix, bee, nodes, alpha, beta, score_matrix, score_old):
   
    bee_aux = [deepcopy(bee),deepcopy(bee)]
    eta_soma = 0
    eta_aux = []
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if not bee_aux[0][0].has_edge(nodes[i],nodes[j]) or not bee_aux[0][0].has_edge(nodes[j],nodes[i]): 
                bee_aux[0][0].add_edge(nodes[i],nodes[j])
                if nx.is_directed_acyclic_graph(bee_aux[0][0]):
                    cost = score_matrix[i][j]
                    aux = ((cost-score_old)**beta)*pheromones_matrix[i][j]**alpha
                    eta_soma += aux
                    eta_aux.append([nodes[i],nodes[j],aux])
                else:
                    eta_aux.append([nodes[i],nodes[j],-9999999])
                bee_aux[0][0].remove_edge(nodes[i],nodes[j])
    return eta_aux, eta_soma


def Probabilistic_transition_rule_v2_aux(eta_aux, eta_soma, q0):
    q = random.random()
    if q<= q0:
        max_eta = eta_aux[0]
        for i in range(len(eta_aux)):
            if max_eta[2]<eta_aux[i][2]:
                max_eta = eta_aux[i]
        return max_eta
    else:
        prob = []
        for i in range(len(eta_aux)):
            if eta_aux[i][2]  ==  -9999999:
                prob.append(0)
            else:
                prob.append(eta_aux[i][2]/eta_soma)                
        esc = roleta(prob)
        return eta_aux[esc]         
    return
   

def neighbor_KG_search_v2(bee,score_matriz,nodes,data):
    bee_aux = [deepcopy(bee),deepcopy(bee)]
    new_bee_add = [[0,0],999999]
    new_bee_remove = [[0,0],0]
    for i in nodes:
        for j in nodes:         
            if not bee_aux[0][0].has_edge(i,j):                
                bee_aux[0][0].add_edge(i,j)
                if nx.is_directed_acyclic_graph(bee_aux[0][0]):
                    ix_i = nodes.index(i)
                    ix_j = nodes.index(j)
                    cost = score_matriz[ix_i][ix_j] 
                    cost = abs(cost)
                    if new_bee_add[1]>cost:
                        new_bee_add[0] = [i,j]
                        new_bee_add[1] = cost
                bee_aux[0][0].remove_edge(i,j)
            if bee_aux[0][0].has_edge(i,j):
                ix_i = nodes.index(i)
                ix_j = nodes.index(j)
                cost = score_matriz[ix_i][ix_j] 
                cost = abs(cost)
                if new_bee_remove[1]<cost:
                    new_bee_remove[0] = [i,j]
                    new_bee_remove[1] = cost
    bee_add = bee_aux[0][0].add_edge(new_bee_add[0][0],new_bee_add[0][1])
    [cost_add] = bic_score(bee_aux[0][0],data)
    cost_add = abs(cost_add)
    bee_aux[0][0].remove_edge(new_bee_add[0][0],new_bee_add[0][1])
    bee_add = bee_aux[0][0].add_edge(new_bee_add[0][1],new_bee_add[0][0])
    if nx.is_directed_acyclic_graph(bee_aux[0][0]):
        [cost_add2] = bic_score(bee_aux[0][0],data)
        cost_add2 = abs(cost_add2)
        bee_aux[0][0].remove_edge(new_bee_add[0][1],new_bee_add[0][0])
        if cost_add>cost_add2:
            cost_add = cost_add2
            aux = new_bee_add[0][1]
            new_bee_add[0][1] = new_bee_add[0][0]
            new_bee_add[0][0] = aux        
    else:
        bee_add = bee_aux[0][0].remove_edge(new_bee_add[0][1],new_bee_add[0][0])  
    bee_remove = bee_aux[0][0].remove_edge(new_bee_remove[0][0],new_bee_remove[0][1])
    [cost_remove] = bic_score(bee_aux[0][0],data)
    cost_remove = abs(cost_remove)
    if cost_add <= bee[1] and cost_add <= cost_remove:
        return bee_add,cost_add
    elif cost_remove <= bee[1] and cost_add > cost_remove:
        return bee_remove,cost_remove


def neighbor_search(bee,nodes,data):
    bee_aux = [deepcopy(bee),deepcopy(bee),deepcopy(bee),deepcopy(bee)] # 1 add 2 remove 3 reverse 4 move
    old_bee = deepcopy(bee)
    add = True
    cont = 0
    cont_max = 100
    while (add and cont<cont_max):
        cont += 1
        node1 = random.randint(0,len(nodes)-1)
        node2 = node1
        while (node1  ==  node2):
            node1 = random.randint(0,len(nodes)-1)
            node2 = random.randint(0,len(nodes)-1)
        if not bee_aux[0][0].has_edge(nodes[node1],nodes[node2]):
            bee_aux[0][0].add_edge(nodes[node1],nodes[node2])
            search_dag(bee_aux[0][0],nodes[node1],nodes[node2])
            add = False
    remove = True
    while(remove):
        node1 = random.randint(0,len(nodes)-1)
        node2 = node1
        while(node1  ==  node2):
            node2 = random.randint(0,len(nodes)-1)
        if bee_aux[1][0].has_edge(nodes[node1],nodes[node2]):
            bee_aux[1][0].remove_edge(nodes[node1],nodes[node2])
            remove = False
    reverse = True
    while(reverse):
        node1 = random.randint(0,len(nodes)-1)
        node2 = node1
        while(node1  ==  node2):
            node2 =  random.randint(0,len(nodes)-1)
        if bee_aux[2][0].has_edge(nodes[node1],nodes[node2]):
            bee_aux[2][0].remove_edge(nodes[node1],nodes[node2])
            bee_aux[2][0].add_edge(nodes[node2],nodes[node1])
            search_dag(bee_aux[2][0],nodes[node2],nodes[node1])        
            reverse = False
    move = True
    cont = 0
    while(move and cont<300):
        cont += 1
        node1 =  random.randint(0,len(nodes)-1)
        node2 =  node1
        while (node1  ==  node2):
            node2 =  random.randint(0,len(nodes)-1)
        parents_node1 = list(bee_aux[3][0].predecessors(nodes[node1]))
        parents_node2 = list(bee_aux[3][0].predecessors(nodes[node2]))
        if parents_node1 and parents_node2:  
            parent1 =  random.randint(0,len(parents_node1)-1)
            parent2 =  random.randint(0,len(parents_node2)-1)
            if parents_node1  !=  parents_node2:
                while parents_node1[parent1]  ==  parents_node2[parent2]:
                    parent1 =  random.randint(0,len(parents_node1)-1)
                    parent2 =  random.randint(0,len(parents_node2)-1)
                if parents_node1[parent1] not in parents_node2 and parents_node2[parent2] not in parents_node1:
                    if parents_node1[parent1]  !=  nodes[node2] and parents_node2[parent2]  !=  nodes[node1]:
                        aux_nodes = deepcopy(nodes)
                        index_parant1 = nodes.index(parents_node1[parent1])
                        index_parant2 = nodes.index(parents_node2[parent2])
                        aux = deepcopy(aux_nodes[index_parant2])
                        aux_nodes[index_parant2] = aux_nodes[index_parant1]
                        aux_nodes[index_parant1] = aux
                        mapping = {}    
                        for k in range(len(aux_nodes)):
                            mapping.update({nodes[k]:aux_nodes[k]})
                        bee_aux[3][0] = nx.relabel_nodes(bee_aux[3][0], mapping)
                        cost = bic_score(bee_aux[3][0],data)                    
                        cost = abs(cost)
                        move = False
    new_bee = old_bee[0]
    new_score = old_bee[1]
    for i in range(4):
        cost = bic_score(bee_aux[i][0],data) 
        cost = abs(cost)
        if cost<new_score:
            new_bee = bee_aux[i][0]
            new_score = cost
    return new_bee,new_score
         
        
def ABC_B(K, q0, qd, alpha, beta, p, limit, data, nodes, max_iter, goalscore):
    t = 0
    iteration = 0   
    bees = []
    net = nx.DiGraph()
    for a in nodes:
        net.add_node(a) 
    pheromones_matrix = []
    for i in nodes:
        pheromones_matrix_aux = []
        for j in nodes:
            if i == j:
                pheromones_matrix_aux.append(0)                
            else:
                net.add_edge(i,j)
                cost = bic_score(net,data)                
                cost = abs(cost)
                pheromones_matrix_aux.append(1/(len(nodes)*cost))
                net.remove_edge(i,j)
        pheromones_matrix.append(pheromones_matrix_aux)    
    for i in range(K):
        G = nx.fast_gnp_random_graph(len(nodes),0.3,directed = True)
        state  =  nx.DiGraph([(u,v,{'weight':1}) for (u,v) in G.edges() if u<v])
        random.shuffle(nodes)
        mapping = {}
        random.shuffle(nodes)
        for k in range(len(nodes)):
            mapping.update({k:nodes[k]})
        state = nx.relabel_nodes(state, mapping)
        for a in nodes:
            if a not in state.nodes():
                state.add_node(a)
        cost = bic_score(state,data)
        bees.append([state,cost])
    t = 0
    best_solution = bees[0]
    state = nx.DiGraph()
    for a in nodes:
        state.add_node(a)
    cost_inicial = bic_score(state,data)
    #cost_inicial = 1
    #cost_inicial = abs(cost_inicial)
    score = float('inf')  
    score_matriz = []   

    # VERIFICAR SE CORRETAMENTE IMPLEMENTADO O HEURISTIC INFORMATION
    # IDEIA: SUBSTITUIR O BIC PELO K2 SCORE LOCAL COMO FEITO NO ARTIGO

    for i in nodes:
        score_matriz_aux = []
        for j in nodes:
            if i != j:
                state.add_edge(i,j)
                cost = bic_score(state,data)
                score_matriz_aux.append(cost)
                state.remove_edge(i,j)
                if cost < score:
                    score = cost
            else:
                score_matriz_aux.append(0)
        score_matriz.append(score_matriz_aux)
    score_matriz = np.array(score_matriz)
    #bees = np.array(bees)
    pheromones_matrix = np.array(pheromones_matrix)
    besttemp = []
    while iteration < max_iter and best_solution[1]>(goalscore+0.00000001):
        print("best solution")
        print(best_solution[1])
        #Neighbor search phase of employed bees
        t += 1
        #print(t)
        old_bees = deepcopy(bees)
        for i in range(K):
            [new_bee,new_score] = neighbor_search(bees[i],nodes,data)
            if bees[i][1]>new_score:
                bees[i][0] = new_bee
                bees[i][1] = new_score
        #Neighbor search phase of onlookers
        soma_score = 0
        for i in range(K):
            soma_score += bees[i][1]
        prob_bee = [bees[i][1]/soma_score for i in range(K)]
        
        for i in range(K):
            chose = roleta(prob_bee)
            q = random.random()
            if q<qd:
                [new_bee,new_score] = neighbor_KG_search_v2(bees[chose],score_matriz,nodes,data)
            else:

                [new_bee,new_score] = neighbor_search(bees[chose],nodes,data)
            if bees[i][1]>new_score:
                bees[i][0] = new_bee
                bees[i][1] = new_score                
        #Exploring new solutions by scouts
        Ck = 0
        for i in range(K):
            if bees[i][1] == old_bees[i][1]:
                Ck += 1
            else:
                Ck = 0
            if Ck == limit:                
                bees[i] = bees[i]                
                net = nx.DiGraph()
                for a in nodes:
                    net.add_node(a)
                cost = bic_score(net,data)
                cost = abs(cost)
                new_bee = [net,cost]
                old_bee = [new_bee[0],cost+1]
                key = True
                pheromones_matrix_2 = pheromones_matrix[:]
                [eta_aux,eta_soma] = Probabilistic_transition_rule_v2_new(pheromones_matrix_2, new_bee, nodes, alpha, beta, score_matriz, cost_inicial)
                while old_bee[1]>new_bee[1] or key:
                    key =  False        
                    old_bee = new_bee[:]
                    edge_add = Probabilistic_transition_rule_v2_aux(eta_aux, eta_soma, q0)
                    if new_bee[0].has_edge(edge_add[1],edge_add[0]):
                        index_i = nodes.index(edge_add[0])
                        index_j = nodes.index(edge_add[1])                
                        pheromones_matrix_2[index_i][index_j] = 0
                        key = True
                    else:        
                        new_bee[0].add_edge(edge_add[0],edge_add[1])
                        search_dag(new_bee[0],edge_add[0],edge_add[1])                            
                    cost = bic_score(new_bee[0],data)
                    new_bee[1] = abs(cost)
                bees[i] = deepcopy(old_bee)                                                
                Ck = 0
        #Memorize the best solution G found so far
        for i in range(K):
            if best_solution[1]>bees[i][1]:
                best_solution = deepcopy(bees[i])       
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if best_solution[0].has_edge(nodes[i],nodes[j]):
                    delta_pheromone = 1/best_solution[1]
                else:
                    delta_pheromone = pheromones_matrix[i][j]                    
                pheromones_matrix[i][j] = (1-p)*pheromones_matrix[i][j]+p*delta_pheromone
        besttemp.append(best_solution[1])

        iteration += 1
        
    return best_solution[0],best_solution[1],t,besttemp
   
def compute_metrics():
    best_score_bic = bic_score(best_graph, data)
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
    parser.add_argument('--mu', type=float, help='Lagrangian multiplier.')
    parser.add_argument('--feasible_only', action='store_true')
    parser.add_argument('--no-feasible_only', dest='feasible_only', action='store_false')
    parser.add_argument('--feasible_only_init_pop', action='store_true')
    parser.add_argument('--no-feasible_only_init_pop', dest='feasible_only_init_pop', action='store_false')
    parser.add_argument('--num_runs', type=int, help='Number of runs', default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    parser.add_argument('--K', type=int, help='kk')
    parser.add_argument('--q0', type=float, help='q0')
    parser.add_argument('--qd', type=float, help='qd')
    parser.add_argument('--alpha', type=float, help='alpha')
    parser.add_argument('--beta', type=float, help='beta')
    parser.add_argument('--p', type=float, help='p')
    parser.add_argument('--limit', type=int, help='limit')
    args = parser.parse_args()

    PATH = '/home/joao/Desktop/UFMG/PhD/code/EA-DAG/results/ABC/' + args.data + '/' 


    # Parameters
    n_runs = args.num_runs
    K = args.K
    q0 = args.q0
    qd = args.qd
    alpha = args.alpha
    beta = args.beta
    p = args.p
    limit = args.limit
    max_iter = args.max_iter


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

        goal_bic = bic_score(ground_truth, data)
        print('Goal BIC:', goal_bic)

        # measure time
        start = time.time()

        # Create initial population
        nodes = data.columns
        nodes = list(nodes)

        # Evolve population
        print("Flying Bees")

        #def ABC_B(K, q0, qd, alpha, beta, p, limit, data, nodes, max_iter, goalscore):
        best_graph, best_score, t, besttemp = ABC_B(K, q0, qd, alpha, beta, p, limit, data, nodes, max_iter, goal_bic)

        # Compute metrics
        compute_metrics()

        print("Algorithm ended. Computing Results")
        end = time.time()
        time_vector.append(end-start)


    print("Mean time:", statistics.mean(time_vector))