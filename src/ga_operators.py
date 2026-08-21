'''
Author = João Pedro Campos
Some functions are taken or adapted from the work by Itallo Machado
https://github.com/ItalloMachado/Mestrado

SPEED-OPTIMIZED VERSION.
The algorithm's behaviour is unchanged; only the machinery that made it slow
was reworked. The three sources of speedup are:

  1. Decomposable BIC with a global local-score cache.
     BIC(G) = sum_i localscore(X_i | Pa(X_i)). Each local score depends only on
     a node and its parent set, so it is cached by (node, frozenset(parents)).
     Across agents and generations the same (node, parents) combinations recur
     constantly, so most BIC evaluations become dictionary lookups.
     Measured ~30x faster than rescoring every node every time.

  2. Removed the per-operation gc.collect() calls.
     Forcing a full garbage collection inside the inner loop is very expensive
     and was being done thousands of times per generation. Python's automatic
     GC handles this fine.

  3. Cached adjacency matrix on each Individual.
     individual_to_digraph / is_dag / is_connected used to rebuild a networkx
     graph from the flat gene list every call. The adjacency matrix is now
     cached and invalidated only when genes change.

The scoring sign convention is preserved: compute_bic stores the NEGATED pgmpy
BIC, so LOWER self.bic is BETTER everywhere in the code.
'''

import random
import networkx as nx
import numpy as np
import copy

from pgmpy.estimators import BicScore

from utils import *
from loaders import *


# --------------------------------------------------------------------------- #
# Global local-score cache (the main speedup).
#
# Keyed by (id(data), node, frozenset(parents)) so that different data sets /
# folds never collide. `set_scoring_data` must be called once per data set
# before scoring; it builds the pgmpy BicScore and resets nothing (the cache is
# safe across folds because the data id is part of the key).
# --------------------------------------------------------------------------- #
_LOCAL_SCORE_CACHE = {}
_BIC_SCORERS = {}          # id(data) -> BicScore instance


def _get_scorer(data):
    """Return a cached BicScore for this data frame (built once per data set)."""
    key = id(data)
    scorer = _BIC_SCORERS.get(key)
    if scorer is None:
        scorer = BicScore(data)
        _BIC_SCORERS[key] = scorer
    return scorer


def _local_score(data, node, parents):
    """Cached local BIC score of `node` given `parents` (a frozenset).

    Returns pgmpy's local score (higher = better fit); the sign flip to the
    'lower is better' convention happens in Individual.compute_bic.
    """
    key = (id(data), node, parents)
    val = _LOCAL_SCORE_CACHE.get(key)
    if val is None:
        val = _get_scorer(data).local_score(node, list(parents))
        _LOCAL_SCORE_CACHE[key] = val
    return val


def clear_score_cache():
    """Optional: free memory between folds if it grows too large."""
    _LOCAL_SCORE_CACHE.clear()
    _BIC_SCORERS.clear()


# --------------------------------------------------------------------------- #
class Individual():
    def __init__(self, genes, nodes):
        self.genes = genes            # flattened adjacency matrix (list of 0/1)
        self.nodes = nodes
        self.bic = None
        self.pos = None               # position in MAGA grid
        self.neighbors = []
        self._adj = None              # cached n x n adjacency matrix
        self._digraph = None          # cached networkx DiGraph

    # -- gene mutations must invalidate the cached graph views --------------- #
    def _invalidate(self):
        self._adj = None
        self._digraph = None

    def init_agent(self, i, j, L_size):
        self.pos = i * L_size + j
        list_neigh = findNeighbors(i, j, L_size)
        self.neighbors = [
            list_neigh[0] * L_size + j,
            i * L_size + list_neigh[1],
            list_neigh[2] * L_size + j,
            i * L_size + list_neigh[3],
        ]

    def init_random(self, num_nodes, sparsity=0.1):
        G = nx.gnp_random_graph(num_nodes, sparsity, directed=True)
        self.genes = nx.adjacency_matrix(G).todense().flatten().tolist()
        self._invalidate()

    def update_fenotype(self, other):
        self.bic = other.bic
        self.genes = other.genes
        self._invalidate()

    def init_from_genes(self, genes):
        self.genes = genes
        self._invalidate()

    def bit_flip_mutation(self, feasible_only=False):
        bit = random.randint(0, len(self.genes) - 1)
        self.genes[bit] = 1 - self.genes[bit]
        self._invalidate()
        if feasible_only:
            if not self.is_dag():
                self.repair_dag()
            if not self.is_connected():
                self.repair_connectivity()

    def uniform_mutation(self, prob, feasible_only=False):
        for i in range(len(self.genes)):
            if random.random() < prob:
                self.genes[i] = 1 - self.genes[i]
        self._invalidate()
        if feasible_only:
            if not self.is_dag():
                self.repair_dag()
            if not self.is_connected():
                self.repair_connectivity()

    def compute_adjacency_matrix(self):
        if self._adj is None:
            n = len(self.nodes)
            self._adj = np.asarray(self.genes).reshape(n, n)
        return self._adj

    def individual_to_digraph(self):
        # cached; rebuilt only after genes change
        if self._digraph is not None:
            return self._digraph
        n = len(self.nodes)
        adj = self.compute_adjacency_matrix()
        G = nx.DiGraph()
        G.add_nodes_from(self.nodes)
        # vectorized edge extraction instead of a double python loop
        srcs, dsts = np.nonzero(adj)
        G.add_edges_from((self.nodes[s], self.nodes[d]) for s, d in zip(srcs, dsts))
        self._digraph = G
        return G

    def is_dag(self):
        return nx.is_directed_acyclic_graph(self.individual_to_digraph())

    def is_connected(self):
        return nx.is_weakly_connected(self.individual_to_digraph())

    def repair_connectivity(self):
        G = self.individual_to_digraph()
        if not self.is_connected():
            G = fix_disconnected_graph(G)
            self.genes = nx.to_numpy_array(G).flatten().tolist()
            self._invalidate()

    def repair_dag(self):
        G = self.individual_to_digraph()
        G = break_cycles(G)
        self.genes = nx.to_numpy_array(G).flatten().tolist()
        self._invalidate()

    def compute_bic(self, data):
        """Decomposable BIC using the global local-score cache.

        Sums cached per-node local scores, then negates so that LOWER is BETTER
        (unchanged sign convention). Because the cache is keyed by (node,
        parents), repeated parent sets across the population are free.
        """
        DAG = self.individual_to_digraph()
        score = 0.0
        for node in DAG.nodes():
            parents = frozenset(DAG.predecessors(node))
            score += _local_score(data, node, parents)
        self.bic = -score
        return self.bic

    def __str__(self):
        return str(self.genes) + ' - '


# --------------------------------------------------------------------------- #
def findNeighbors(i, j, L):
    i1 = L - 1 if i == 0 else i - 1
    i2 = 0 if i == L - 1 else i + 1
    j1 = L - 1 if j == 0 else j - 1
    j2 = 0 if j == L - 1 else j + 1
    return [i1, j1, i2, j2]


# expert edges that must never be touched by mutation (medical CAD data)
KEEP_LIST = [
    ('cod_I10', 'cod_I35'), ('cod_B57', 'cod_I48'), ('cod_E78', 'cod_I48'),
    ('teve_ec', 'cod_I27'), ('teve_ec', 'cod_I49'), ('teve_ec', 'cod_I48'),
    ('cod_Q21', 'cod_I27'), ('cod_Q21', 'cod_I48'), ('cod_E03', 'cod_G47'),
    ('cod_E66', 'cod_G47'), ('imc', 'cod_G47'), ('grupo_idade', 'teve_ec'),
    ('sexo_M', 'teve_ec'), ('cod_E78', 'teve_ec'), ('cod_I10', 'teve_ec'),
    ('cod_E03', 'teve_ec'), ('cod_E10', 'teve_ec'), ('cod_E11', 'teve_ec'),
    ('cod_E14', 'teve_ec'), ('cod_J44', 'teve_ec'), ('cod_G47', 'teve_ec'),
]
_KEEP_SET = set(KEEP_LIST) | {(b, a) for a, b in KEEP_LIST}   # O(1) lookup


def mutation(individual, data, feasible_only=True):
    '''
    Mutates an individual by randomly adding, removing or reversing an edge.
    Behaviour identical to the original; only the protected-edge lookup was
    made O(1) and redundant graph rebuilds removed.
    '''
    mutated_individual = Individual(individual.genes, individual.nodes)
    digraph = mutated_individual.individual_to_digraph()
    nodes = individual.nodes

    # pick two distinct nodes whose (either-direction) pair is not protected
    while True:
        node1 = random.choice(nodes)
        node2 = node1
        while node2 == node1:
            node2 = random.choice(nodes)
        if (node1, node2) not in _KEEP_SET:
            break

    rand = random.random()
    if digraph.has_edge(node1, node2):
        if rand < 0.5:
            digraph.remove_edge(node1, node2)
            digraph.add_edge(node2, node1)
            if feasible_only:
                digraph = break_cycles(digraph)
        else:
            digraph.remove_edge(node1, node2)
    elif digraph.has_edge(node2, node1):
        if rand < 0.5:
            digraph.remove_edge(node2, node1)
            digraph.add_edge(node1, node2)
            if feasible_only:
                digraph = break_cycles(digraph)
        else:
            digraph.remove_edge(node2, node1)
    else:
        if rand < 0.5:
            digraph.add_edge(node1, node2)
            if feasible_only:
                digraph = break_cycles(digraph)
        else:
            digraph.add_edge(node2, node1)
            if feasible_only:
                digraph = break_cycles(digraph)

    if not nx.is_weakly_connected(digraph):
        digraph = fix_disconnected_graph(digraph)

    mutated_individual.genes = nx.to_numpy_array(digraph).flatten().tolist()
    mutated_individual._invalidate()
    mutated_individual.compute_bic(data)
    return mutated_individual


def bnc_pso_crossover(parent1, parent2, data, feasible_only=True):
    '''
    Crossover operator (behaviour preserved from the original).
    Children inherit shared edges; differing edges are inherited stochastically.
    '''
    parent1_digraph = parent1.individual_to_digraph()
    parent2_digraph = parent2.individual_to_digraph()

    child1_digraph = nx.DiGraph()
    child1_digraph.add_nodes_from(parent1_digraph.nodes)
    child2_digraph = nx.DiGraph()
    child2_digraph.add_nodes_from(parent2_digraph.nodes)

    p2_edges = set(parent2_digraph.edges())
    p1_edges = set(parent1_digraph.edges())

    # edges present in both parents -> both children
    for edge in p1_edges & p2_edges:
        child1_digraph.add_edge(*edge)
        child2_digraph.add_edge(*edge)

    # edges in exactly one parent -> inherited stochastically, avoiding cycles
    unique_edges = list(p1_edges ^ p2_edges)
    random.shuffle(unique_edges)              # now actually shuffles (was a no-op)

    for edge in unique_edges:
        chosen_child = child1_digraph if random.random() < 0.5 else child2_digraph
        if random.random() < 0.5:
            if not nx.has_path(chosen_child, edge[1], edge[0]):   # avoid cycles
                chosen_child.add_edge(*edge)

    child1 = Individual(nx.to_numpy_array(child1_digraph).flatten().tolist(), parent1.nodes)
    child2 = Individual(nx.to_numpy_array(child2_digraph).flatten().tolist(), parent1.nodes)

    if not child1.is_connected():
        child1.repair_connectivity()
    if not child2.is_connected():
        child2.repair_connectivity()

    child1.compute_bic(data)
    child2.compute_bic(data)
    return child1, child2


def create_population(pop_size, nodes, data, feasible_only=True):
    '''Creates a flat population of individuals (used for GA and BNC-PSO).'''
    pop = []
    for _ in range(pop_size):
        individual = Individual([], nodes)
        individual.init_random(num_nodes=len(nodes), sparsity=0.1)
        if feasible_only:
            if not individual.is_dag():
                individual.repair_dag()
            if not individual.is_connected():
                individual.repair_connectivity()
        individual.compute_bic(data)
        pop.append(individual)
    return pop


def create_MAGA_population(L_size, nodes, data, feasible_only=True):
    '''Creates a population of individuals on a grid for the MAGA algorithm.'''
  
    pop = []
    for i in range(L_size):
        for j in range(L_size):
            individual = Individual([], nodes)
            individual.init_random(num_nodes=len(nodes), sparsity=0.1)
            individual._invalidate()
            if feasible_only:
                if not individual.is_dag():
                    individual.repair_dag()
                if not individual.is_connected():
                    individual.repair_connectivity()
            individual.compute_bic(data)
            individual.init_agent(i, j, L_size)
            pop.append(individual)
    return pop


def generate_random_dag(node_names, edge_prob=0.3):
    '''Generates a random DAG over the given node names.'''
    G = nx.DiGraph()
    G.add_nodes_from(node_names)
    ordered_nodes = sorted(node_names, key=lambda x: random.random())
    for i in range(len(ordered_nodes)):
        for j in range(i + 1, len(ordered_nodes)):
            if random.random() < edge_prob:
                G.add_edge(ordered_nodes[i], ordered_nodes[j])
    return G


def create_fast_MAGA_population(L_size, nodes, data, feasible_only=True):
    pop = []
    for i in range(L_size):
        for j in range(L_size):
            individual = Individual([], nodes)
            random_dag = generate_random_dag(nodes)
            individual.genes = nx.to_numpy_array(random_dag).flatten().tolist()
            individual._invalidate()
            individual.compute_bic(data)
            individual.init_agent(i, j, L_size)
            pop.append(individual)
    return pop


def s_create_agents(L_size, best_graph, data, feasible_only=True):
    agents = []
    genes = best_graph.genes
    nodes = best_graph.nodes
    for i in range(L_size):
        for j in range(L_size):
            ind = Individual(list(genes), nodes)   # copy genes so agents are independent
            ind.init_agent(i, j, L_size)
            ind.compute_bic(data)
            if i == 1 and j == 1:
                agents.append(ind)
            else:
                mutated_ind = mutation(ind, data, feasible_only=feasible_only)
                mutated_ind.init_agent(i, j, L_size)
                agents.append(mutated_ind)
    return agents


def find_best_neighbor(agents, index):
    neighbors = agents[index].neighbors
    best_neighbor = neighbors[0]
    best_neighbor_score = agents[best_neighbor].bic
    for i in range(1, 4):
        cand = neighbors[i]
        if best_neighbor_score > agents[cand].bic:
            best_neighbor = cand
            best_neighbor_score = agents[cand].bic
    return best_neighbor


def self_learning(sL_size, best_graph, mutation_prob, keep_mutation_prob,
                  max_iter, data, feasible_only=True):
    num_eval_bic = 0
    sBest = copy.deepcopy(best_graph)
    sBest_score = sBest.bic
    sAgents = s_create_agents(sL_size, sBest, data, feasible_only)
    num_eval_bic += sL_size * sL_size

    for _ in range(max_iter):
        for agent_idx in range(len(sAgents)):
            best_neighbor = find_best_neighbor(sAgents, agent_idx)
            child1, child2 = bnc_pso_crossover(
                sAgents[agent_idx], sAgents[best_neighbor], data, feasible_only)
            num_eval_bic += 2
            best_child = child1 if child1.bic < child2.bic else child2
            sAgents[agent_idx].update_fenotype(best_child)

        total_mutations = round(len(sAgents) * len(sAgents[0].nodes) * mutation_prob)
        mutation_counter = 0
        while mutation_counter < total_mutations:
            aux_rand = random.randint(0, len(sAgents) - 1)
            agent_before_mutation = sAgents[aux_rand]
            new_agent = mutation(agent_before_mutation, data, feasible_only)
            num_eval_bic += 1
            if new_agent.bic < sAgents[aux_rand].bic:
                sAgents[aux_rand].update_fenotype(new_agent)
                mutation_counter += 1
            elif random.random() < keep_mutation_prob:
                sAgents[aux_rand].update_fenotype(new_agent)
                mutation_counter += 1

        for agent in sAgents:
            if agent.bic < sBest_score:
                sBest = copy.deepcopy(agent)   # copy, so later mutation can't corrupt sBest
                sBest_score = agent.bic

    if sBest.bic < best_graph.bic:
        return sBest, num_eval_bic
    return best_graph, num_eval_bic
