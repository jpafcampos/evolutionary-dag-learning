''' Hill Climbing Search
    Adapted from pgmpy library
'''

#!/usr/bin/env python
from collections import deque
from itertools import permutations

import networkx as nx
from tqdm.auto import trange

#from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import (
    AICScore,
    BDeuScore,
    BDsScore,
    BicScore,
    K2Score,
    ScoreCache,
    StructureEstimator,
    StructureScore,
)

import csv
import argparse
import copy
import os
import pandas as pd
from utils import *
from eval_metrics import *
from ga_operators import *
from loaders import *
import time
import matplotlib.pyplot as plt

class CustomHillClimbSearch(StructureEstimator):
    """
    Class for heuristic hill climb searches for DAGs, to learn
    network structure from data. `estimate` attempts to find a model with optimal score.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.NaN`.
        Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    use_caching: boolean
        If True, uses caching of score for faster computation.
        Note: Caching only works for scoring methods which are decomposable. Can
        give wrong results in case of custom scoring methods.

    References
    ----------
    Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.4.3 (page 811ff)
    """

    def __init__(self, data, use_cache=True, **kwargs):
        self.use_cache = use_cache
        self.num_eval = 0
        self.reached_goal = False
        super(CustomHillClimbSearch, self).__init__(data, **kwargs)

    def _legal_operations(
        self,
        model,
        score,
        structure_score,
        tabu_list,
        max_indegree,
        black_list,
        white_list,
        fixed_edges,
    ):
        """Generates a list of legal (= not in tabu_list) graph modifications
        for a given model, together with their score changes. Possible graph modifications:
        (1) add, (2) remove, or (3) flip a single edge. For details on scoring
        see Koller & Friedman, Probabilistic Graphical Models, Section 18.4.3.3 (page 818).
        If a number `max_indegree` is provided, only modifications that keep the number
        of parents for each node below `max_indegree` are considered. A list of
        edges can optionally be passed as `black_list` or `white_list` to exclude those
        edges or to limit the search.
        """

        tabu_list = set(tabu_list)

        # Step 1: Get all legal operations for adding edges.
        potential_new_edges = (
            set(permutations(self.variables, 2))
            - set(model.edges())
            - set([(Y, X) for (X, Y) in model.edges()])
        )

        for X, Y in potential_new_edges:
            # Check if adding (X, Y) will create a cycle.
            if not nx.has_path(model, Y, X):
                operation = ("+", (X, Y))
                if (
                    (operation not in tabu_list)
                    and ((X, Y) not in black_list)
                    and ((X, Y) in white_list)
                ):
                    old_parents = model.get_parents(Y)
                    new_parents = old_parents + [X]
                    if len(new_parents) <= max_indegree:
                        score_delta = score(Y, new_parents) - score(Y, old_parents)
                        self.num_eval += 2
                        score_delta += structure_score("+")
                        yield (operation, score_delta)

        # Step 2: Get all legal operations for removing edges
        for X, Y in model.edges():
            operation = ("-", (X, Y))
            if (operation not in tabu_list) and ((X, Y) not in fixed_edges):
                old_parents = model.get_parents(Y)
                new_parents = [var for var in old_parents if var != X]
                score_delta = score(Y, new_parents) - score(Y, old_parents)
                self.num_eval += 2
                score_delta += structure_score("-")
                yield (operation, score_delta)

        # Step 3: Get all legal operations for flipping edges
        for X, Y in model.edges():
            # Check if flipping creates any cycles
            if not any(
                map(lambda path: len(path) > 2, nx.all_simple_paths(model, X, Y))
            ):
                operation = ("flip", (X, Y))
                if (
                    ((operation not in tabu_list) and ("flip", (Y, X)) not in tabu_list)
                    and ((X, Y) not in fixed_edges)
                    and ((Y, X) not in black_list)
                    and ((Y, X) in white_list)
                ):
                    old_X_parents = model.get_parents(X)
                    old_Y_parents = model.get_parents(Y)
                    new_X_parents = old_X_parents + [Y]
                    new_Y_parents = [var for var in old_Y_parents if var != X]
                    if len(new_X_parents) <= max_indegree:
                        score_delta = (
                            score(X, new_X_parents)
                            + score(Y, new_Y_parents)
                            - score(X, old_X_parents)
                            - score(Y, old_Y_parents)
                        )
                        self.num_eval += 4
                        score_delta += structure_score("flip")
                        yield (operation, score_delta)

    def estimate(
        self,
        scoring_method="k2score",
        start_dag=None,
        fixed_edges=set(),
        tabu_length=100,
        max_indegree=None,
        black_list=None,
        white_list=None,
        epsilon=1e-4,
        max_iter=1e6,
        show_progress=True,
        file=None,
        goal_bic=None,
        ground_truth=None
    ):
        """
        Performs local hill climb search to estimates the `DAG` structure that
        has optimal score, according to the scoring method supplied. Starts at
        model `start_dag` and proceeds by step-by-step network modifications
        until a local maximum is reached. Only estimates network structure, no
        parametrization.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation.  Supported
            structure scores: k2score, bdeuscore, bdsscore, bicscore, aicscore. Also accepts a
            custom score, but it should be an instance of `StructureScore`.

        start_dag: DAG instance
            The starting point for the local search. By default, a completely
            disconnected network is used.

        fixed_edges: iterable
            A list of edges that will always be there in the final learned model.
            The algorithm will add these edges at the start of the algorithm and
            will never change it.

        tabu_length: int
            If provided, the last `tabu_length` graph modifications cannot be reversed
            during the search procedure. This serves to enforce a wider exploration
            of the search space. Default value: 100.

        max_indegree: int or None
            If provided and unequal None, the procedure only searches among models
            where all nodes have at most `max_indegree` parents. Defaults to None.

        black_list: list or None
            If a list of edges is provided as `black_list`, they are excluded from the search
            and the resulting model will not contain any of those edges. Default: None

        white_list: list or None
            If a list of edges is provided as `white_list`, the search is limited to those
            edges. The resulting model will then only contain edges that are in `white_list`.
            Default: None

        epsilon: float (default: 1e-4)
            Defines the exit condition. If the improvement in score is less than `epsilon`,
            the learned model is returned.

        max_iter: int (default: 1e6)
            The maximum number of iterations allowed. Returns the learned model when the
            number of iterations is greater than `max_iter`.

        Returns
        -------
        Estimated model: pgmpy.base.DAG
            A `DAG` at a (local) score maximum.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import HillClimbSearch, BicScore
        >>> # create data sample with 9 random variables:
        ... data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 9)), columns=list('ABCDEFGHI'))
        >>> # add 10th dependent variable
        ... data['J'] = data['A'] * data['B']
        >>> est = HillClimbSearch(data)
        >>> best_model = est.estimate(scoring_method=BicScore(data))
        >>> sorted(best_model.nodes())
        ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        >>> best_model.edges()
        OutEdgeView([('B', 'J'), ('A', 'J')])
        >>> # search a model with restriction on the number of parents:
        >>> est.estimate(max_indegree=1).edges()
        OutEdgeView([('J', 'A'), ('B', 'J')])
        """

        # Step 1: Initial checks and setup for arguments
        # Step 1.1: Check scoring_method
        supported_methods = {
            "k2score": K2Score,
            "bdeuscore": BDeuScore,
            "bdsscore": BDsScore,
            "bicscore": BicScore,
            "aicscore": AICScore,
        }
        if (
            (
                isinstance(scoring_method, str)
                and (scoring_method.lower() not in supported_methods)
            )
        ) and (not isinstance(scoring_method, StructureScore)):
            raise ValueError(
                "scoring_method should either be one of k2score, bdeuscore, bicscore, bdsscore, aicscore, or an instance of StructureScore"
            )

        if isinstance(scoring_method, str):
            score = supported_methods[scoring_method.lower()](data=self.data)
        else:
            score = scoring_method

        if self.use_cache:
            score_fn = ScoreCache(score, self.data).local_score
        else:
            score_fn = score.local_score

        # Step 1.2: Check the start_dag
        if start_dag is None:
            start_dag = DAG()
            start_dag.add_nodes_from(self.variables)
        elif not isinstance(start_dag, DAG) or not set(start_dag.nodes()) == set(
            self.variables
        ):
            raise ValueError(
                "'start_dag' should be a DAG with the same variables as the data set, or 'None'."
            )

        # Step 1.3: Check fixed_edges
        if not hasattr(fixed_edges, "__iter__"):
            raise ValueError("fixed_edges must be an iterable")
        else:
            fixed_edges = set(fixed_edges)
            start_dag.add_edges_from(fixed_edges)
            if not nx.is_directed_acyclic_graph(start_dag):
                raise ValueError(
                    "fixed_edges creates a cycle in start_dag. Please modify either fixed_edges or start_dag."
                )

        # Step 1.4: Check black list and white list
        black_list = set() if black_list is None else set(black_list)
        white_list = (
            set([(u, v) for u in self.variables for v in self.variables])
            if white_list is None
            else set(white_list)
        )

        # Step 1.5: Initialize max_indegree, tabu_list, and progress bar
        if max_indegree is None:
            max_indegree = float("inf")

        tabu_list = deque(maxlen=tabu_length)
        current_model = start_dag


        # Step 2: For each iteration, find the best scoring operation and
        #         do that to the current model. If no legal operation is
        #         possible, sets best_operation=None.
        while self.num_eval < max_iter:
            best_operation, best_score_delta = max(
                self._legal_operations(
                    current_model,
                    score_fn,
                    score.structure_prior_ratio,
                    tabu_list,
                    max_indegree,
                    black_list,
                    white_list,
                    fixed_edges,
                ),
                key=lambda t: t[1],
                default=(None, None),
            )

            if best_operation is None or best_score_delta < epsilon:
                break
            elif best_operation[0] == "+":
                current_model.add_edge(*best_operation[1])
                tabu_list.append(("-", best_operation[1]))
            elif best_operation[0] == "-":
                current_model.remove_edge(*best_operation[1])
                tabu_list.append(("+", best_operation[1]))
            elif best_operation[0] == "flip":
                X, Y = best_operation[1]
                current_model.remove_edge(X, Y)
                current_model.add_edge(Y, X)
                tabu_list.append(best_operation)
            
            write_metrics_history(file, current_model, goal_bic, ground_truth, data)

        # Step 3: Return if no more improvements or maximum iterations reached.
        bic = np.abs((structure_score(current_model, data, scoring_method="bic")))
        if bic <= goal_bic:
            self.reached_goal = True
        return current_model
    

def write_metrics_history(file, current_model, goal_bic, ground_truth, data):
    # calculate bic
    bic = np.abs((structure_score(current_model, data, scoring_method="bic")))
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(current_model.nodes())
    nx_graph.add_edges_from(current_model.edges())
    SLF, TLF = learning_factors(nx_graph, ground_truth)
    f1score, accuracy, precision, recall, SHD = accuracy_metrics(nx_graph, ground_truth)

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

    filename = PATH +f'HC_all_runs_results_{args.data}.csv'

    with open(filename, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.data, args.type_exp,
                          end-start, goal_bic, num_eval_bic, reached_goal, best_score_bic, best_score_ls, SLF, TLF])

if __name__ == '__main__':
        # Load arguments
    parser = argparse.ArgumentParser(description='Hill Climbing Algorithm for BN structural learning.')
    parser.add_argument('--data', type=str, help='Name of the dataset.')
    parser.add_argument('--num_runs', type=int, help='Number of runs', default=1)
    parser.add_argument('--tabu', type=int, help='Length of tabu search list', default=100)
    parser.add_argument('--random', type=int, help='Wether the sample is random or not', default=1)
    parser.add_argument('--type_exp', type=int, help='Type of experiment (ratio parameters)', default=1)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    args = parser.parse_args()

    PATH = '/home/joao/Desktop/UFMG/PhD/code/EA-DAG/results/HC/' + args.data + '/'
    # if PATH does not exist, change it to the path in the server
    if not os.path.exists(PATH):
        PATH = '/home/joaocampos/phd_code/evolutionary-dag-learning/results/HC/' + args.data + '/'
    if not os.path.exists(PATH):
        PATH = '/home/bessani/phd_code/evolutionary-dag-learning/results/HC/' + args.data + '/'

    randomized = True if args.random == 1 else False
    sample_size, max_bic_eval = load_samplesize_num_evals(args.data, args.type_exp)
    tabu_length = args.tabu

    time_vector = []
    for i in range(args.num_runs):
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
        file = PATH + 'results_' + str(i) + '.csv'
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['BIC - Goal', 'BIC', 'F1 Score', 'Accuracy', 'Precision', 'Recall', 'SHD', 'SLF', 'TLF'])

        # Run Hill Climbing
        start = time.time()
        hc = CustomHillClimbSearch(data)
        hc_model = hc.estimate(scoring_method='bicscore', max_iter=max_bic_eval, tabu_length=tabu_length, show_progress=args.verbose, file=file, goal_bic=goal_bic, ground_truth=ground_truth)
        num_eval_bic = hc.num_eval
        reached_goal = hc.reached_goal
        end = time.time()
        time_vector.append(end-start)
        best_graph = nx.DiGraph()
        best_graph.add_nodes_from(hc_model.nodes())
        best_graph.add_edges_from(hc_model.edges())
        compute_metrics()
