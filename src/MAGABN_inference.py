import random
import networkx as nx
import pandas as pd
import time
import statistics
import argparse
import copy
import os

from pgmpy.models import BayesianNetwork
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from utils import *
from ga_operators import *
from loaders import *


# --------------------------------------------------------------------------- #
# Leak-free discretization of continuous columns.
#
# Bin edges are fit on the TRAINING fold only (quantiles of the training rows)
# and then applied to the test fold. Fitting qcut on the full data before the
# split would let each test fold's values influence its own bin boundaries,
# a subtle form of leakage that inflates cross-validated metrics.
#
# Labels are integer codes (0..k-1) so they match the rest of the pipeline,
# which casts every other column to int and feeds pgmpy discrete scoring.
# Outer edges are set to +/-inf so a test value outside the training range
# still lands in the first/last bin instead of becoming NaN.
# --------------------------------------------------------------------------- #
CONTINUOUS_COLS = {"imc": 4, "idade": 10}   # column -> number of quantile bins


def fit_bin_edges(train_series, n_bins):
    _, edges = pd.qcut(train_series.astype(float), q=n_bins,
                       retbins=True, duplicates='drop')
    edges = edges.copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def apply_bin_edges(series, edges):
    return pd.cut(series.astype(float), bins=edges, labels=False,
                  include_lowest=True).astype(int)


def discretize_train_test(train_df, test_df, cont_cols):
    """Bin continuous cols using train-fold edges; cast the rest to int."""
    train_df = train_df.copy()
    test_df = test_df.copy()
    for col, n_bins in cont_cols.items():
        if col not in train_df.columns:
            continue
        edges = fit_bin_edges(train_df[col], n_bins)
        train_df[col] = apply_bin_edges(train_df[col], edges)
        test_df[col] = apply_bin_edges(test_df[col], edges)
    for col in train_df.columns:
        if col not in cont_cols:
            train_df[col] = train_df[col].astype(int)
            test_df[col] = test_df[col].astype(int)
    return train_df, test_df


# --------------------------------------------------------------------------- #
def MAGA(population, data, max_eval_bic, Pm_min, Pm_max, Po, Pc_min, Pc_max,
         t_max, sL, sPm, sGen, self_learn, feasible_only, verbose=False):
    """Multi-Agent Genetic Algorithm for BN structure learning.

    Behaviour preserved from the original. Speed changes:
      - all gc.collect() calls removed (they were firing thousands of times
        per generation and dominating wall-clock time);
      - the recorded best graph is deep-copied so later mutation of its grid
        slot cannot silently corrupt it.
    Scoring convention unchanged: lower .bic is better.
    """
    best_bic = float('inf')
    best_graph = None
    best_pos = 0
    iteration = 0
    t = 0
    num_eval_bic = 0

    while num_eval_bic < max_eval_bic:
        if verbose:
            print('Iteration:', iteration, '| BIC evals:', num_eval_bic)

        Pc = Pc_min - t * (Pc_min - Pc_max) / t_max
        Pm = Pm_min - t * (Pm_min - Pm_max) / t_max
        t += 1

        # --- crossover sweep ---
        best_ind = 0
        for agent_idx in range(len(population)):
            agent = population[agent_idx]
            aux_best_bic = agent.bic if agent.bic is not None else agent.compute_bic(data)
            aux_best_idx = agent_idx

            if random.random() < Pc:
                best_neighbor = find_best_neighbor(population, agent_idx)
                if population[agent_idx].bic > population[best_neighbor].bic:
                    child1, child2 = bnc_pso_crossover(
                        population[agent_idx], population[best_neighbor],
                        data, feasible_only)
                    num_eval_bic += 2
                    best_child = child1 if child1.bic < child2.bic else child2
                    population[agent_idx].update_fenotype(best_child)
                    aux_best_bic = population[agent_idx].bic

            if aux_best_bic < population[best_ind].bic:
                best_ind = aux_best_idx

        # --- mutation sweep ---
        total_m = round(len(population) * len(population[0].nodes) * Pm)
        aux_m = 0
        if verbose:
            print('  performing mutations')
        while aux_m < total_m:
            aux_rand = random.randint(0, len(population) - 1)
            if aux_rand != best_ind:
                aux_m += 1
                agent_before_mutation = population[aux_rand]
                new_agent = mutation(agent_before_mutation, data, feasible_only)
                num_eval_bic += 1
                if new_agent.bic < agent_before_mutation.bic:
                    population[aux_rand].update_fenotype(new_agent)
                    aux_m += 1
                elif random.random() < Po:
                    population[aux_rand].update_fenotype(new_agent)
                    aux_m += 1

        # --- record global best (deep-copied so it can't be mutated later) ---
        for agent in population:
            if agent.bic < best_bic:
                best_bic = agent.bic
                best_graph = copy.deepcopy(agent)
                best_pos = agent.pos

        # --- self-learning local search on the best ---
        if self_learn:
            if verbose:
                print('  performing self learning')
            best_graph, num_eval_bic_sl = self_learning(
                sL, best_graph, sPm, Po, sGen, data, feasible_only)
            num_eval_bic += num_eval_bic_sl

        population[best_pos] = best_graph
        best_bic = best_graph.bic
        print(f"  best BIC: {best_bic:.2f} at pos {best_pos}")
        iteration += 1

    return best_graph, num_eval_bic, population


def learning_factors(ind, target):
    """Structure/Topology learning factors vs a ground-truth DAG."""
    TC = 0
    TE = len(target.edges())
    IE = 0
    for a, b in target.edges():
        if ind.has_edge(a, b):
            TC += 1
        if ind.has_edge(b, a):
            IE += 1
    SLF = TC / TE
    TLF = (TC + IE) / TE
    return SLF, TLF


# --------------------------------------------------------------------------- #
def evaluate_fold(best_graph, train_data, test_data, target="teve_ec"):
    """Fit parameters on the fold and return metrics on the test split.

    Uses predict_probability for AUC (a threshold-free metric needs the
    posterior of the positive class, not the hard 0/1 label). Accuracy,
    sensitivity, specificity, precision and F1 use the hard prediction.
    """
    structure = best_graph.individual_to_digraph()
    bn_model = BayesianNetwork(structure.edges())
    bn_model.fit(train_data)

    X_test = test_data.drop(target, axis=1)
    y_true = test_data[target]

    y_pred = bn_model.predict(X_test)[target]

    # posterior P(target = 1) for a proper AUC
    proba = bn_model.predict_probability(X_test)
    pos_col = f"{target}_1"
    if pos_col in proba.columns:
        y_score = proba[pos_col].values
    else:
        # fall back to hard predictions if the column name differs
        y_score = y_pred

    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred),
        "specificity": recall_score(y_true, y_pred, pos_label=0),
        "precision":   precision_score(y_true, y_pred),
        "f1":          f1_score(y_true, y_pred),
        "roc_auc":     roc_auc_score(y_true, y_score),
    }


# --------------------------------------------------------------------------- #
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Multi-Agent Genetic Algorithm for BN structural learning.')
    parser.add_argument('--sGen', type=int, help='Max iterations in self learning.')
    parser.add_argument('--sPm', type=float, help='Mutation prob inside self learning.')
    parser.add_argument('--Po', type=float, help='Prob of keeping worse individual.')
    parser.add_argument('--Pm_min', type=float, help='Min probability of mutation.')
    parser.add_argument('--Pm_max', type=float, help='Max probability of mutation.')
    parser.add_argument('--Pc_min', type=float, help='Min probability of crossover.')
    parser.add_argument('--Pc_max', type=float, help='Max probability of crossover.')
    parser.add_argument('--L_size', type=int, help='Grid size.')
    parser.add_argument('--sL', type=int, help='Small grid size.')
    parser.add_argument('--mu', type=float, help='Lagrangian multiplier.')
    parser.add_argument('--self_learn', type=int, default=1,
                        help='Whether to use self learning.')
    parser.add_argument('--feasible_only', action='store_true')
    parser.add_argument('--no-feasible_only', dest='feasible_only', action='store_false')
    parser.add_argument('--feasible_only_init_pop', action='store_true')
    parser.add_argument('--no-feasible_only_init_pop',
                        dest='feasible_only_init_pop', action='store_false')
    parser.add_argument('--random', type=int, default=1,
                        help='Whether the sample is random or not.')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--no-verbose', dest='verbose', action='store_false')
    args = parser.parse_args()

    PATH = './'
    TARGET = "EC"
    N_FOLDS = 10
    max_bic_eval = 60000
    t_max = 10

    self_learn = args.self_learn == 1

    # --- load (continuous cols left RAW here; discretized per-fold below) ---
    data = load_medical_data()
    print(data.head())

    nodes = list(data.columns)

    # --- stratified k-fold on the target ---
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    y_strat = data[TARGET].values

    metric_rows = []
    fold_times = []

    for i, (train_index, test_index) in enumerate(skf.split(data, y_strat), start=1):
        print(f"\n=== Fold {i}/{N_FOLDS} ===")
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        # leak-free discretization: edges fit on THIS fold's training rows only
        train_data, test_data = discretize_train_test(
            train_data, test_data, CONTINUOUS_COLS)

        t_fold = time.perf_counter()

        print("Creating initial population")
        population = create_MAGA_population(
            args.L_size, nodes, train_data,
            feasible_only=args.feasible_only_init_pop)

        print("Running MAGABN")
        best_graph, num_eval_bic, population = MAGA(
            population, train_data, max_bic_eval,
            args.Pm_min, args.Pm_max, args.Po, args.Pc_min, args.Pc_max,
            t_max, args.sL, args.sPm, args.sGen, self_learn,
            args.feasible_only, verbose=args.verbose)

        # save the learned structure for this fold (for cross-fold explainability)
        nx.write_gml(best_graph.individual_to_digraph(),
                     os.path.join(PATH, f'complete_data_best_graph_fold_{i}.gml'))

        print("Fitting BN and evaluating")
        m = evaluate_fold(best_graph, train_data, test_data, target=TARGET)

        elapsed = time.perf_counter() - t_fold
        fold_times.append(elapsed)
        m["fold"] = i
        m["time_s"] = elapsed
        m["bic_evals"] = num_eval_bic
        metric_rows.append(m)

        print(f"  fold {i}: acc={m['accuracy']:.4f} sens={m['sensitivity']:.4f} "
              f"spec={m['specificity']:.4f} auc={m['roc_auc']:.4f} "
              f"({elapsed:.1f}s)")

    # --- summary ---
    results = pd.DataFrame(metric_rows).set_index("fold")
    metric_cols = ["accuracy", "sensitivity", "specificity", "precision", "f1", "roc_auc"]

    print("\nPer-fold results:")
    print(results[metric_cols + ["time_s"]].round(4).to_string())

    print("\nSummary (mean ± std across folds):")
    for c in metric_cols:
        print(f"  {c:12s} {results[c].mean():.4f} ± {results[c].std():.4f}")
    print(f"\nMean fold time: {statistics.mean(fold_times):.1f}s "
          f"(total {sum(fold_times):.1f}s)")

    # --- persist, matching the original two-row mean/std CSV layout ---
    out = pd.DataFrame(columns=["Accuracy", "Sensitivity", "Specificity",
                                "Precision", "F1", "ROC AUC"])
    out.loc[0] = [results[c].mean() for c in metric_cols]
    out.loc[1] = [results[c].std() for c in metric_cols]
    out.to_csv(os.path.join(PATH, 'complete_data_results_MAGABN.csv'), index=False)
    # also save the full per-fold table
    results.to_csv(os.path.join(PATH, 'complete_data_results_MAGABN_perfold.csv'))