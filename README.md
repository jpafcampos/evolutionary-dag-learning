# Multi-agent genetic algorithm for bayesian networks structural learning
Official repository for:

> Campos, J. P., Machado, I. G., & Bessani, M. (2025). *Multi-agent genetic algorithm for Bayesian networks structural learning.* **Knowledge-Based Systems, 310, 113025.**
> [https://doi.org/10.1016/j.knosys.2025.113025](https://doi.org/10.1016/j.knosys.2025.113025)

## Overview

This repository implements a multi-agent genetic algorithm for learning the structure of Bayesian networks from data.

**MAGABN** adapts the original Multi-Agent Genetic Algorithm (MAGA), originally designed for global numerical optimization, to the NP-hard problem of learning Bayesian network structure from data.

Candidate DAGs are represented as agents on a toroidal *L×L* lattice, where each agent interacts only with its neighbors. This neighborhood-restricted interaction balances exploration and exploitation, preventing high-scoring solutions from dominating the population and avoiding premature convergence. Each agent is scored by the **BIC**. The algorithm combines:

- **Crossover** with the best neighbor: shared parental edges are inherited by both children; differing edges are assigned randomly. The higher-scoring child is kept.
- **Mutation**: picks two nodes and reverses/removes an existing edge, or adds one in a random direction.
- **Repair**: removes or reverses random edges in detected cycles to restore acyclicity after crossover/mutation.
- **Self-learning**: a local search that refines the generation's best agent by replaying MAGABN on a smaller lattice.

## Results

MAGABN was compared against a classical **GA** [Larrañaga et al.], **BNCPSO**, and **HC-Tabu** on the ASIA, CHILD, and INSURANCE benchmarks, across sample-size ratios *n/|Θ|* ∈ {1, 20, 50} (20 runs each, randomized factorial design). Evaluation used the Structure Learning Factor (SLF, edge presence + direction) and Topology Learning Factor (TLF, presence only).

MAGABN achieved the best mean SLF in 6/8 cases and the lowest variance in 6/8, excelling especially on the larger CHILD and INSURANCE networks. A factorial ANOVA plus Tukey test confirmed MAGABN is systematically superior to GA and BNCPSO (+0.041 SLF) and to HC-Tabu (+0.100).

## Installation

```bash
git clone https://github.com/<user>/EA-DAG.git
cd EA-DAG
pip install -r requirements.txt
```

Requires Python with `networkx` (graph manipulation), `numpy`, and `pgmpy` (BIC scoring). Datasets are sampled from ground-truth networks in the [bnlearn repository](https://www.bnlearn.com/bnrepository/).


## Features

- [Multi-agent GA search over DAG space]
- [Scoring functions supported, e.g. BIC/BDeu]
- [Acyclicity enforcement / repair operators]
- [Benchmark networks and evaluation metrics: SHD, F1, etc.]

  
## Citation

If you use this work, please cite:

```bibtex
@article{campos2025multi,
  title   = {Multi-agent genetic algorithm for Bayesian networks structural learning},
  author  = {Campos, Joao P. A. F. and Machado, Itallo G. and Bessani, Michel},
  journal = {Knowledge-Based Systems},
  volume  = {310},
  pages   = {113025},
  year    = {2025},
  publisher = {Elsevier}
}
```
