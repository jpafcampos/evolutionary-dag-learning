#!/bin/bash

# Run the GA algorithm with the asia dataset
python3 GA.py --num_runs 20 --data "asia" --mutation_rate 0.1 --crossover_rate 0.7 --popSize 100 \
 --mu 5 --selection_pressure 1.2 --type_exp 1 --random 1 --feasible_only --feasible_only_init_pop --no-verbose