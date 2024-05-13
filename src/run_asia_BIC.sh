#!/bin/bash

# Run the GA algorithm with the asia dataset
python3 GA.py --sample_size 1000 --num_runs 10 --data "asia" --max_iter 200 --mutation_rate 0.1 \
  --crossover_rate 0.7 --popSize 100 --patience 200 --mu 5 --selection_pressure 1.2 --feasible_only --feasible_only_init_pop --no-verbose --crossover_function 'bnc_pso'
