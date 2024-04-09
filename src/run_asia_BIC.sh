#!/bin/bash

# Run the GA algorithm with the asia dataset
python GA.py --sample_size 1000 --data "asia" --max_iter 10 --mutation_rate 0.1 --crossover_rate 1.0 --popSize 40 --patience 10 --mu 5 --feasible_only --feasible_only_init_pop --verbose
