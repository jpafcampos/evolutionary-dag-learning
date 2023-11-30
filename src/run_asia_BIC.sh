#!/bin/bash

# Run the GA algorithm with the asia dataset
python GA.py --sample_size 1000 --data "asia" --max_iter 100 --mutation_rate 0.5 --crossover_rate 1.0 --popSize 40 --patience 10 --density_factor 0.1 --mu 5 --feasible_only --feasible_only_init_pop
