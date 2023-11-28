#!/bin/bash

# Run the GA algorithm with the Adult dataset
python GA.py --sample_size 1000 --data "adult" --max_iter 100 --mutation_rate 0.4 --crossover_rate 1.0 --popSize 20 --patience 10 --density_factor 0.5 --mu 5 --no-feasible_only --no-feasible_only_init_pop
