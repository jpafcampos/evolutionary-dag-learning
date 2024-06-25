#!/bin/bash

# Run the GA algorithm with the Adult dataset
python GA.py --sample_size 500 --num_runs 10 --data "child" --max_iter 200 --mutation_rate 0.1 \
    --crossover_rate 0.7 --popSize 100 --patience 200 --mu 5 --selection_pressure 1.2 --feasible_only --feasible_only_init_pop --no-verbose
