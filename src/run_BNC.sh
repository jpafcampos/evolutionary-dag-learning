#!/bin/bash

# Run the BNC-PSO algorithm with the asia dataset
python3 bnc_pso.py --num_runs 20 --type_exp 1 --data "child" --popSize 100 --mu 5 --feasible_only --feasible_only_init_pop --no-verbose
