#!/bin/bash

# Run the ABC algorithm with the asia dataset
python3 bees.py --sample_size 1000 --num_runs 1 --data "asia" --max_iter 10 --K 100 --q0 0.8 --qd 0.0 --alpha 1.0 --beta 2.0 --p 1.0 --limit 3 --mu 5 --feasible_only --feasible_only_init_pop --verbose
