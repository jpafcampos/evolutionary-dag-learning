#!/bin/bash

# Run the MAGA algorithm with the asia dataset
python3 MAGA_original.py --sample_size 1000 --num_runs 10 --data "asia" --max_iter 50 --Pm_min 0.01 \
  --Pm_max 0.01 --Po 0.05 --Pc_min 0.95 --Pc_max 0.95 --L_size 10 --sL 3 --sPm 0.01 --sGen 10 --mu 5 --feasible_only --feasible_only_init_pop --verbose
