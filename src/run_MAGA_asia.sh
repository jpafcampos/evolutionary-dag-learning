#!/bin/bash

# Run the MAGA algorithm with the asia dataset
python3 MAGA_original.py --type_exp 1 --num_runs 1 --data "asia"  --Pm_min 0.01 --Pm_max 0.01 --Po 0.05 --Pc_min 0.95 \
  --Pc_max 0.95 --L_size 10 --sL 3 --sPm 0.01 --sGen 5 --mu 5 --random 0 --self_learn 1 --feasible_only --feasible_only_init_pop --no-verbose
