#!/bin/bash

# srun -p s2_bigdata --gres=gpu:1 python test.py --cfg configs/eval_t384.yaml
srun -p bigdata_alg --gres=gpu:1 python test.py --cfg configs/unimernet_base_eval.yaml
