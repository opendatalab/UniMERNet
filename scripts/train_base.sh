#!/bin/bash

set -x
# srun -p s2_bigdata --gres=gpu:4 --quotatype=reserved  torchrun --nproc-per-node 4 --master_port 29500 train.py --cfg-path configs/train_t384.yaml
srun -p bigdata_alg --gres=gpu:4 --quotatype=reserved  torchrun --nproc-per-node 4 --master_port 29500 train.py --cfg-path configs/unimernet_base_train.yaml
set +x