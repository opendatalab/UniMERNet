#!/bin/bash

# for base size
python -m torch.distributed.launch --nproc_per_node=8 train.py --cfg-path configs/train/unimernet_base_encoder6666_decoder8_dim1024.yaml

# # for small size
# python -m torch.distributed.launch --nproc_per_node=8 train.py --cfg-path configs/train/unimernet_small_encoder6666_decoder8_dim768.yaml

# # for tiny size
# python -m torch.distributed.launch --nproc_per_node=8 train.py --cfg-path configs/train/unimernet_tiny_encoder6666_decoder8_dim768.yaml