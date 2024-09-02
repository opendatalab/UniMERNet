#!/bin/bash


# eval unimernet_base
python test.py --cfg configs/val/unimernet_base.yaml

# eval unimernet_small
python test.py --cfg configs/val/unimernet_small.yaml

# eval unimernet_tiny
python test.py --cfg configs/val/unimernet_tiny.yaml