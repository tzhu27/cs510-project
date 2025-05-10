#! /bin/bash

export CUDA_VISIBLE_DEVICES=1,2,3,4

python3 -m gdig_py.main --config demo_config.yaml
