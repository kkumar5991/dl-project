#!/bin/bash
source "$(dirname "$0")/../env_setup.sh"

torchrun --nproc_per_node=1 --standalone \
    -m physics_jepa.train_supervised \
    configs/train_activematter_supervised.yaml \
    $1
