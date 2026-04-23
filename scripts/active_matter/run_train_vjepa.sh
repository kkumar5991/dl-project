#!/bin/bash
source "$(dirname "$0")/../env_setup.sh"

torchrun --nproc_per_node=1 --standalone \
    -m physics_jepa.train_vjepa \
    configs/train_activematter_vjepa.yaml \
    $1
