#!/bin/bash
source "$(dirname "$0")/../env_setup.sh"

# Train a velocity-only JEPA (channels 1,2)
torchrun --nproc_per_node=1 --standalone \
    -m physics_jepa.train_jepa \
    configs/train_activematter_small_velocity.yaml \
    $1
