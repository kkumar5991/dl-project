#!/bin/bash
source "$(dirname "$0")/../env_setup.sh"

# Linear probe: full-model z → per-field z
# $1 = path to full (11-channel) encoder checkpoint
# $2 = path to velocity (2-channel) encoder checkpoint
python3 -m physics_jepa.field_latent_probe \
    configs/train_activematter_small.yaml \
    --full_ckpt_path "$1" \
    --field_ckpt_path "$2" \
    --field_indices 1 2 \
    --num_epochs 50 \
    --lr 1e-3 \
    --save_path field_latent_probe_velocity.pt
