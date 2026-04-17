#!/bin/bash
source "$(dirname "$0")/../env_setup.sh"

# Pass the pretrained checkpoint path as $1
python3 -m physics_jepa.field_importance \
    configs/train_activematter_small.yaml \
    --trained_model_path $1 \
    --save_path field_importance.png
