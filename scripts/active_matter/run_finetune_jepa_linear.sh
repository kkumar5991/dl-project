#!/bin/bash
# Linear-probe finetune of a JEPA / V-JEPA encoder.
# Usage: bash run_finetune_jepa_linear.sh <ConvEncoder_xx.pth>
source "$(dirname "$0")/../env_setup.sh"

python -m physics_jepa.finetune \
    configs/train_activematter_small.yaml \
    ft.use_attentive_pooling=false \
    ft.head_type=linear \
    ft.run_name=activematter-jepa-linear-eval \
    --trained_model_path "$1"
