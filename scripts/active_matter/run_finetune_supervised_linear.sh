#!/bin/bash
# Linear-probe evaluation of the SUPERVISED-baseline-trained encoder.
# Usage: bash run_finetune_supervised_linear.sh <ConvEncoder_xx.pth>
source "$(dirname "$0")/../env_setup.sh"

python -m physics_jepa.finetune \
    configs/train_activematter_small.yaml \
    model.objective=supervised \
    ft.use_attentive_pooling=false \
    ft.head_type=linear \
    ft.run_name=activematter-supervised-linear-eval \
    --trained_model_path "$1"
