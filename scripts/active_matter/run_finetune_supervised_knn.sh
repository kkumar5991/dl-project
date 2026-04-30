#!/bin/bash
# kNN evaluation of the SUPERVISED-baseline-trained encoder.
# Usage: bash run_finetune_supervised_knn.sh <ConvEncoder_xx.pth>
source "$(dirname "$0")/../env_setup.sh"

python -m physics_jepa.finetune \
    configs/train_activematter_small.yaml \
    ft=knn \
    model.objective=supervised \
    ft.run_name=activematter-supervised-knn-eval \
    --trained_model_path "$1"
