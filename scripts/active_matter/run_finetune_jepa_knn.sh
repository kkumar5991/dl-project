#!/bin/bash
# kNN evaluation of a JEPA / V-JEPA encoder.
# Usage: bash run_finetune_jepa_knn.sh <ConvEncoder_xx.pth>
source "$(dirname "$0")/../env_setup.sh"

python -m physics_jepa.finetune \
    configs/train_activematter_small.yaml \
    ft=knn \
    ft.run_name=activematter-jepa-knn-eval \
    --trained_model_path "$1"
