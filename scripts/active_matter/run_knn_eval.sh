#!/bin/bash
# Pass the pretrained checkpoint path as $1
python3 -m physics_jepa.knn_eval \
    configs/train_activematter_small.yaml \
    ft=knn \
    --trained_model_path $1
