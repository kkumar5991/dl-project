#!/bin/bash
python -m physics_jepa.train_jepa \
    configs/train_rayleigh_benard_channel_jepa.yaml \
    --channel_masked
