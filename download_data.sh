#!/bin/bash
# Download active_matter dataset from HuggingFace to /scratch/$NETID/data
# Run this from a data transfer node: ssh $NETID@dtn.torch.hpc.nyu.edu

NETID=$(whoami)
DATA_DIR=/scratch/${NETID}/dl-proj/data

mkdir -p ${DATA_DIR}

echo "Downloading polymathic-ai/active_matter (~52 GB) to ${DATA_DIR} ..."

hf download \
    polymathic-ai/active_matter \
    --repo-type dataset \
    --local-dir ${DATA_DIR}/active_matter \

echo "Done. Dataset at: ${DATA_DIR}/active_matter"
