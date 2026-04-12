#!/bin/bash
#SBATCH --job-name=vjepa2-active-matter
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/logs/vjepa2_%j.out
#SBATCH --error=/scratch/%u/logs/vjepa2_%j.err
#SBATCH --requeue

# ── Environment ────────────────────────────────────────────────────────────────
NETID=$(whoami)
SCRATCH=/scratch/${NETID}
OVERLAY=${SCRATCH}/overlay-15GB-500K.ext3
SIF=${SCRATCH}/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

mkdir -p ${SCRATCH}/logs

# ── Launch via Singularity ──────────────────────────────────────────────────────
singularity exec \
    --nv \
    --overlay ${OVERLAY}:ro \
    ${SIF} \
    /bin/bash -c "

        export NETID=${NETID}
        export WANDB_API_KEY=\$(cat ~/.wandb_key 2>/dev/null || echo '')
        export THE_WELL_DATA_DIR=/scratch/${NETID}/data

        cd /scratch/${NETID}/dl-project/dl-project
        pip install --user --break-system-packages -r requirements.txt
        export PATH=\$HOME/.local/bin:\$PATH
        python3 generate_subset_config.py \
            --dataset active_matter \
            --num_frames 16 \
            --split train \
            --stride 1 \
            --output subset_config.json
        
    "
