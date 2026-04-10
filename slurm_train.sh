#!/bin/bash
#SBATCH --job-name=vjepa2-active-matter
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
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
SIF=/share/apps/images/cuda12.3.2-cudnn9.0.0-ubuntu-22.04.sif

mkdir -p ${SCRATCH}/logs

# ── Launch via Singularity ──────────────────────────────────────────────────────
singularity exec \
    --nv \
    --overlay ${OVERLAY}:ro \
    ${SIF} \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate dl_env

        export NETID=${NETID}
        export WANDB_API_KEY=\$(cat ~/.wandb_key 2>/dev/null || echo '')

        cd /scratch/${NETID}/dl-project
        jupyter nbconvert --to script vjepa2_active_matter.ipynb --stdout | python
    "
