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

        # Use or create a venv in /scratch (writable)
        VENV=/scratch/${NETID}/dl_venv
        cd /scratch/${NETID}/dl-project
        if [ ! -f \${VENV}/bin/activate ]; then
            echo 'Creating virtual environment...'
            python3 -m venv \${VENV}
            source \${VENV}/bin/activate
            pip install -r requirements.txt
        else
            source \${VENV}/bin/activate
        fi

        bash scripts/active_matter/run_train_jepa.sh
    "
