#!/bin/bash
#SBATCH --job-name=field-latent-probe
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/logs/field_latent_probe_%j.out
#SBATCH --error=/scratch/%u/logs/field_latent_probe_%j.err
#SBATCH --requeue
#SBATCH --signal=B:SIGTERM@120

# Usage: sbatch slurm_field_latent_probe.sh /path/to/full_ckpt.pt /path/to/velocity_ckpt.pt
FULL_CKPT=${1:?"Usage: sbatch slurm_field_latent_probe.sh <full_ckpt_path> <field_ckpt_path>"}
FIELD_CKPT=${2:?"Usage: sbatch slurm_field_latent_probe.sh <full_ckpt_path> <field_ckpt_path>"}

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

        bash scripts/active_matter/run_field_latent_probe.sh ${FULL_CKPT} ${FIELD_CKPT}
    " &

# Wait for the child and forward signals so the Python process can checkpoint
CHILD_PID=$!
trap 'kill -TERM ${CHILD_PID}; wait ${CHILD_PID}' SIGTERM SIGUSR1
wait ${CHILD_PID}
