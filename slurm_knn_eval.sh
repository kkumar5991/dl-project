#!/bin/bash
#SBATCH --job-name=knn-eval
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/logs/knn_eval_%j.out
#SBATCH --error=/scratch/%u/logs/knn_eval_%j.err

# ── Environment ────────────────────────────────────────────────────────────────
NETID=$(whoami)
SCRATCH=/scratch/${NETID}
OVERLAY=${SCRATCH}/overlay-15GB-500K.ext3
SIF=${SCRATCH}/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

mkdir -p ${SCRATCH}/logs

# Pass the checkpoint path as the first argument: sbatch slurm_knn_eval.sh <checkpoint_path>
CHECKPOINT=${1:-"checkpoints/active_matter-16frames-cnn-jepa-Baseline_15_pct_subset/ConvEncoder_1.pth"}

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

        bash scripts/active_matter/run_knn_eval.sh ${CHECKPOINT}
    " &

CHILD_PID=$!
trap 'kill -TERM ${CHILD_PID}; wait ${CHILD_PID}' SIGTERM SIGUSR1
wait ${CHILD_PID}
