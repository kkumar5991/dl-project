#!/bin/bash
#SBATCH --job-name=gen-subsets
#SBATCH --account=csci_ga_2572-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/%u/logs/gen_subsets_%j.out
#SBATCH --error=/scratch/%u/logs/gen_subsets_%j.err

# Generates multiple subset_config JSON files covering different
# (alpha, zeta) combinations and trajectory counts.
#
# Naming convention:
#   subset_a<num_alpha>_z<num_zeta>_<approx_pct>pct.json
#
# Full dataset: 5 alpha × 9 zeta = 45 param pairs
# Each param pair has multiple trajectories with many temporal windows.

NETID=$(whoami)
SCRATCH=/scratch/${NETID}
OVERLAY=${SCRATCH}/overlay-15GB-500K.ext3
SIF=${SCRATCH}/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif

mkdir -p ${SCRATCH}/logs

singularity exec \
    --overlay ${OVERLAY}:ro \
    ${SIF} \
    /bin/bash -c "
        export THE_WELL_DATA_DIR=/scratch/${NETID}/data
        cd /scratch/${NETID}/dl-project/dl-project
        pip install --user --break-system-packages -r requirements.txt 2>/dev/null
        export PATH=\$HOME/.local/bin:\$PATH

        SCRIPT='python3 generate_subset_config.py --dataset active_matter --num_frames 16 --split train --stride 1'

        mkdir -p subset_configs configs/dataset/subset_configs
        echo '=== Generating subsets =='

        # ── Full phase space (all 5 alpha × 5 zeta, 2 traj/pair) ≈ 20%
        echo 'subset_a5_z5_20pct.json ...'
        \${SCRIPT} \
            --alpha -1 -2 -3 -4 -5 \
            --zeta 1 5 9 13 17 \
            --n_traj 2 \
            --output configs/dataset/subset_configs/subset_a5_z5_20pct.json

        # ── Full phase space (all 5 alpha × 5 zeta, 4 traj/pair) ≈ 40%
        echo 'subset_a5_z5_40pct.json ...'
        \${SCRIPT} \
            --alpha -1 -2 -3 -4 -5 \
            --zeta 1 5 9 13 17 \
            --n_traj 4 \
            --output configs/dataset/subset_configs/subset_a5_z5_40pct.json

        # ── All 9 zeta, all 5 alpha, 2 traj/pair ≈ 40% (9/5 ratio)
        echo 'subset_a5_z9_40pct.json ...'
        \${SCRIPT} \
            --alpha -1 -2 -3 -4 -5 \
            --zeta 1 3 5 7 9 11 13 15 17 \
            --n_traj 2 \
            --output configs/dataset/subset_configs/subset_a5_z9_40pct.json

        # ── Sparse: 3 alpha × 3 zeta, 2 traj/pair ≈ 10%
        echo 'subset_a3_z3_10pct.json ...'
        \${SCRIPT} \
            --alpha -1 -3 -5 \
            --zeta 1 9 17 \
            --n_traj 2 \
            --output configs/dataset/subset_configs/subset_a3_z3_10pct.json

        # ── Medium: 3 alpha × 5 zeta, 2 traj/pair ≈ 15%
        echo 'subset_a3_z5_15pct.json ...'
        \${SCRIPT} \
            --alpha -1 -3 -5 \
            --zeta 1 5 9 13 17 \
            --n_traj 2 \
            --output configs/dataset/subset_configs/subset_a3_z5_15pct.json

        # ── Full phase space, all trajectories (100%)
        echo 'subset_a5_z9_100pct.json ...'
        \${SCRIPT} \
            --alpha -1 -2 -3 -4 -5 \
            --zeta 1 3 5 7 9 11 13 15 17 \
            --output configs/dataset/subset_configs/subset_a5_z9_100pct.json

        echo '=== Done. Files written to configs/dataset/subset_configs/ ==='
        ls -lh configs/dataset/subset_configs/
    "
