"""
generate_subset_config.py
--------------------------
Generates a subset_config.json compatible with WellDatasetForJEPA's
subset_config_path argument.

Works by instantiating the dataset (index-only, no data loaded),
then selecting flat indices whose (alpha, zeta) fall in the chosen grid.

Usage:
    THE_WELL_DATA_DIR=/path/to/well python generate_subset_config.py \
        --dataset active_matter \
        --num_frames 4 \
        --output subset_config.json

Then pass subset_config_path="subset_config.json" to get_dataset() or
get_train_dataloader().
"""

import os
import json
import argparse
import numpy as np
from pathlib import Path

# ── Parameter grid to keep ────────────────────────────────────────────────────
# All 5 alpha values, 5 evenly-spaced zeta values → covers full phase space
ALPHA_KEEP = {-1.0, -2.0, -3.0, -4.0, -5.0}
ZETA_KEEP  = {1.0, 5.0, 9.0, 13.0, 17.0}
N_TRAJ_PER_PARAM = None   # how many distinct (obj_id, t0) trajectories to keep
                           # per (alpha, zeta) pair. Set to None to keep all.

# ── Tolerances for float param matching ───────────────────────────────────────
ALPHA_TOL = 0.1
ZETA_TOL  = 0.1


def build_subset_indices(dataset, alpha_keep, zeta_keep, n_traj_per_param):
    """
    Walk dataset.index and select entries whose file's physical params
    match the desired (alpha, zeta) grid.

    dataset.physical_params_idx: Dict[filename -> [alpha_val, zeta_val]]
    (order matches the HDF5 scalars keys alphabetically, with L excluded)

    Note: confirmed from training logs — params are [alpha, zeta] at indices 0 and 1.
    Adjust ALPHA_IDX / ZETA_IDX below if your dataset differs.
    """
    ALPHA_IDX = 0   # ← index of alpha in physical_params_idx values
    ZETA_IDX  = 1   # ← index of zeta  in physical_params_idx values
    # If params print as [zeta, alpha], swap these two constants.

    selected = []
    # Track how many windows we've taken per (alpha, zeta) pair
    # We count distinct (file_id, obj_id) pairs, not t0 windows,
    # so all temporal windows of a kept trajectory are included.
    kept_trajs: dict[tuple, set] = {}  # (alpha, zeta) → set of (file_id, obj_id)

    for flat_idx, (file_id, obj_id, t0) in enumerate(dataset.index):
        params = dataset.physical_params_idx.get(file_id)
        if params is None:
            continue

        alpha_val = float(params[ALPHA_IDX])
        zeta_val  = float(params[ZETA_IDX])

        # Match against desired grid
        alpha_match = any(abs(alpha_val - a) < ALPHA_TOL for a in alpha_keep)
        zeta_match  = any(abs(zeta_val  - z) < ZETA_TOL  for z in zeta_keep)

        if not alpha_match or not zeta_match:
            continue

        # Round to nearest kept value for grouping
        alpha_key = min(alpha_keep, key=lambda a: abs(a - alpha_val))
        zeta_key  = min(zeta_keep,  key=lambda z: abs(z - zeta_val))
        param_key = (alpha_key, zeta_key)

        traj_key = (file_id, obj_id)
        trajs_for_param = kept_trajs.setdefault(param_key, set())

        # Track trajectory for summary reporting (always)
        if traj_key not in trajs_for_param:
            # Enforce trajectory cap if set
            if n_traj_per_param is not None and len(trajs_for_param) >= n_traj_per_param:
                continue  # already have enough distinct trajectories
            trajs_for_param.add(traj_key)

        selected.append(flat_idx)

    return selected, kept_trajs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    default="active_matter")
    parser.add_argument("--num_frames", type=int, default=4,
                        help="Must match your training config")
    parser.add_argument("--split",      default="train")
    parser.add_argument("--output",     default="subset_config.json")
    parser.add_argument("--stride",     type=int, default=None,
                        help="Temporal stride (default: num_frames, i.e. non-overlapping)")
    parser.add_argument("--alpha",      type=float, nargs="+", default=None,
                        help="Alpha values to keep (default: -1 -2 -3 -4 -5)")
    parser.add_argument("--zeta",       type=float, nargs="+", default=None,
                        help="Zeta values to keep (default: 1 5 9 13 17)")
    parser.add_argument("--n_traj",     type=int, default=None,
                        help="Max trajectories per (alpha,zeta) pair (default: all)")
    args = parser.parse_args()

    if args.alpha is not None:
        ALPHA_KEEP.clear()
        ALPHA_KEEP.update(args.alpha)
    if args.zeta is not None:
        ZETA_KEEP.clear()
        ZETA_KEEP.update(args.zeta)
    global N_TRAJ_PER_PARAM
    if args.n_traj is not None:
        N_TRAJ_PER_PARAM = args.n_traj

    well_data_dir = os.environ.get("THE_WELL_DATA_DIR")
    if well_data_dir is None:
        raise ValueError("Set THE_WELL_DATA_DIR environment variable")

    # Import here so the script is runnable in the same env as your training code
    from physics_jepa.data import WellDatasetForJEPA  # adjust import to your module name

    print(f"Building index for {args.dataset} / {args.split} ...")
    ds = WellDatasetForJEPA(
        data_dir=str(Path(well_data_dir) / args.dataset),
        num_frames=args.num_frames,
        split=args.split,
        stride=args.stride,
        subset_config_path=None,   # full dataset for index building
    )

    print(f"\nFull index size: {len(ds.index)} windows")
    print(f"Selecting alpha ∈ {sorted(ALPHA_KEEP)}, zeta ∈ {sorted(ZETA_KEEP)}, "
          f"max {N_TRAJ_PER_PARAM} traj/param_set\n")

    selected, kept_trajs = build_subset_indices(
        ds, ALPHA_KEEP, ZETA_KEEP, N_TRAJ_PER_PARAM
    )

    # ── Summary ──
    print("Kept trajectories per (alpha, zeta):")
    for (alpha, zeta), trajs in sorted(kept_trajs.items()):
        print(f"  alpha={alpha:5.1f}, zeta={zeta:5.1f} → {len(trajs)} traj(s)")

    # Count windows
    print(f"\nTotal windows selected : {len(selected)}")
    print(f"Fraction of full data  : {len(selected)/len(ds.index)*100:.1f}%")

    # Verify no duplicates
    assert len(selected) == len(set(selected)), "Duplicate indices — something went wrong"

    # ── Write JSON ──
    out = Path(args.output)
    out.write_text(json.dumps({"subset_indices": selected}, indent=2))
    print(f"\nWritten to: {out.resolve()}")
    print("\nUsage in training:")
    print(f"  get_dataset('{args.dataset}', num_frames={args.num_frames}, "
          f"subset_config_path='{out}')")


if __name__ == "__main__":
    main()
