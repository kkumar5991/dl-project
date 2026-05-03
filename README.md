# V-JEPA 2 for Active Matter Physical Simulation

**NYU Deep Learning — Spring 2026 Final Project**

Self-supervised representation learning on the `polymathic-ai/active_matter` dataset
using a Video JEPA 2 (Joint-Embedding Predictive Architecture) approach, plus an
end-to-end supervised baseline. Frozen encoders are evaluated with both **linear
probe** and **kNN regression** on the (α, ζ) physical parameters, per the project
spec.

---

## Method

**V-JEPA 2** learns representations by predicting latent patch embeddings of masked
spatiotemporal regions, using an EMA-stabilised target encoder.

| Component | Architecture | Params |
|---|---|---|
| Context Encoder | ViT-Small (depth=12, dim=384, heads=6) | ~2.5M |
| Target Encoder | EMA copy of context encoder | — |
| Predictor | Narrow ViT (depth=6, dim=192, heads=6) | ~0.8M |
| **Total** | | **~3.3M** |

> Well within the 100M parameter budget.

**Input:** 16 frames × 224×224 × 11 physical channels  
**Patching:** 3D tubelets (2 × 16 × 16) → 1568 patches per clip  
**Masking:** Multi-block 3D masking — context encoder sees ~85%, predicts ~15-20%

---

## Dataset

`polymathic-ai/active_matter` — 52 GB, from [The Well](https://huggingface.co/polymathic-ai).

| Split | Samples |
|---|---|
| Train | 8,750 |
| Validation | 1,200 |
| Test | 1,300 |

**Physical parameters (evaluation labels only):**
- α — active dipole strength (5 discrete values)
- ζ — steric alignment (9 discrete values)

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download data (on HPC — run from `dtn.torch.hpc.nyu.edu`)

```bash
export NETID=<your_netid>
bash download_data.sh
```

---

## Running on HPC (SLURM)

All training and evaluation is launched through two SLURM scripts:

```bash
# Self-supervised / supervised pre-training
sbatch slurm_train.sh

# Linear-probe or kNN evaluation of a frozen encoder
sbatch slurm_fine_tune.sh
```

Each SLURM script wraps a Singularity container, sets the env, and at the end
invokes a single line of the form:

```bash
bash scripts/active_matter/<run_script>.sh [args]
```

To switch between pre-training methods or evaluation modes, **edit only that
final `bash scripts/active_matter/...` line** in the relevant SLURM script.
The SBATCH directives, container setup, and signal-handling block don't need
to change.

### `slurm_train.sh` — pre-training

Replace the inner `bash scripts/...` line with one of:

| Goal | Inner command |
|---|---|
| V-JEPA (EMA teacher, smooth-L1) | `bash scripts/active_matter/run_train_vjepa.sh` |
| Original CNN-JEPA (VICReg) | `bash scripts/active_matter/run_train_jepa.sh` |
| Channel-wise JEPA ablation | `bash scripts/active_matter/run_train_channel_jepa.sh` |
| Supervised baseline (end-to-end, linear head) | `bash scripts/active_matter/run_train_supervised.sh` |

> kNN cannot be trained end-to-end (no learnable parameters, neighbor selection
> is non-differentiable), so the only end-to-end "supervised" option is the
> linear-head one above. kNN appears only on the evaluation side.

### `slurm_fine_tune.sh` — frozen-encoder evaluation

Pass the encoder checkpoint as the script argument. Pretrained checkpoints
from our runs are committed under [checkpoints/](checkpoints/) — pick the
one matching the encoder you want to evaluate:

| Pretrained encoder | Checkpoint path |
|---|---|
| Original CNN-JEPA (VICReg, baseline) | `checkpoints/active_matter-16frames-cnn-jepa-baseline-full/ConvEncoder_29.pth` |
| EMA-temporal V-JEPA | `checkpoints/active_matter-16frames-cnn-jepa-vjepa-ema-temporal-full/ConvEncoder_29.pth` |
| Field-mask + EMA V-JEPA | `checkpoints/active_matter-16frames-cnn-jepa-vjepa-fieldmask-ema-temporal-full/ConvEncoder_29.pth` |
| Supervised baseline (end-to-end) | `checkpoints/active_matter-16frames-cnn-supervised-baseline-linear/ConvEncoder_29.pth` |

| Goal | Inner command |
|---|---|
| JEPA / V-JEPA → **linear probe** | `bash scripts/active_matter/run_finetune_jepa_linear.sh <ConvEncoder_xx.pth>` |
| JEPA / V-JEPA → **kNN regression** | `bash scripts/active_matter/run_finetune_jepa_knn.sh <ConvEncoder_xx.pth>` |
| Supervised encoder → **linear probe** | `bash scripts/active_matter/run_finetune_supervised_linear.sh <ConvEncoder_xx.pth>` |
| Supervised encoder → **kNN regression** | `bash scripts/active_matter/run_finetune_supervised_knn.sh <ConvEncoder_xx.pth>` |
| VideoMAE finetune (legacy) | `bash scripts/active_matter/run_finetune_videomae.sh <ckpt.pth>` |

> If you train your own encoder, the script writes a fresh
> `ConvEncoder_<epoch>.pth` under `./checkpoints/<run_name>/` — pass that
> path instead.

### Reproducing the reported numbers

Each row of the results tables in the report corresponds to one
(pretraining, evaluation) pair. The table below maps each row to the
exact pretrain command (or the committed checkpoint to skip pretraining)
and the matching evaluation script. Validation MSE is on $z$-scored
$(\alpha,\zeta)$, full train split.

| # | Encoder (pretrain) | Pretrain command | Eval head | Eval command | Approx. val MSE (avg) |
|---|---|---|---|---|---|
| 1 | Baseline JEPA (VICReg) | `bash scripts/active_matter/run_train_jepa.sh` | linear | `run_finetune_jepa_linear.sh checkpoints/active_matter-16frames-cnn-jepa-baseline-full/ConvEncoder_29.pth` | 0.669 |
| 2 | Baseline JEPA (VICReg) | (same as #1) | kNN ($k{=}20$) | `run_finetune_jepa_knn.sh <same ckpt>` | 0.99 |
| 3 | EMA-temporal V-JEPA | `bash scripts/active_matter/run_train_vjepa.sh` | linear | `run_finetune_jepa_linear.sh checkpoints/active_matter-16frames-cnn-jepa-vjepa-ema-temporal-full/ConvEncoder_29.pth` | 0.407 |
| 4 | EMA-temporal V-JEPA | (same as #3) | kNN ($k{=}20$) | `run_finetune_jepa_knn.sh <same ckpt>` | 0.771 |
| 5 | Field-mask EMA V-JEPA | `bash scripts/active_matter/run_train_vjepa.sh` with `train.field_masked=true` (and optionally `train.inverse_target=true train.learnable_mask=true`) | linear | `run_finetune_jepa_linear.sh checkpoints/active_matter-16frames-cnn-jepa-vjepa-fieldmask-ema-temporal-full/ConvEncoder_29.pth` | 0.482 |
| 6 | Field-mask EMA V-JEPA | (same as #5) | kNN ($k{=}20$) | `run_finetune_jepa_knn.sh <same ckpt>` | 0.418 |
| 7 | Supervised baseline | `bash scripts/active_matter/run_train_supervised.sh` | end-to-end linear | (training itself is the eval — see `val/loss` in wandb) | 0.055 |
| 8 | Supervised encoder | (same as #7) | kNN ($k{=}20$) | `run_finetune_supervised_knn.sh checkpoints/active_matter-16frames-cnn-supervised-baseline-linear/ConvEncoder_29.pth` | 0.056 |

All eval commands assume you're at the repo root and that
`scripts/active_matter/...` is the script-name prefix. To skip pretraining
entirely, point the eval script at the matching path under
`checkpoints/`. To re-pretrain from scratch, run the pretrain command in
`slurm_train.sh` first; the encoder file will be written under
`./checkpoints/<run_name>/ConvEncoder_29.pth`.

#### Common Hydra overrides (append to the inner command)

```bash
# Sweep different k values for kNN
'ft.n_neighbors_list=[1,5,15]'
ft.knn_weights=uniform        # default: distance
ft.knn_metric=cosine          # default: minkowski (L2)
'ft.n_neighbors_list=null' ft.n_neighbors=10   # single k, no sweep

# Linear probe knobs
ft.lr=5e-4
ft.num_epochs=200
ft.batch_size=64

# Reseed
--seed 7
```

Example — kNN eval of the field-mask V-JEPA checkpoint with a custom k
sweep, by editing the inner line in `slurm_fine_tune.sh`:

```bash
bash scripts/active_matter/run_finetune_jepa_knn.sh \
    checkpoints/active_matter-16frames-cnn-jepa-vjepa-fieldmask-ema-temporal-full/ConvEncoder_29.pth \
    'ft.n_neighbors_list=[1,3,5,10,20,50,100]' \
    ft.knn_weights=uniform
```

### Logged metrics

For regression tasks, both linear-probe and kNN runs log:

- `val/loss` — mean MSE over both targets in z-scored space
- `val/loss_dim_0` — MSE on **α** (alpha)
- `val/loss_dim_1` — MSE on **ζ** (zeta)

kNN additionally logs `knn/k` per step and `best/k`, `best/val_loss`,
`best/val_loss_dim_*` in `wandb.summary` after the sweep.

---

## Constraints

- No pretrained weights
- No external data
- Evaluation: frozen backbone only (no fine-tuning)
- Parameter count < 100M
- Labels (α, ζ) used only for evaluation, not training

---

## Data Subsets

Subset configs are generated by `generate_all_subsets.sh` and stored in `subset_configs/`.
Each config is a JSON file with `subset_indices` selecting a fraction of the training windows.

The full dataset has **5 alpha × 9 zeta = 45 parameter pairs**, each with multiple trajectories
and many temporal windows per trajectory.

| Config file | Alpha values | Zeta values | Traj/pair | ~Dataset % | Expected use |
|---|---|---|---|---|---|
| `subset_a5_z5_20pct.json` | all 5 (-1…-5) | 5 evenly spaced (1,5,9,13,17) | 2 | ~20% | Baseline ablation with full phase space coverage at low cost |
| `subset_a5_z5_40pct.json` | all 5 (-1…-5) | 5 evenly spaced (1,5,9,13,17) | 4 | ~40% | More data per param pair, same phase space coverage |
| `subset_a5_z9_40pct.json` | all 5 (-1…-5) | all 9 (1,3,5,7,9,11,13,15,17) | 2 | ~40% | Full zeta coverage, tests whether denser zeta sampling helps |
| `subset_a3_z3_10pct.json` | 3 extreme (-1,-3,-5) | 3 extreme (1,9,17) | 2 | ~10% | Minimal subset covering extreme corners of phase space only |
| `subset_a3_z5_15pct.json` | 3 extreme (-1,-3,-5) | 5 evenly spaced (1,5,9,13,17) | 2 | ~15% | Reduced alpha diversity, full zeta range — tests alpha sensitivity |
| `subset_a5_z9_100pct.json` | all 5 (-1…-5) | all 9 (1,3,5,7,9,11,13,15,17) | all | 100% | Full dataset, no subsampling |

**To use a subset config**, add to `configs/train_activematter_small.yaml`:
```yaml
dataset:
  subset_config_path: "./configs/dataset/subset_configs/subset_a5_z5_20pct.json"
```

**To regenerate subset configs** (run on HPC):
```bash
sbatch generate_all_subsets.sh
```

---

## References

- [V-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2312.15638)
- [EB-JEPA GitHub](https://github.com/facebookresearch/jepa)
- [The Well / active_matter](https://huggingface.co/datasets/polymathic-ai/active_matter)
- [Baseline paper: arXiv:2603.13227](https://arxiv.org/abs/2603.13227)

---

## Experiment tracking

All runs reported in the paper (linear-probe and kNN evaluations across
the four encoders) are logged to W&B at:

[https://wandb.ai/ar9799-new-york-university/physics-jepa/overview](https://wandb.ai/ar9799-new-york-university/physics-jepa/overview)

Run-name conventions used in the dashboard:

- `activematter-baseline-full-FT-linear` — baseline JEPA + linear probe
- `activematter-jepa-baseline-full-FT-knn` — baseline JEPA + kNN
- `activematter-vjepa-ema-temporal-full-FT-{linear,knn}` — EMA-temporal V-JEPA
- `activematter-vjepa-fieldmask-ema-temporal-full-FT-{linear,knn}` — field-mask V-JEPA
- `active_matter-16frames-cnn-supervised-baseline-linear` — end-to-end supervised
- `activematter-supervised-baseline-FT-knn` — supervised encoder + kNN
