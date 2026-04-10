# V-JEPA 2 for Active Matter Physical Simulation

**NYU Deep Learning — Spring 2026 Final Project**

Self-supervised representation learning on the `polymathic-ai/active_matter` dataset
using a Video JEPA 2 (Joint-Embedding Predictive Architecture) approach.

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

### 2. Download data (on HPC — run from dtn.torch.hpc.nyu.edu)

```bash
export NETID=<your_netid>
bash download_data.sh
```

### 3. Run training (on HPC)

```bash
# Interactive
jupyter notebook vjepa2_active_matter.ipynb

# Batch (SLURM)
sbatch slurm_train.sh
```

---

## Evaluation

Training periodically evaluates frozen representations (every 10 epochs) via:

- **Linear Probe** — Ridge regression on z-scored α and ζ (MSE)
- **kNN Regression** — cosine-distance kNN (k=20) on z-scored α and ζ (MSE)

No labels are used during pre-training. The backbone is frozen for evaluation.

---

## Constraints

- No pretrained weights
- No external data
- Evaluation: frozen backbone only (no fine-tuning)
- Parameter count < 100M
- Labels (α, ζ) used only for evaluation, not training

---

## References

- [V-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2312.15638)
- [EB-JEPA GitHub](https://github.com/facebookresearch/jepa)
- [The Well / active_matter](https://huggingface.co/datasets/polymathic-ai/active_matter)
- [Baseline paper: arXiv:2603.13227](https://arxiv.org/abs/2603.13227)
