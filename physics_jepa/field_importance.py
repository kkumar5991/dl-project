"""
Field Importance Evaluation via Representation MSE.

For each field, zero-mask its channels in every training sample, encode both the
full and masked inputs, and compute MSE(z_full, z_masked).  A higher MSE means
the encoder relies more heavily on that field — i.e. the field is more important
to the learned representation.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .data import get_train_dataloader
from .model import get_model_and_loss_cnn
from .utils.hydra import compose

# ── field definitions (active-matter, matches train_jepa.py) ──────────────
FIELD_GROUPS = [[0], [1, 2], [3, 4, 5, 6], [7, 8, 9, 10]]
FIELD_NAMES = ["concentration", "velocity", "orientation", "strain"]


def load_encoder(cfg, trained_model_path):
    """Instantiate ConvEncoder and load pretrained weights."""
    encoder, _, _ = get_model_and_loss_cnn(
        cfg.model.dims,
        cfg.model.num_res_blocks,
        cfg.dataset.num_frames,
        in_chans=cfg.dataset.num_chans,
    )
    if trained_model_path is not None:
        print(f"Loading checkpoint from {trained_model_path}")
        checkpoint = torch.load(trained_model_path, map_location="cpu")
        if "model_states" in checkpoint:
            state_dict = checkpoint["model_states"][encoder.__class__.__name__]
        else:
            state_dict = checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict)
    else:
        print("No checkpoint provided – using randomly initialised encoder")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def evaluate_field_importance(cfg, encoder, device="cuda"):
    """Run the full training set through the encoder with and without each
    field and return per-field average MSE."""

    loader = get_train_dataloader(
        dataset_name=cfg.dataset.name,
        num_frames=cfg.dataset.num_frames,
        num_examples=cfg.dataset.get("num_examples", None),
        batch_size=cfg.ft.batch_size,
        shuffle=False,
        include_labels=False,
        predict_n_steps=False,
        resolution=cfg.dataset.get("resolution", None),
        offset=cfg.dataset.get("offset", None),
    )

    encoder.to(device)
    num_fields = len(FIELD_GROUPS)

    # accumulators: sum of MSE and count of batches per field
    mse_sums = np.zeros(num_fields, dtype=np.float64)
    num_batches = 0

    for batch in tqdm(loader, desc="field-importance"):
        ctx = batch["context"].to(device)  # (B, C, T, H, W)
        if ctx.shape[2] < 4:
            ctx = F.pad(ctx, (0, 0, 0, 0, 0, 4 - ctx.shape[2]))

        z_full = encoder(ctx)  # (B, C', T', H', W')

        for f_idx, channels in enumerate(FIELD_GROUPS):
            ctx_masked = ctx.clone()
            ctx_masked[:, channels] = 0.0
            z_masked = encoder(ctx_masked)

            mse = F.mse_loss(z_full, z_masked).item()
            mse_sums[f_idx] += mse

        num_batches += 1

    avg_mse = mse_sums / max(num_batches, 1)

    # also compute per-channel-normalised MSE (divide by number of channels in each field)
    field_sizes = np.array([len(ch) for ch in FIELD_GROUPS], dtype=np.float64)
    avg_mse_per_channel = avg_mse / field_sizes

    return avg_mse, avg_mse_per_channel


def print_results(avg_mse, avg_mse_per_channel):
    print(f"\n{'='*60}")
    print("Field Importance — Representation MSE")
    print(f"{'='*60}")
    print(f"{'Field':<20} {'# Channels':<12} {'Avg MSE':<15} {'MSE/Channel':<15}")
    print(f"{'-'*60}")
    for i, name in enumerate(FIELD_NAMES):
        n_ch = len(FIELD_GROUPS[i])
        print(f"{name:<20} {n_ch:<12} {avg_mse[i]:<15.6f} {avg_mse_per_channel[i]:<15.6f}")
    print(f"{'='*60}\n")

    # rank by importance
    ranking = np.argsort(-avg_mse)
    print("Ranking (most → least important by raw MSE):")
    for rank, idx in enumerate(ranking, 1):
        print(f"  {rank}. {FIELD_NAMES[idx]} (MSE={avg_mse[idx]:.6f})")
    print()


def plot_results(avg_mse, avg_mse_per_channel, save_path="field_importance.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(FIELD_NAMES))

    axes[0].bar(x, avg_mse, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(FIELD_NAMES, rotation=30, ha="right")
    axes[0].set_ylabel("Average MSE")
    axes[0].set_title("Field Importance (raw MSE)")

    axes[1].bar(x, avg_mse_per_channel, color="coral")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(FIELD_NAMES, rotation=30, ha="right")
    axes[1].set_ylabel("Average MSE / channel")
    axes[1].set_title("Field Importance (per-channel MSE)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Bar chart saved to {save_path}")
    return fig


def try_log_wandb(cfg, avg_mse, avg_mse_per_channel, fig_path):
    try:
        import wandb
    except ImportError:
        return

    run_name = cfg.ft.get(
        "run_name",
        f"{cfg.dataset.name}-{cfg.dataset.num_frames}frames-field-importance",
    )
    wandb.init(project="physics-jepa", name=run_name,
               config=OmegaConf.to_container(cfg))

    for i, name in enumerate(FIELD_NAMES):
        wandb.log({
            f"field_importance/mse_{name}": avg_mse[i],
            f"field_importance/mse_per_channel_{name}": avg_mse_per_channel[i],
        })

    columns = ["field", "num_channels", "avg_mse", "mse_per_channel"]
    table = wandb.Table(columns=columns)
    for i, name in enumerate(FIELD_NAMES):
        table.add_data(name, len(FIELD_GROUPS[i]), avg_mse[i], avg_mse_per_channel[i])
    wandb.log({"field_importance/results_table": table})

    if Path(fig_path).exists():
        wandb.log({"field_importance/bar_chart": wandb.Image(fig_path)})

    wandb.finish()


# ── CLI ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate field importance via representation MSE")
    parser.add_argument("config", type=str, help="Path to config YAML file")
    parser.add_argument("overrides", nargs="*",
                        help="Hydra-style config overrides")
    parser.add_argument("--trained_model_path", type=str, default=None,
                        help="Path to pretrained encoder checkpoint")
    parser.add_argument("--save_path", type=str, default="field_importance.png",
                        help="Where to save the bar-chart image")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run

    print(OmegaConf.to_yaml(cfg, resolve=True))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = load_encoder(cfg, args.trained_model_path)

    avg_mse, avg_mse_per_channel = evaluate_field_importance(
        cfg, encoder, device=device)

    print_results(avg_mse, avg_mse_per_channel)
    plot_results(avg_mse, avg_mse_per_channel, save_path=args.save_path)

    if not args.dry_run:
        try_log_wandb(cfg, avg_mse, avg_mse_per_channel, args.save_path)
