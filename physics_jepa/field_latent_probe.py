"""
Field Latent Probe: predict per-field encoder representations from full-model
encoder representations using a linear probe.

Usage:
    python -m physics_jepa.field_latent_probe \
        configs/train_activematter_small.yaml \
        --full_ckpt_path  checkpoints/full-model/latest.pt \
        --field_ckpt_path checkpoints/velocity-only/latest.pt \
        --field_indices 1 2 \
        --num_epochs 50 \
        --lr 1e-3
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from .data import get_train_dataloader, get_val_dataloader
from .model import get_model_and_loss_cnn
from .utils.hydra import compose


def load_encoder(cfg, ckpt_path, in_chans):
    """Create a ConvEncoder with the given in_chans and load weights."""
    encoder, _, _ = get_model_and_loss_cnn(
        cfg.model.dims,
        cfg.model.num_res_blocks,
        cfg.dataset.num_frames,
        in_chans=in_chans,
    )
    if ckpt_path is not None:
        print(f"Loading encoder from {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        if "model_states" in checkpoint:
            state_dict = checkpoint["model_states"][encoder.__class__.__name__]
        else:
            state_dict = checkpoint
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict)
    else:
        print("No checkpoint — using random init")

    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False
    return encoder


@torch.no_grad()
def compute_embeddings(loader, full_encoder, field_encoder, field_indices, device):
    """Encode the full context and the field-only context, return flattened pairs."""
    full_list, field_list = [], []
    for batch in tqdm(loader, desc="encoding"):
        ctx = batch["context"].to(device)  # (B, C, T, H, W)
        if ctx.shape[2] < 4:
            ctx = F.pad(ctx, (0, 0, 0, 0, 0, 4 - ctx.shape[2]))

        z_full = full_encoder(ctx)                       # (B, D, H', W')
        z_field = field_encoder(ctx[:, field_indices])    # (B, D, H', W')

        full_list.append(z_full.flatten(1).cpu())   # (B, D*H'*W')
        field_list.append(z_field.flatten(1).cpu())

    return torch.cat(full_list), torch.cat(field_list)


def train_probe(z_full_train, z_field_train, z_full_val, z_field_val,
                num_epochs, lr, weight_decay, batch_size, device):
    """Train a linear probe mapping z_full -> z_field."""
    in_dim = z_full_train.shape[1]
    out_dim = z_field_train.shape[1]
    probe = nn.Linear(in_dim, out_dim).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)

    n_train = z_full_train.shape[0]
    best_val_mse = float("inf")

    for epoch in range(num_epochs):
        probe.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            idx = perm[start : start + batch_size]
            x = z_full_train[idx].to(device)
            y = z_field_train[idx].to(device)

            pred = probe(x)
            loss = F.mse_loss(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_mse = epoch_loss / max(n_batches, 1)

        # Validation
        probe.eval()
        with torch.no_grad():
            val_pred = probe(z_full_val.to(device))
            val_target = z_field_val.to(device)
            val_mse = F.mse_loss(val_pred, val_target).item()

            # R² score
            ss_res = ((val_pred - val_target) ** 2).sum().item()
            ss_tot = ((val_target - val_target.mean(dim=0)) ** 2).sum().item()
            r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

        if val_mse < best_val_mse:
            best_val_mse = val_mse

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs}  "
                  f"train_mse={train_mse:.6f}  val_mse={val_mse:.6f}  R²={r2:.4f}")

    # Final report
    print(f"\nBest val MSE: {best_val_mse:.6f}  |  Final R²: {r2:.4f}")

    # Random baseline
    with torch.no_grad():
        rand_probe = nn.Linear(in_dim, out_dim).to(device)
        rand_pred = rand_probe(z_full_val.to(device))
        rand_mse = F.mse_loss(rand_pred, z_field_val.to(device)).item()
        ss_res_rand = ((rand_pred - val_target) ** 2).sum().item()
        r2_rand = 1.0 - ss_res_rand / max(ss_tot, 1e-8)
    print(f"Random baseline — val_mse={rand_mse:.6f}  R²={r2_rand:.4f}")

    return probe


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Linear probe: full-model z → per-field z")
    parser.add_argument("config", type=str, help="Path to config YAML")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--full_ckpt_path", type=str, required=True,
                        help="Checkpoint for the full (all-channel) encoder")
    parser.add_argument("--field_ckpt_path", type=str, required=True,
                        help="Checkpoint for the per-field encoder")
    parser.add_argument("--field_indices", type=int, nargs="+", required=True,
                        help="Channel indices for the field (e.g. 1 2 for velocity)")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--save_path", type=str, default=None,
                        help="Save probe weights to this path")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    field_indices = args.field_indices

    print(f"Field indices: {field_indices}  →  in_chans={len(field_indices)}")

    # Load both encoders
    full_encoder = load_encoder(cfg, args.full_ckpt_path,
                                in_chans=cfg.dataset.num_chans).to(device)
    field_encoder = load_encoder(cfg, args.field_ckpt_path,
                                 in_chans=len(field_indices)).to(device)

    # Build dataloaders (no labels needed)
    train_loader = get_train_dataloader(
        dataset_name=cfg.dataset.name,
        num_frames=cfg.dataset.num_frames,
        num_examples=cfg.dataset.get("num_examples", None),
        batch_size=cfg.ft.batch_size,
        shuffle=False,
        include_labels=False,
        predict_n_steps=False,
        resolution=cfg.dataset.get("resolution", None),
        offset=cfg.dataset.get("offset", None),
        noise_std=cfg.train.get("noise_std", 0.0),
    )
    val_loader = get_val_dataloader(
        dataset_name=cfg.dataset.name,
        num_frames=cfg.dataset.num_frames,
        num_examples=cfg.dataset.get("num_examples", None),
        batch_size=cfg.ft.batch_size,
        shuffle=False,
        include_labels=False,
        predict_n_steps=False,
        resolution=cfg.dataset.get("resolution", None),
        offset=cfg.dataset.get("offset", None),
        noise_std=0.0,
    )

    print("Computing train embeddings...")
    z_full_train, z_field_train = compute_embeddings(
        train_loader, full_encoder, field_encoder, field_indices, device)
    print(f"  z_full_train: {z_full_train.shape}  z_field_train: {z_field_train.shape}")

    print("Computing val embeddings...")
    z_full_val, z_field_val = compute_embeddings(
        val_loader, full_encoder, field_encoder, field_indices, device)
    print(f"  z_full_val: {z_full_val.shape}  z_field_val: {z_field_val.shape}")

    print("\nTraining linear probe...")
    probe = train_probe(
        z_full_train, z_field_train,
        z_full_val, z_field_val,
        num_epochs=args.num_epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        device=device,
    )

    if args.save_path:
        Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(probe.state_dict(), args.save_path)
        print(f"Probe saved to {args.save_path}")
