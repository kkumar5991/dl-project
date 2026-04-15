import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.metrics import mean_squared_error, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from .data import get_train_dataloader, get_val_dataloader, get_dataset_metadata
from .model import get_model_and_loss_cnn
from .utils.data_utils import normalize_labels
from .utils.hydra import compose
from .videomae import vit_small_patch16_224, vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch16_224

import json
import h5py
import os
import wandb


def extract_embeddings(encoder, dataloader, label_stats, device, model_type="jepa", use_attentive_pooling=False):
    """Extract embeddings and labels from a dataloader using a frozen encoder."""
    all_embeddings = []
    all_labels = []

    encoder.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            ctx = batch['context'].to(device)
            if ctx.shape[2] < 4:
                ctx = F.pad(ctx, (0, 0, 0, 0, 0, 4 - ctx.shape[2]))

            labels = normalize_labels(batch['physical_params'], stats=label_stats)

            if model_type == "jepa":
                enc_ctx = encoder(ctx)
                if use_attentive_pooling:
                    # (B, C, H, W) -> (B, H*W, C)
                    from einops import rearrange
                    enc_ctx = rearrange(enc_ctx, 'b c h w -> b (h w) c')
                    enc_ctx = enc_ctx.mean(dim=1)  # global average pool over tokens
                else:
                    # (B, C, H, W) -> (B, C) via global average pooling
                    enc_ctx = enc_ctx.mean(dim=(-2, -1))
            elif model_type == "videomae":
                enc_ctx = encoder(ctx)
                if enc_ctx.dim() == 3:
                    enc_ctx = enc_ctx.mean(dim=1)  # pool over tokens
                elif enc_ctx.dim() == 4:
                    enc_ctx = enc_ctx.mean(dim=(-2, -1))

            all_embeddings.append(enc_ctx.cpu().numpy())
            all_labels.append(labels.numpy())

    return np.concatenate(all_embeddings, axis=0), np.concatenate(all_labels, axis=0)


def load_jepa_encoder(cfg, trained_model_path):
    """Load a pretrained JEPA encoder."""
    encoder, _, _ = get_model_and_loss_cnn(
        cfg.model.dims,
        cfg.model.num_res_blocks,
        cfg.dataset.num_frames,
        in_chans=cfg.dataset.num_chans if not cfg.ft.get('fields', None) else len(cfg.ft.fields),
    )
    if trained_model_path is not None:
        print(f"Loading state dict from {trained_model_path}")
        state_dict = torch.load(trained_model_path, map_location='cpu')
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        encoder.load_state_dict(state_dict)
    else:
        print("No pretrained model path provided, using random initialization")
    encoder.eval()
    return encoder


def load_videomae_encoder(cfg, trained_model_path):
    """Load a pretrained VideoMAE encoder."""
    if trained_model_path is not None:
        model_config = json.load(open(Path(trained_model_path).parent / "config.json"))
        model_arch = model_config["model"]
    else:
        model_arch = 'pretrain_videomae_small_patch16_224'

    model_functions = {
        'pretrain_videomae_small_patch16_224': vit_small_patch16_224,
        'pretrain_videomae_base_patch16_224': vit_base_patch16_224,
        'pretrain_videomae_large_patch16_224': vit_large_patch16_224,
        'pretrain_videomae_huge_patch16_224': vit_huge_patch16_224,
    }

    encoder = model_functions[model_arch](
        pretrained=False,
        drop_path_rate=0.0,
        in_chans=cfg.dataset.num_chans if not cfg.ft.get('fields', None) else len(cfg.ft.fields),
        all_frames=cfg.dataset.num_frames,
        num_classes=0,
        use_mean_pooling=False,
    )

    if trained_model_path is not None:
        checkpoint = torch.load(trained_model_path, map_location='cpu')
        pretrained_state_dict = checkpoint['model']
        encoder_state_dict = {}
        for key, value in pretrained_state_dict.items():
            if key.startswith('encoder.'):
                new_key = key[8:]
                if new_key in encoder.state_dict():
                    encoder_state_dict[new_key] = value
        encoder.load_state_dict(encoder_state_dict, strict=False)
    else:
        print("No pretrained model path provided, using random initialization")

    encoder.eval()
    return encoder


def run_knn_evaluation(cfg, trained_model_path=None):
    """Run KNN evaluation on pretrained encoder representations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Label statistics for normalization
    STATS = {
        "active_matter": {
            "means": [-3.0, 9.0],
            "stds": [1.41, 5.16],
        },
        "shear_flow": {
            "means": [4.85, 2.69],
            "stds": [0.61, 3.38],
            "compression": ["log", None],
        },
        "rayleigh_benard": {
            "means": [2.69, 8.0],
            "stds": [3.38, 1.41],
            "compression": [None, "log"],
        },
    }
    label_stats = STATS[cfg.dataset.name]

    # Load encoder
    if cfg.model.objective == "jepa":
        encoder = load_jepa_encoder(cfg, trained_model_path)
    elif cfg.model.objective == "videomae":
        encoder = load_videomae_encoder(cfg, trained_model_path)
    else:
        raise ValueError(f"Unknown objective: {cfg.model.objective}")

    encoder.to(device)
    for param in encoder.parameters():
        param.requires_grad = False

    # Create data loaders
    batch_size = cfg.ft.get("batch_size", 32)
    train_loader = get_train_dataloader(
        cfg.dataset.name,
        cfg.dataset.num_frames,
        cfg.dataset.get("num_examples", None),
        batch_size,
        shuffle=False,
        include_labels=True,
        predict_n_steps=False,
        task=cfg.ft.get("task", "regression"),
        fields=cfg.ft.get("fields", None),
        resolution=cfg.dataset.get("resolution", None),
        offset=cfg.dataset.get("offset", None),
        noise_std=0.0,
    )
    val_loader = get_val_dataloader(
        cfg.dataset.name,
        cfg.dataset.num_frames,
        cfg.dataset.get("num_examples", None),
        batch_size,
        shuffle=False,
        include_labels=True,
        predict_n_steps=False,
        task=cfg.ft.get("task", "regression"),
        fields=cfg.ft.get("fields", None),
        resolution=cfg.dataset.get("resolution", None),
        offset=cfg.dataset.get("offset", None),
        noise_std=0.0,
    )

    model_type = cfg.model.objective
    use_attentive_pooling = cfg.ft.get("use_attentive_pooling", False)

    # Extract embeddings
    print("Extracting train embeddings...")
    train_embeddings, train_labels = extract_embeddings(
        encoder, train_loader, label_stats, device, model_type, use_attentive_pooling
    )
    print(f"Train embeddings shape: {train_embeddings.shape}, labels shape: {train_labels.shape}")

    print("Extracting val embeddings...")
    val_embeddings, val_labels = extract_embeddings(
        encoder, val_loader, label_stats, device, model_type, use_attentive_pooling
    )
    print(f"Val embeddings shape: {val_embeddings.shape}, labels shape: {val_labels.shape}")

    # Normalize embeddings
    scaler = StandardScaler()
    train_embeddings = scaler.fit_transform(train_embeddings)
    val_embeddings = scaler.transform(val_embeddings)

    # KNN evaluation
    k_values = cfg.ft.get("k_values", [1, 3, 5, 10, 20])
    task = cfg.ft.get("task", "regression")
    metric = cfg.ft.get("knn_metric", "euclidean")
    weights = cfg.ft.get("knn_weights", "distance")

    metadata = get_dataset_metadata(cfg.dataset.name)
    param_names = metadata.constant_scalar_names

    results = {}

    print(f"\n{'='*60}")
    print(f"KNN Evaluation - Task: {task}, Metric: {metric}, Weights: {weights}")
    print(f"Dataset: {cfg.dataset.name}, Model: {cfg.model.objective}")
    print(f"{'='*60}\n")

    for k in k_values:
        print(f"\n--- K = {k} ---")
        if task == "regression":
            knn = KNeighborsRegressor(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1)
            knn.fit(train_embeddings, train_labels)

            train_preds = knn.predict(train_embeddings)
            val_preds = knn.predict(val_embeddings)

            train_mse = mean_squared_error(train_labels, train_preds)
            val_mse = mean_squared_error(val_labels, val_preds)

            results[k] = {"train_mse": train_mse, "val_mse": val_mse}

            print(f"  Train MSE: {train_mse:.6f}")
            print(f"  Val MSE:   {val_mse:.6f}")

            # Per-parameter MSE
            for j, name in enumerate(param_names):
                param_train_mse = mean_squared_error(train_labels[:, j], train_preds[:, j])
                param_val_mse = mean_squared_error(val_labels[:, j], val_preds[:, j])
                results[k][f"train_mse_{name}"] = param_train_mse
                results[k][f"val_mse_{name}"] = param_val_mse
                print(f"  {name} - Train MSE: {param_train_mse:.6f}, Val MSE: {param_val_mse:.6f}")

        elif task in ("classification", "binary_classification"):
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric, weights=weights, n_jobs=-1)
            knn.fit(train_embeddings, train_labels.ravel() if train_labels.ndim > 1 and train_labels.shape[1] == 1 else train_labels)

            val_preds = knn.predict(val_embeddings)
            val_labels_flat = val_labels.ravel() if val_labels.ndim > 1 and val_labels.shape[1] == 1 else val_labels

            val_acc = accuracy_score(val_labels_flat, val_preds)
            val_f1 = f1_score(val_labels_flat, val_preds, average='macro', zero_division=0)

            results[k] = {"val_accuracy": val_acc, "val_macro_f1": val_f1}
            print(f"  Val Accuracy: {val_acc:.4f}")
            print(f"  Val Macro F1: {val_f1:.4f}")

    # Log to wandb if not dry run
    if not cfg.get("dry_run", False):
        run_name = cfg.ft.get("run_name",
            f"{cfg.dataset.name}-{cfg.dataset.num_frames}frames-{cfg.model.objective}-knn"
            f"{'-randominit' if trained_model_path is None else ''}")
        wandb.init(project="physics-jepa", name=run_name,
                   config=OmegaConf.to_container(cfg))
        for k, metrics in results.items():
            for metric_name, value in metrics.items():
                wandb.log({f"knn/k{k}/{metric_name}": value})
        wandb.finish()

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if task == "regression":
        best_k = min(results, key=lambda k: results[k]["val_mse"])
        print(f"Best K: {best_k} (Val MSE: {results[best_k]['val_mse']:.6f})")
    else:
        best_k = max(results, key=lambda k: results[k]["val_accuracy"])
        print(f"Best K: {best_k} (Val Accuracy: {results[best_k]['val_accuracy']:.4f})")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KNN evaluation of Physics JEPA representations")
    parser.add_argument("config", type=str, help="Path to config YAML file")
    parser.add_argument("overrides", nargs="*", help="Hydra-style config overrides")
    parser.add_argument("--trained_model_path", type=str, default=None,
                        help="Path to pretrained encoder checkpoint")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run
    cfg.seed = args.seed

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(OmegaConf.to_yaml(cfg, resolve=True))

    results = run_knn_evaluation(cfg, trained_model_path=args.trained_model_path)
