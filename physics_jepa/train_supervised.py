"""
Fully supervised baseline trainer (linear-head version).

Trains the same ConvEncoder used in the SSL pipelines, but end-to-end with a
*single linear layer* head on top of globally-pooled features and MSE loss on
the physical-parameter regression task (e.g. (alpha, zeta) for active matter).
Both the encoder AND the linear layer are optimized — there is no frozen-
feature step. Use this as the apples-to-apples comparison against your
V-JEPA-pretrain + linear-probe results: same encoder architecture, same input
data, same label normalization, same global-average-pool + linear head, same
val set; only the training objective differs (end-to-end MSE-on-labels vs
SSL-pretrain-then-probe).

Reads the same YAML schema as the JEPA configs but with `model.objective:
supervised`. We build the encoder ourselves (rather than going through the
base Trainer's supervised branch, which hardcodes an AttentiveClassifier head)
so that the head is a pure `nn.Linear(embed_dim, num_classes)`.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from omegaconf import OmegaConf

from .data import get_dataset_metadata
from .model import get_model_and_loss_cnn
from .train import Trainer
from .utils.hydra import compose
from .utils.data_utils import normalize_labels
from .utils.misc import distprint
from .utils.model_summary import summarize_convs


# ============================================================================
# LABEL NORMALIZATION STATS
# ----------------------------------------------------------------------------
# Hardcoded per-dataset means / stds used to standardize the physical-parameter
# targets before computing MSE. These MUST match the constants in
# physics_jepa/finetuner.py so that supervised-baseline val/loss is directly
# comparable to the SSL+probe pipeline's val/loss numbers.
# ============================================================================
LABEL_STATS = {
    "active_matter": {
        "means": [-3.0, 9.0],     # alpha, zeta
        "stds":  [1.41, 5.16],
    },
    "shear_flow": {
        "means": [4.85, 2.69],    # rayleigh, schmidt
        "stds":  [0.61, 3.38],
        "compression": ["log", None],
    },
    "rayleigh_benard": {
        "means": [2.69, 8.0],     # prandtl, rayleigh
        "stds":  [3.38, 1.41],
        "compression": [None, "log"],
    },
}


# ============================================================================
# SUPERVISED TRAINER
# ----------------------------------------------------------------------------
# Subclass of the base Trainer. We override get_model_components so the head
# is a single nn.Linear(embed_dim, num_classes) on top of globally-pooled
# encoder features — matching the linear-probe head used in the FT pipeline
# when use_attentive_pooling=False. The pred_fn:
#   1. Runs the encoder on the context frames.
#   2. Globally average-pools the feature map over all spatial / temporal
#      dims to a [B, embed_dim] vector.
#   3. Applies the single linear layer to get [B, num_classes] predictions.
#   4. Normalizes labels via LABEL_STATS and returns MSE.
# ============================================================================
class SupervisedTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        assert cfg.dataset.name in LABEL_STATS, (
            f"No label stats configured for dataset {cfg.dataset.name}"
        )
        self.label_stats = LABEL_STATS[cfg.dataset.name]
        self.label_name = "physical_params"

    # ------------------------------------------------------------------
    # Build [encoder, linear_head]. We don't go through the base Trainer's
    # supervised branch because it hardcodes an AttentiveClassifier head;
    # we want a single linear layer instead.
    # ------------------------------------------------------------------
    def get_model_components(self):
        encoder, _, _ = get_model_and_loss_cnn(
            self.cfg.model.dims,
            self.cfg.model.num_res_blocks,
            self.cfg.dataset.num_frames,
            in_chans=self.cfg.dataset.num_chans
            if 'fields' not in self.train_cfg
            else len(self.train_cfg.fields),
        )

        metadata = get_dataset_metadata(self.cfg.dataset.name)
        num_classes = len(metadata.constant_scalar_names)
        head = nn.Linear(self.cfg.model.dims[-1], num_classes)

        distprint(
            f"num encoder parameters: {sum(p.numel() for p in encoder.parameters())}",
            local_rank=self.rank,
        )
        distprint(
            f"num head parameters: {sum(p.numel() for p in head.parameters())}",
            local_rank=self.rank,
        )
        distprint(summarize_convs(encoder), local_rank=self.rank)

        # Move to device. (DDP wrapping in the base Trainer is already broken;
        # follow the same pattern here for consistency.)
        encoder = encoder.to(self.rank)
        head = head.to(self.rank)

        loss_fn = nn.MSELoss()
        return [encoder, head], loss_fn

    def pred_fn(self, batch, model_components, loss_fn):
        encoder, head = model_components[0], model_components[1]

        # 1. Encoder forward — produces [B, C, H, W] (time has been squeezed
        # by the encoder's final stage) for 16-frame inputs, or [B, C, T, H, W]
        # for 4-frame inputs that didn't fully collapse the time dim.
        ctx = batch['context']
        feat = encoder(ctx)

        # 2. Global average pool over all non-batch, non-channel dims so the
        # linear head sees a [B, embed_dim] vector. flatten(2).mean(-1) handles
        # both 4D ([B, C, H, W] -> [B, C, H*W]) and 5D ([B, C, T, H, W] ->
        # [B, C, T*H*W]) feature maps.
        pooled = feat.flatten(2).mean(dim=-1)

        # 3. Single linear layer -> [B, num_classes].
        pred = head(pooled)

        # 4. Normalize labels into standardized space — same transform the
        # finetuner uses, so the resulting MSE is directly comparable.
        labels = normalize_labels(
            batch[self.label_name], stats=self.label_stats
        ).to(self.rank)

        loss = loss_fn(pred, labels)
        loss_dict = {"loss": loss}

        # Per-target MSE for regression — surfaces α (dim_0) vs ζ (dim_1)
        # separately. gather_losses_and_report prefixes train/ or val/ to
        # every key, so this lands as train/loss_dim_0, val/loss_dim_0, etc.
        if pred.dim() == 2:
            per_dim = ((pred.detach() - labels.detach()) ** 2).mean(dim=0)
            for i in range(per_dim.shape[0]):
                loss_dict[f"loss_dim_{i}"] = per_dim[i]

        return pred, loss_dict


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        default=f"{Path(__file__).parent.parent}/configs/train_activematter_supervised.yaml",
    )
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run

    # Force supervised objective regardless of what the YAML says — defensive.
    cfg.model.objective = "supervised"

    print(OmegaConf.to_yaml(cfg, resolve=True))
    trainer = SupervisedTrainer(cfg)
    trainer.train()
