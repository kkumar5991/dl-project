"""
V-JEPA trainer (Path A) for the physics_jepa codebase.

WHAT THIS DOES (vs. the original JepaTrainer in train_jepa.py):
  - Keeps the dataloader unchanged: each sample is 32 frames split into
    context (past 16 frames) and target (future 16 frames). The "JEPA task"
    is still: predict the embedding of future frames from the embedding of
    past frames.
  - Replaces the shared-weights target encoder with a momentum-EMA teacher.
    The student (context encoder) gets gradients; the teacher (target
    encoder) is updated by exponential moving average of the student.
  - Replaces the VICReg loss with a smooth-L1 (or MSE) loss in feature
    space. There is no variance/covariance regularization — collapse is
    prevented purely by the teacher being out-of-sync with the student.

WHY THIS MATTERS:
  The original codebase's "JEPA" used shared-weights symmetric training with
  VICReg. That's closer to a contrastive objective than to true V-JEPA.
  V-JEPA's defining feature is the EMA teacher: the student is asked to
  predict a quasi-stable target produced by a slowly-evolving copy of itself.
  This is what makes "predict your own embeddings" not collapse to a constant.

CONFIG FIELDS (under `train:`):
    ema_base_momentum: 0.998          # cosine momentum start
    ema_final_momentum: 1.0           # cosine momentum end
    vjepa_loss: "smooth_l1"           # or "mse"
    # The original VICReg coefficients (sim_coeff, std_coeff, cov_coeff)
    # must still be in the config because the base Trainer's
    # get_model_components factory expects them. They are unused here.
"""

# ============================================================================
# IMPORTS
# ============================================================================
import argparse
import copy
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from .train import Trainer
from .utils.hydra import compose


# ============================================================================
# EMA TEACHER
# ----------------------------------------------------------------------------
# A momentum-updated copy of the student encoder, used to encode the target
# (future) frames into stable embeddings the student tries to predict.
#
# Properties:
#   - Never receives gradients (requires_grad=False on all parameters).
#   - Always in eval() mode (so e.g. BatchNorm/Dropout behave consistently).
#   - Updated after every training step by:
#         theta_teacher <- tau * theta_teacher + (1 - tau) * theta_student
#   - tau follows a cosine schedule from `base_momentum` to `final_momentum`
#     over the full training run. Lower momentum early = teacher tracks
#     student more closely = stronger early signal. Higher momentum late =
#     teacher locks in stable targets near convergence.
# ============================================================================
class EMATeacher:
    def __init__(self, encoder, base_momentum=0.998, final_momentum=1.0, total_steps=None):
        self.encoder = copy.deepcopy(encoder).eval()
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.base_momentum = base_momentum
        self.final_momentum = final_momentum
        self.total_steps = total_steps
        self.step_count = 0

    def _current_momentum(self):
        """Cosine ramp from base_momentum to final_momentum over total_steps."""
        if self.total_steps is None or self.total_steps == 0:
            return self.base_momentum
        t = min(self.step_count / max(self.total_steps, 1), 1.0)
        return self.final_momentum - (
            self.final_momentum - self.base_momentum
        ) * (math.cos(math.pi * t) + 1) / 2

    @torch.no_grad()
    def update(self, encoder):
        tau = self._current_momentum()
        for tp, sp in zip(self.encoder.parameters(), encoder.parameters()):
            tp.data.mul_(tau).add_(sp.data, alpha=1.0 - tau)
        # Sync running buffers (e.g. BatchNorm running mean/var) directly
        for tb, sb in zip(self.encoder.buffers(), encoder.buffers()):
            tb.data.copy_(sb.data)
        self.step_count += 1

    @torch.no_grad()
    def __call__(self, x):
        return self.encoder(x)


# ============================================================================
# V-JEPA LOSS
# ----------------------------------------------------------------------------
# Per-element smooth L1 (or MSE) between the predictor's output and the EMA
# teacher's target embedding. The loss is averaged over all spatial (and any
# remaining temporal) positions.
#
# Returns a dict matching the contract used by gather_losses_and_report:
#   - 'loss': scalar tensor used for backward()
#   - 'repr_loss': detached copy for logging (mirrors the original VICReg key)
# ============================================================================
def smooth_l1_jepa_loss(pred, target, loss_type="smooth_l1"):
    if loss_type == "smooth_l1":
        loss = F.smooth_l1_loss(pred, target)
    elif loss_type == "mse":
        loss = F.mse_loss(pred, target)
    else:
        raise ValueError(f"Unknown vjepa_loss type: {loss_type}")
    return {"loss": loss, "repr_loss": loss.detach()}


# ============================================================================
# V-JEPA TRAINER
# ----------------------------------------------------------------------------
# Subclass of the base Trainer. We override two methods:
#
#   get_model_components()
#     - Reuse the base factory to build encoder + predictor.
#     - Throw away the VICReg loss it returns; build our own smooth-L1 loss.
#     - Build the EMA teacher (deepcopy of the encoder, on the same device).
#
#   pred_fn()
#     - Encode past frames with the student encoder (gradients flow).
#     - Encode future frames with the EMA teacher (no gradients).
#     - Run the predictor on the student's context embedding.
#     - Compute smooth-L1 between prediction and EMA target.
#     - Update the EMA teacher's weights from the student (training only).
# ============================================================================
class VJepaTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.ema_base = cfg.train.get("ema_base_momentum", 0.998)
        self.ema_final = cfg.train.get("ema_final_momentum", 1.0)
        self.vjepa_loss_type = cfg.train.get("vjepa_loss", "smooth_l1")
        self.ema_teacher = None  # built lazily in get_model_components

    # ------------------------------------------------------------------
    # Build encoder + predictor (via base factory) and EMA teacher.
    # The base Trainer.get_model_components moves components onto the
    # current rank, so the encoder is already on-device when we deepcopy it.
    # ------------------------------------------------------------------
    def get_model_components(self):
        # Base factory returns [encoder, predictor] for objective=jepa, plus
        # a VICReg loss_fn that we ignore.
        model_components, _ = super().get_model_components()
        encoder = model_components[0]

        # Compute total optimizer steps for the cosine momentum schedule.
        # For map-style datasets this is num_epochs * len(loader); for
        # iterable datasets we fall back to a config-supplied step budget.
        if isinstance(self.train_loader.dataset, torch.utils.data.IterableDataset):
            total_steps = self.train_cfg.num_epochs * self.train_cfg.get("steps", 1000)
        else:
            total_steps = self.train_cfg.num_epochs * len(self.train_loader)

        self.ema_teacher = EMATeacher(
            encoder,
            base_momentum=self.ema_base,
            final_momentum=self.ema_final,
            total_steps=total_steps,
        )

        # Smooth-L1 / MSE loss closure with the configured loss type baked in.
        def loss_fn(pred, tgt):
            return smooth_l1_jepa_loss(pred, tgt, loss_type=self.vjepa_loss_type)

        return model_components, loss_fn

    # ------------------------------------------------------------------
    # Per-batch forward + loss.
    # ------------------------------------------------------------------
    def pred_fn(self, batch, model_components, loss_fn):
        encoder, predictor = model_components[0], model_components[1]

        # 1. Student encodes past frames (gradients flow back through here).
        ctx_embed = encoder(batch['context'])

        # 2. EMA teacher encodes future frames. No gradients (no_grad inside).
        tgt_embed = self.ema_teacher(batch['target'])

        # 3. Predictor maps student's context embedding to predicted target.
        pred = predictor(ctx_embed)

        # 4. Smooth-L1 between predictor output and EMA-teacher target.
        loss_dict = loss_fn(pred, tgt_embed)

        # 5. EMA update — only during training, NOT during validation.
        #    The base Trainer.val() puts the encoder into eval() mode, which
        #    is how we detect the validation pass without plumbing extra flags.
        if encoder.training:
            self.ema_teacher.update(encoder)

        # Keep the channel_loss key for logging compatibility with the
        # original JepaTrainer (the main loop expects this key to exist).
        loss_dict['channel_loss'] = torch.tensor(0.0, device=batch['context'].device)
        return pred, loss_dict


# ============================================================================
# ENTRY POINT
# ============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        type=str,
        default=f"{Path(__file__).parent.parent}/configs/train_activematter_vjepa.yaml",
    )
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run
    cfg.model.objective = "jepa"

    print(OmegaConf.to_yaml(cfg, resolve=True))
    trainer = VJepaTrainer(cfg)
    trainer.train()
