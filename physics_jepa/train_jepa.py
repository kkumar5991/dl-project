import argparse
from pathlib import Path
from omegaconf import OmegaConf
import torch
import torch.nn as nn

from .train import Trainer
from .utils.hydra import compose


class MaskToken(nn.Module):
    """Learnable scalar broadcast across (T, H, W) of a masked channel."""
    def __init__(self):
        super().__init__()
        self.value = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """Fill x: (B, T, H, W) with the learned mask value."""
        return self.value.view(1, 1, 1, 1).expand_as(x)

class JepaTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.channel_masked   = cfg.train.get("channel_masked",   False)
        self.field_masked     = cfg.train.get("field_masked",     False)
        self.inverse_target   = cfg.train.get("inverse_target",   False)
        self.num_chans        = cfg.dataset.num_chans
        self.num_fields       = 4  # HARDCODED: for active matter.
        self.fields           = [[0], [1,2], [3,4,5,6], [7,8,9,10]]  # concentration, velocity, orientation, strain

    def get_model_components(self):
        model_components, loss_fn = super().get_model_components()
        if self.channel_masked or self.field_masked:
            model_components.append(MaskToken())
        return model_components, loss_fn

    def pred_fn(self, batch, model_components, loss_fn):
        if self.channel_masked or self.field_masked:
            encoder, predictor, mask_token = model_components
        else:
            encoder, predictor = model_components

        chosen_field = None

        # masking ctx input
        if self.channel_masked:
            masked_channel = torch.randint(0, self.num_chans, (1,)).item()
            ctx_input = batch['context'].clone()
            ctx_input[:, masked_channel] = mask_token(ctx_input[:, masked_channel])
        elif self.field_masked:
            chosen_field = torch.randint(0, self.num_fields, (1,)).item()
            ctx_input = batch['context'].clone()
            for chan in self.fields[chosen_field]:
                ctx_input[:, chan] = mask_token(ctx_input[:, chan])
        else:
            ctx_input = batch['context']

        # masking target input (inverse of masked field in ctx)
        if self.inverse_target:
            if chosen_field is None:
                target_input = batch['target']
                print("Warning: inverse_target is True but no field masked in encoder. Try setting field_masked=True.")
            else:
                target_input = batch['target'].clone()
                for chan in range(self.num_chans):
                    if chan not in self.fields[chosen_field]:
                        target_input[:, chan] = mask_token(target_input[:, chan])
        else:
            target_input = batch['target']

        ctx_embed = encoder(ctx_input)
        tgt_embed = encoder(target_input)
        pred = predictor(ctx_embed)

        if len(pred.shape) < 5:
            loss_dict = loss_fn(pred.unsqueeze(2), tgt_embed.unsqueeze(2))
        else:
            loss_dict = loss_fn(pred, tgt_embed)

        return pred, loss_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default=f"{Path(__file__).parent.parent}/configs/train_grayscott.yml")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--predictor_path", type=str, default=None)
    parser.add_argument("--channel_masked", action="store_true",
                        help="Enable channel-masked auxiliary loss")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run
    # cfg.train.encoder_path = args.encoder_path
    # cfg.train.predictor_path = args.predictor_path
    
    cfg.model.objective = "jepa"
    cfg.train.channel_masked = args.channel_masked

    print(OmegaConf.to_yaml(cfg, resolve=True))

    trainer = JepaTrainer(cfg)
    trainer.train()