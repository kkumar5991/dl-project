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

class FieldConditionedPredictor(nn.Module):
    def __init__(self, predictor, num_fields=4):
        super().__init__()
        self.predictor = predictor
        # get dim from predictor's first conv weight
        dim = predictor.conv[0].in_channels
        self.field_embed = nn.Embedding(num_fields+1, dim) # +1 for "no field masked" case


    def forward(self, x, field_id=None):
        if field_id is not None:
            if isinstance(field_id, int):
                fid = torch.tensor([field_id], device=x.device)
            else:
                fid = field_id.to(x.device)
            bias = self.field_embed(fid).view(1, -1, 1, 1)
            x = x + bias
        return self.predictor(x)
class JepaTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.channel_masked   = cfg.train.get("channel_masked",   False)
        self.field_masked     = cfg.train.get("field_masked",     False)
        self.inverse_target   = cfg.train.get("inverse_target",   False)
        self.learnable_mask   = cfg.train.get("learnable_mask",   False)
        self.num_chans        = cfg.dataset.num_chans
        self.num_fields       = 4  # HARDCODED: for active matter.
        self.fields           = [[0], [1,2], [3,4,5,6], [7,8,9,10]]  # concentration, velocity, orientation, strain
        self.to_apply_mask = self.channel_masked or self.field_masked
        self.apply_mask       = self.to_apply_mask
        self.step_counter = 0
        self.mask_period = cfg.train.get("mask_period", 1)  # mask once every N steps
        self.use_field_z_predictor = cfg.train.get("field_z",False)
    def get_model_components(self):
        model_components, loss_fn = super().get_model_components()
        if self.field_masked and self.use_field_z_predictor:
            encoder = model_components[0]
            predictor = FieldConditionedPredictor(model_components[1], num_fields=self.num_fields).to(self.rank)
            model_components = [encoder, predictor]
        if (self.channel_masked or self.field_masked) and self.learnable_mask:
            model_components.append(MaskToken().to(self.rank))
        return model_components, loss_fn

    def pred_fn(self, batch, model_components, loss_fn):
        if (self.channel_masked or self.field_masked) and self.learnable_mask:
            encoder, predictor, mask_token = model_components
        else:
            encoder, predictor = model_components

        chosen_field = None
        # masking ctx input
        if self.channel_masked and self.apply_mask:
            masked_channel = torch.randint(0, self.num_chans, (1,)).item()
            ctx_input = batch['context'].clone()
            ctx_input[:, masked_channel] = mask_token(ctx_input[:, masked_channel]) if self.learnable_mask else 0.0
        elif self.field_masked and self.apply_mask:
            chosen_field = torch.randint(0, self.num_fields, (1,)).item()
            ctx_input = batch['context'].clone()
            for chan in self.fields[chosen_field]:
                ctx_input[:, chan] = mask_token(ctx_input[:, chan]) if self.learnable_mask else 0.0
        else:
            ctx_input = batch['context']

        # masking target input (inverse of masked field in ctx)
        if self.inverse_target and self.apply_mask:
            if chosen_field is None:
                target_input = batch['target']
                print("Warning: inverse_target is True but no field masked in encoder. Try setting field_masked=True.")
            else:
                target_input = batch['target'].clone()
                for chan in range(self.num_chans):
                    if chan not in self.fields[chosen_field]:
                        target_input[:, chan] = mask_token(target_input[:, chan]) if self.learnable_mask else 0.0
        else:
            target_input = batch['target']

        ctx_embed = encoder(ctx_input)
        tgt_embed = encoder(target_input)
        if self.field_masked and self.use_field_z_predictor:
            if not self.apply_mask:
                chosen_field = self.num_fields  # index 4: "no field masked" token
            pred = predictor(ctx_embed, field_id=chosen_field)
        else:
            pred = predictor(ctx_embed)

        if len(pred.shape) < 5:
            loss_dict = loss_fn(pred.unsqueeze(2), tgt_embed.unsqueeze(2))
        else:
            loss_dict = loss_fn(pred, tgt_embed)

        if self.to_apply_mask:
            self.step_counter += 1
            self.apply_mask = (self.step_counter % self.mask_period == 0)
        return pred, loss_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, default=f"{Path(__file__).parent.parent}/configs/train_grayscott.yml")
    parser.add_argument("overrides", nargs="*")
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--predictor_path", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    cfg = compose(args.config, args.overrides)
    OmegaConf.set_struct(cfg, False)
    cfg.dry_run = args.dry_run
    # cfg.train.encoder_path = args.encoder_path
    # cfg.train.predictor_path = args.predictor_path
    
    cfg.model.objective = "jepa"

    print(OmegaConf.to_yaml(cfg, resolve=True))

    trainer = JepaTrainer(cfg)
    trainer.train()