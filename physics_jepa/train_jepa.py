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
        return self.value.expand_as(x)

class JepaTrainer(Trainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.channel_masked      = cfg.train.get("channel_masked",      False)
        self.channel_mask_prob   = cfg.train.get("channel_mask_prob",   0.5)
        self.channel_loss_weight = cfg.train.get("channel_loss_weight", 1.0)
        self.num_chans           = cfg.dataset.num_chans

    def get_model_components(self):
        model_components, loss_fn = super().get_model_components()
        if self.channel_masked:
            model_components.append(MaskToken())
        return model_components, loss_fn

    def pred_fn(self, batch, model_components, loss_fn):
        if self.channel_masked:
            encoder, predictor, mask_token = model_components
        else:
            encoder, predictor = model_components

        ctx_embed = encoder(batch['context'])
        tgt_embed = encoder(batch['target'])
        pred = predictor(ctx_embed)
        
        # Compute loss on projected embeddings
        if len(pred.shape) < 5:
            loss_dict = loss_fn(pred.unsqueeze(2), tgt_embed.unsqueeze(2))
        else:
            loss_dict = loss_fn(pred, tgt_embed)

        # Optional channel-masked auxiliary loss
        if self.channel_masked and self.num_chans > 1 and torch.rand(1).item() < self.channel_mask_prob:
            masked_channel = torch.randint(0, self.num_chans, (1,)).item() #Select random channel to mask
            ctx_masked = batch['context'].clone()
            ctx_masked[:, masked_channel] = mask_token(ctx_masked[:, masked_channel])# use a mask token to replace the masked channel
            pred_masked = predictor(encoder(ctx_masked))

            if len(pred_masked.shape) < 5:
                channel_loss_dict = loss_fn(pred_masked.unsqueeze(2), tgt_embed.unsqueeze(2))
            else:
                channel_loss_dict = loss_fn(pred_masked, tgt_embed)

            loss_dict['loss'] = loss_dict['loss'] + self.channel_loss_weight * channel_loss_dict['loss']
            loss_dict['channel_loss'] = channel_loss_dict['loss'].detach()
        else:
            loss_dict['channel_loss'] = torch.tensor(0.0, device=batch['context'].device)

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