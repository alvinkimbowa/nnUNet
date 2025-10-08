from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from typing import Union, Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo import OptimizedModule
import sys
import types


try:
    import ml_collections  # type: ignore
except Exception:
    class _ConfigDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def __getattr__(self, item):
            if item in self:
                return self[item]
            v = _ConfigDict()
            self[item] = v
            return v
        def __setattr__(self, key, value):
            self[key] = value
        def get(self, key, default=None):
            return dict.get(self, key, default)

    ml_collections = types.SimpleNamespace(ConfigDict=_ConfigDict)
    sys.modules['ml_collections'] = ml_collections


class TransUNetTrainer(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   patch_size: List[int],
                                   enable_deep_supervision: bool = True) -> nn.Module:
        """
        Build a lightweight TransUNet-like architecture that is compatible with nnUNetTrainer.

        We keep the implementation local to this file so no external changes are necessary. The model
        supports 2D and 3D based on the length of `patch_size`. When `enable_deep_supervision` is True
        the forward pass returns a list where the 0th element is the highest resolution output (this is
        what nnUNetTrainer expects during validation).
        """

        dim = len(patch_size)
        return TransUNet(
            input_channels=num_input_channels,
            num_classes=num_output_channels,
            deep_supervision=enable_deep_supervision,
            dims=dim,
            patch_size=patch_size,
            base_num_features=32,
            num_levels=4,
        )

    def set_deep_supervision_enabled(self, enabled: bool):
        """
        Toggle deep supervision flag on the constructed network. This mirrors other trainer variants
        so nnUNetTrainer can enable/disable deep supervision at runtime.
        """
        if self.is_ddp:
            mod = self.network.module
        else:
            mod = self.network
        # if compiled via torch.compile, the OptimizedModule wrapper stores original module
        if isinstance(mod, OptimizedModule):
            mod = mod._orig_mod

        # our TransUNet stores deep_supervision on the module itself
        if hasattr(mod, 'deep_supervision'):
            mod.deep_supervision = enabled
        # fallback: try to set decoder.deep_supervision if that pattern exists
        elif hasattr(mod, 'decoder'):
            setattr(mod.decoder, 'deep_supervision', enabled)


# Configs
class ConfigDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def __getattr__(self, item):
        if item in self:
            return self[item]
        v = ConfigDict()
        self[item] = v
        return v
    def __setattr__(self, key, value):
        self[key] = value
    def get(self, key, default=None):
        return dict.get(self, key, default)

def get_b16_config():
    cfg = ConfigDict()
    cfg.n_classes = 2
    cfg.n_skip = 3
    cfg.skip_channels = [512, 256, 64, 16]
    cfg.patch_size = 16
    cfg.img_size = 224
    cfg.in_channels = 3
    cfg.embed_dim = 768
    cfg.depth = 12
    cfg.num_heads = 12
    cfg.mlp_ratio = 4
    cfg.qkv_bias = True
    cfg.drop_rate = 0.1
    cfg.attn_drop_rate = 0.0
    cfg.drop_path_rate = 0.1
    cfg.hybrid = False
    return cfg


# VisionTransformer (minimal, only for TransUNet)
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=config.patch_size, in_chans=config.in_channels, embed_dim=config.embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, config.embed_dim))
        self.pos_drop = nn.Dropout(p=config.drop_rate)
        encoder_layer = nn.TransformerEncoderLayer(d_model=config.embed_dim, nhead=config.num_heads, dim_feedforward=config.embed_dim * config.mlp_ratio, dropout=config.drop_rate, activation='gelu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.depth)
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = self.head(x[:, 0])
        return x




class TransUNet(nn.Module):
    """Wraps the VisionTransformer to provide compatibility with nnUNetTrainer.

    - Accepts `input_channels`, `num_classes`, `deep_supervision` and `patch_size` dims
    - This TransUNet is 2D only (the original repo is 2D). If dims != 2 we raise.
    """
    def __init__(self, input_channels: int, num_classes: int, deep_supervision: bool = True, dims: int = 2, patch_size: Optional[List[int]] = None, base_num_features: int = 32, num_levels: int = 4):
        super().__init__()
        if dims != 2:
            raise NotImplementedError("TransUNet implementation is 2D only")
        # build config and adapt
        cfg = get_b16_config()
        cfg.n_classes = num_classes
        try:
            n_skip = int(cfg.n_skip)
        except Exception:
            n_skip = 0
        cfg.n_skip = n_skip
        if not hasattr(cfg, 'skip_channels') or not isinstance(cfg.skip_channels, (list, tuple)):
            cfg.skip_channels = [0, 0, 0, 0]
        img_size = patch_size[0] if (patch_size is not None and len(patch_size) > 0) else 224
        self.vit = VisionTransformer(cfg, img_size=img_size, num_classes=num_classes)
        self.deep_supervision = deep_supervision

    def forward(self, x: torch.Tensor):
        logits = self.vit(x)
        if not self.deep_supervision:
            return logits
        # For compatibility, return a list where index 0 is highest resolution output
        return [logits]


def _pad_to(x: torch.Tensor, spatial_size: tuple):
    """Pad tensor x to have spatial dimensions spatial_size. Only pads on the right/bottom if odd differences."""
    current = x.shape[2:]
    pads = []
    for cur, tgt in zip(reversed(current), reversed(spatial_size)):
        diff = tgt - cur
        if diff < 0:
            # crop
            return x[..., :tgt]
        pads.extend([0, diff])
    # pads needs to be in reverse order for F.pad: (D_right, D_left, H_right, H_left, W_right, W_left)
    pads = tuple(pads)
    return F.pad(x, pads)


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, Conv, BatchNorm):
        super().__init__()
        self.conv1 = Conv(in_ch, out_ch, kernel_size=3, padding=1)
        self.bn1 = BatchNorm(out_ch)
        self.conv2 = Conv(out_ch, out_ch, kernel_size=3, padding=1)
        self.bn2 = BatchNorm(out_ch)

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x, inplace=True)
        return x

    def __repr__(self):
        return f"_ConvBlock({self.conv1.in_channels}->{self.conv2.out_channels})"