import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.mono.mono_layer import Mono2D

class MonoBaseNet(ResidualEncoderUNet):
    def __init__(self, mono_layer_kwargs: dict = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono2d = Mono2D(**mono_layer_kwargs, norm="std")


class MonoUNet(MonoBaseNet):
    """
    ResidualEncoderUNet with Monogenic layer at the front-end (before encoder).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("\n\nUsing MonoUNet with mono2d layer\n\n")
    
    def forward(self, x):
        x = self.mono2d(x)
        return super().forward(x)


class MonoEnhancedUNet(MonoBaseNet):
    """
    ResidualEncoderUNet with Monogenic layer at the front-end (before encoder).
    The layer extracts the local phase features and combines them with the original image
    to enhance structural information (e.g. edges, blobs, lines, etc.) without losing 
    intensity information.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("\n\nUsing MonoEnhancedUNet with mono2d layer\n\n")
    
    def forward(self, x):
        x = x + self.mono2d(x)
        return super().forward(x)


class MonoGatedUNet(MonoBaseNet):
    """
    ResidualEncoderUNet with Monogenic layer at the front-end (before encoder).
    The layer extracts the local phase features and uses them to gate the original image
    to emphasize structural information (e.g. edges, blobs, lines, etc.).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Learnable affine for gate and residual strength
        self.gate_weight = nn.Parameter(torch.ones(1))
        self.gate_bias = nn.Parameter(torch.zeros(1))
        self.alpha = nn.Parameter(torch.randn(1))
        print("\n\nUsing MonoGatedUNet with mono2d layer\n\n")
    
    def forward(self, x):
        mono_feat = self.mono2d(x)
        gate = torch.sigmoid(self.gate_weight * mono_feat + self.gate_bias)
        x = x * (1 + self.alpha * gate)     # Use a residual connection to stabilize training
        return super().forward(x)


class MonoSkipFusionUNet(MonoBaseNet):
    """
    ResidualEncoderUNet with pyramid local phase features fused with the original skip 
    connections before concatenation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono2d_layers = nn.ModuleList([Mono2D(nscale=6, norm="std") for _ in range(len(self.encoder.output_channels) - 1)])
        self.mono_feature_channels = (
            int(self.mono2d.return_input)
            + int(self.mono2d.return_phase)
            + int(self.mono2d.return_ori)
            + int(self.mono2d.return_phase_sym)
            + int(self.mono2d.return_phase_asym)
        )
        if self.mono_feature_channels == 0:
            raise ValueError("Mono2D must return at least one feature map.")
        self.skip_fusion = nn.ModuleList(
            nn.Conv2d(ch + self.mono_feature_channels, ch, kernel_size=1, bias=False)
            for ch in self.encoder.output_channels[:-1]
        )
        print("\n\nUsing MonoSkipFusionUNet with separate mono2d layers for each stage\n\n")

    def _prepare_mono_inputs(self, x: torch.Tensor, target_shapes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        mono_inputs: List[torch.Tensor] = []
        current = x
        current_shape = tuple(x.shape[2:])
        for shape in target_shapes:
            if tuple(shape) != current_shape:
                current = F.interpolate(
                    current,
                    size=shape,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                )
                current_shape = tuple(current.shape[2:])
            mono_inputs.append(current)
        return mono_inputs

    def forward(self, x):
        if not len(self.skip_fusion):
            return super().forward(x)

        skips = self.encoder(x)
        target_shapes = [tuple(s.shape[2:]) for s in skips[:-1]]
        mono_inputs = self._prepare_mono_inputs(x, target_shapes)
        enhanced_skips: List[torch.Tensor] = []
        for idx, (skip, mono) in enumerate(zip(skips[:-1], mono_inputs)):
            mono_features = self.mono2d_layers[idx](mono)
            enhanced_skip = torch.cat((skip, mono_features), dim=1)
            enhanced_skip = self.skip_fusion[idx](enhanced_skip)
            enhanced_skips.append(enhanced_skip)
        enhanced_skips.append(skips[-1])

        return self.decoder(enhanced_skips)


class MonoGatedSkipUNet(MonoBaseNet):
    """
    ResidualEncoderUNet with local phase features gated skip connections.
    The layer downsamples the input image to the same number of stages of the encoder, 
    extracts the local phase features and passess them to the corresponding stage of the
    decoder where they are used to gate the original skip connections before concatenation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono2d_layers = nn.ModuleList([Mono2D(nscale=6, norm="std") for _ in range(len(self.encoder.output_channels) - 1)])
        print("\n\nUsing MonoGatedSkipUNet with separate mono2d layers for each stage\n\n")

    def _build_mono_pyramid(self, x: torch.Tensor, target_shapes: List[Tuple[int, int]]) -> List[torch.Tensor]:
        mono_inputs: List[torch.Tensor] = []
        current = x
        current_shape = tuple(x.shape[2:])
        for shape in target_shapes:
            if current_shape != tuple(shape):
                current = F.interpolate(
                    current,
                    size=shape,
                    mode="bilinear",
                    align_corners=False,
                    recompute_scale_factor=False,
                )
                current_shape = tuple(current.shape[2:])
            mono_inputs.append(current)
        return mono_inputs

    def _gate_skip(self, skip: torch.Tensor, mono_input: torch.Tensor, layer_idx: int) -> torch.Tensor:
        gate = self.mono2d_layers[layer_idx](mono_input)
        return skip * torch.sigmoid(gate)

    def forward(self, x):
        skips = self.encoder(x)
        if len(skips) <= 1:
            return self.decoder(skips)

        skip_shapes = [tuple(feat.shape[2:]) for feat in skips[:-1]]
        mono_inputs = self._build_mono_pyramid(x, skip_shapes)
        gated_skips: List[torch.Tensor] = list(skips)

        for idx, mono_input in enumerate(mono_inputs):
            gated_skips[idx] = self._gate_skip(gated_skips[idx], mono_input, idx)

        return self.decoder(gated_skips)
