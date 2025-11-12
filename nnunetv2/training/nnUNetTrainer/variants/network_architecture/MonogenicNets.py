import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.mono.mono_layer import Mono2D

class MonoUNet(ResidualEncoderUNet):
    """
    ResidualEncoderUNet with Monogenic layer at the front-end (before encoder).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono2d = Mono2D()
        print("\n\nUsing MonoUNet with mono2d layer\n\n")
    
    def forward(self, x):
        x = self.mono2d(x)
        return super().forward(x)


class MonoEnhancedUNet(ResidualEncoderUNet):
    """
    ResidualEncoderUNet with Monogenic layer at the front-end (before encoder).
    The layer extracts the local phase features and combines them with the original image
    to enhance structural information (e.g. edges, blobs, lines, etc.) without losing 
    intensity information.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono2d = Mono2D()
        print("\n\nUsing MonoEnhancedUNet with mono2d layer\n\n")
    
    def forward(self, x):
        x = x + self.mono2d(x)
        return super().forward(x)


class MonoGatedUNet(ResidualEncoderUNet):
    """
    ResidualEncoderUNet with Monogenic layer at the front-end (before encoder).
    The layer extracts the local phase features and uses them to gate the original image
    to emphasize structural information (e.g. edges, blobs, lines, etc.).
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono2d = Mono2D()
        print("\n\nUsing MonoGatedUNet with mono2d layer\n\n")
    
    def forward(self, x):
        x = x * self.mono2d(x)
        return super().forward(x)


class MonoSkipFusionUNet(ResidualEncoderUNet):
    """
    ResidualEncoderUNet with pyramid local phase features fused with the original skip 
    connections before concatenation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono2d = Mono2D()
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
        print("\n\nUsing MonoSkipFusionUNet with mono2d layer\n\n")

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
        skip_shapes = [tuple(feat.shape[2:]) for feat in skips[:-1]]
        mono_inputs = self._prepare_mono_inputs(x, skip_shapes)
        enhanced_skips: List[torch.Tensor] = list(skips)

        for idx, (mono_input, fusion_layer) in enumerate(zip(mono_inputs, self.skip_fusion)):
            mono_features = self.mono2d(mono_input)
            enhanced_skips[idx] = fusion_layer(torch.cat((enhanced_skips[idx], mono_features), dim=1))

        return self.decoder(enhanced_skips)


class MonoGatedSkipUNet(ResidualEncoderUNet):
    """
    ResidualEncoderUNet with local phase features gated skip connections.
    The layer downsamples the input image to the same number of stages of the encoder, 
    extracts the local phase features and passess them to the corresponding stage of the
    decoder where they are used to gate the original skip connections before concatenation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mono2d = Mono2D()
        print("\n\nUsing MonoGatedSkipUNet with mono2d layer\n\n")

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

    def _gate_skip(self, skip: torch.Tensor, mono_input: torch.Tensor) -> torch.Tensor:
        gate = self.mono2d(mono_input)
        return skip * torch.sigmoid(gate)

    def forward(self, x):
        skips = self.encoder(x)
        if len(skips) <= 1:
            return self.decoder(skips)

        skip_shapes = [tuple(feat.shape[2:]) for feat in skips[:-1]]
        mono_inputs = self._build_mono_pyramid(x, skip_shapes)
        gated_skips: List[torch.Tensor] = list(skips)

        for idx, mono_input in enumerate(mono_inputs):
            gated_skips[idx] = self._gate_skip(gated_skips[idx], mono_input)

        return self.decoder(gated_skips)
