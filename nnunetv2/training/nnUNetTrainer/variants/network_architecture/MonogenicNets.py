import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple, Union, Type
import numpy as np

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BottleneckD, BasicBlockD
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
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


class MonoEncUNet(ResidualEncoderUNet):
    def __init__(self, mono_layer_kwargs: dict = {}, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Extract parameters that shouldn't be passed to encoder
        num_classes = kwargs.pop("num_classes", None)
        n_conv_per_stage_decoder = kwargs.pop("n_conv_per_stage_decoder", None)
        deep_supervision = kwargs.pop("deep_supervision", None)

        # Extract input_channels from kwargs and pass mono_layer_kwargs as keyword argument
        input_channels = kwargs.pop("input_channels")
        self.encoder = MonoResidualEncoder(
            input_channels=input_channels,
            mono_layer_kwargs=mono_layer_kwargs,
            return_skips=True,
            disable_default_stem=False,
            **kwargs
        )

        self.decoder = UNetDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision)

        self.mono2d = Mono2D(in_channels=input_channels, norm="std", **mono_layer_kwargs)
        print("\n\nUsing MonoEncUNet with mono2d layer\n\n")
        
    def forward(self, x):
        x = self.mono2d(x)
        return super().forward(x)


class MonoResidualEncoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 n_stages: int,
                 features_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_op: Type[_ConvNd],
                 kernel_sizes: Union[int, List[int], Tuple[int, ...]],
                 strides: Union[int, List[int], Tuple[int, ...], Tuple[Tuple[int, ...], ...]],
                 n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
                 conv_bias: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 block: Union[Type[BasicBlockD], Type[BottleneckD]] = BasicBlockD,
                 bottleneck_channels: Union[int, List[int], Tuple[int, ...]] = None,
                 return_skips: bool = False,
                 disable_default_stem: bool = False,
                 stem_channels: int = None,
                 pool_type: str = 'conv',
                 stochastic_depth_p: float = 0.0,
                 squeeze_excitation: bool = False,
                 squeeze_excitation_reduction_ratio: float = 1. / 16,
                 mono_layer_kwargs: dict = {}
                 ):
        """

        :param input_channels:
        :param n_stages:
        :param features_per_stage: Note: If the block is BottleneckD, then this number is supposed to be the number of
        features AFTER the expansion (which is not coded implicitly in this repository)! See todo!
        :param conv_op:
        :param kernel_sizes:
        :param strides:
        :param n_blocks_per_stage:
        :param conv_bias:
        :param norm_op:
        :param norm_op_kwargs:
        :param dropout_op:
        :param dropout_op_kwargs:
        :param nonlin:
        :param nonlin_kwargs:
        :param block:
        :param bottleneck_channels: only needed if block is BottleneckD
        :param return_skips: set this to True if used as encoder in a U-Net like network
        :param disable_default_stem: If True then no stem will be created. You need to build your own and ensure it is executed first, see todo.
        The stem in this implementation does not so stride/pooling so building your own stem is a necessity if you need this.
        :param stem_channels: if None, features_per_stage[0] will be used for the default stem. Not recommended for BottleneckD
        :param pool_type: if conv, strided conv will be used. avg = average pooling, max = max pooling
        """
        super().__init__()
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if bottleneck_channels is None or isinstance(bottleneck_channels, int):
            bottleneck_channels = [bottleneck_channels] * n_stages
        assert len(
            bottleneck_channels) == n_stages, "bottleneck_channels must be None or have as many entries as we have resolution stages (n_stages)"
        assert len(
            kernel_sizes) == n_stages, "kernel_sizes must have as many entries as we have resolution stages (n_stages)"
        assert len(
            n_blocks_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(
            features_per_stage) == n_stages, "features_per_stage must have as many entries as we have resolution stages (n_stages)"
        assert len(strides) == n_stages, "strides must have as many entries as we have resolution stages (n_stages). " \
                                         "Important: first entry is recommended to be 1, else we run strided conv drectly on the input"

        pool_op = get_matching_pool_op(conv_op, pool_type=pool_type) if pool_type != 'conv' else None

        # build a stem, Todo maybe we need more flexibility for this in the future. For now, if you need a custom
        #  stem you can just disable the stem and build your own.
        #  THE STEM DOES NOT DO STRIDE/POOLING IN THIS IMPLEMENTATION
        if not disable_default_stem:
            if stem_channels is None:
                stem_channels = features_per_stage[0]
            self.stem = StackedConvBlocks(1, conv_op, input_channels, stem_channels, kernel_sizes[0], 1, conv_bias,
                                          norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs)
            input_channels = stem_channels
        else:
            self.stem = None
        
        # now build the network
        stages = []
        mono_layers = []
        for s in range(n_stages):
            stride_for_conv = strides[s] if pool_op is None else 1

            # Add a Mono2DV2 layer at each stage of the encoder to compensate for lost capacity
            mono_layers.append(Mono2D(in_channels=input_channels, norm="std", **mono_layer_kwargs))

            stage = StackedResidualBlocks(
                n_blocks_per_stage[s], conv_op, input_channels, features_per_stage[s], kernel_sizes[s], stride_for_conv,
                conv_bias, norm_op, norm_op_kwargs, dropout_op, dropout_op_kwargs, nonlin, nonlin_kwargs,
                block=block, bottleneck_channels=bottleneck_channels[s], stochastic_depth_p=stochastic_depth_p,
                squeeze_excitation=squeeze_excitation,
                squeeze_excitation_reduction_ratio=squeeze_excitation_reduction_ratio
            )

            if pool_op is not None:
                stage = nn.Sequential(pool_op(strides[s]), stage)

            stages.append(stage)
            input_channels = features_per_stage[s]

        self.stages = nn.Sequential(*stages)
        self.mono_layers = nn.ModuleList(mono_layers)
        self.output_channels = features_per_stage
        self.strides = [maybe_convert_scalar_to_list(conv_op, i) for i in strides]
        self.return_skips = return_skips

        # we store some things that a potential decoder needs
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes
        
        print("\n\nUsing MonoResidualEncoder with mono2d layer\n\n")


    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for mono_layer, s in zip(self.mono_layers, self.stages):
            x = mono_layer(x)
            x = s(x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]

    def compute_conv_feature_map_size(self, input_size):
        if self.stem is not None:
            output = self.stem.compute_conv_feature_map_size(input_size)
        else:
            output = np.int64(0)

        for s in range(len(self.stages)):
            output += self.stages[s].compute_conv_feature_map_size(input_size)
            input_size = [i // j for i, j in zip(input_size, self.strides[s])]

        return output


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
