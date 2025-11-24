import torch
import torch.nn.functional as F
from torch import nn
from typing import List, Tuple

from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.residual_encoders import ResidualEncoder
from dynamic_network_architectures.building_blocks.unet_decoder import UNetDecoder
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.mono.mono_layer import Mono2DV2
import torch
from torch import nn
import numpy as np
from typing import Union, Type, List, Tuple

from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.residual import StackedResidualBlocks, BottleneckD, BasicBlockD
from dynamic_network_architectures.building_blocks.helper import maybe_convert_scalar_to_list, get_matching_pool_op
from dynamic_network_architectures.building_blocks.simple_conv_blocks import StackedConvBlocks
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.plain_conv_encoder import PlainConvEncoder


class Monov2UNet(ResidualEncoderUNet):
    """
    Has no instance normalization after the mono2d layer. Normalization is static and done inside the Mono2DV2 layer.
    """
    def __init__(self, mono_layer_kwargs: dict = {}, input_channels: int = 1, *args, **kwargs):
        mono_frontend = Mono2DV2(in_channels=input_channels, norm="std", **mono_layer_kwargs)
        input_channels = mono_frontend.out_channels
        super().__init__(input_channels=input_channels, *args, **kwargs)
        self.encoder.mono_frontend = mono_frontend
    
    def forward(self, x):
        x = self.encoder.mono_frontend(x)
        return super().forward(x)


class Monov2UNet01(ResidualEncoderUNet):
    """
    Has instance normalization after the mono2d layer.
    Aim is to see if learning the normalization parameters is better than using a static normalization.
    """
    def __init__(self, mono_layer_kwargs: dict = {}, input_channels: int = 1, *args, **kwargs):
        mono_frontend = Mono2DV2(in_channels=input_channels, norm=None, **mono_layer_kwargs)
        input_channels = mono_frontend.out_channels
        super().__init__(input_channels=input_channels, *args, **kwargs)
        self.encoder.mono_frontend = nn.Sequential(
            mono_frontend, 
            nn.InstanceNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        )

    def forward(self, x):
        x = self.encoder.mono_frontend(x)
        return super().forward(x)


class Monov2UNetEncoder(ResidualEncoderUNet):
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
        
    def forward(self, x):
        return super().forward(x)


class Monov2EncDecUNet(Monov2UNetEncoder):
    def __init__(self, mono_layer_kwargs: dict = {}, input_channels: int = 1, *args, **kwargs):
        super().__init__(mono_layer_kwargs=mono_layer_kwargs, input_channels=input_channels, *args, **kwargs)

        self.decoder = MonoUNetDecoder(
            mono_layer_kwargs,
            self.encoder,
            kwargs.get("num_classes"),
            kwargs.get("n_conv_per_stage_decoder"),
            kwargs.get("deep_supervision")
        )


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

        # Add a Mono2DV2 layer at the encoder visual front-end
        mono_frontend = Mono2DV2(in_channels=input_channels, norm=None, **mono_layer_kwargs)
        input_channels = mono_frontend.out_channels
        self.mono_frontend = nn.Sequential(
            mono_frontend, 
            nn.InstanceNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        )

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
            mono_layer = Mono2DV2(in_channels=input_channels, norm=None, **mono_layer_kwargs)
            input_channels = mono_layer.out_channels# + input_channels
            mono_layers.append(
                nn.Sequential(
                    mono_layer,
                    nn.InstanceNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
                )
            )

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
        x = self.mono_frontend(x)
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


class MonoUNetDecoder(nn.Module):
    def __init__(self,
                 mono_layer_kwargs: dict,
                 encoder: Union[PlainConvEncoder, ResidualEncoder],
                 num_classes: int,
                 n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
                 deep_supervision,
                 nonlin_first: bool = False,
                 norm_op: Union[None, Type[nn.Module]] = None,
                 norm_op_kwargs: dict = None,
                 dropout_op: Union[None, Type[_DropoutNd]] = None,
                 dropout_op_kwargs: dict = None,
                 nonlin: Union[None, Type[torch.nn.Module]] = None,
                 nonlin_kwargs: dict = None,
                 conv_bias: bool = None
                 ):
        """
        This class needs the skips of the encoder as input in its forward.

        the encoder goes all the way to the bottleneck, so that's where the decoder picks up. stages in the decoder
        are sorted by order of computation, so the first stage has the lowest resolution and takes the bottleneck
        features and the lowest skip as inputs
        the decoder has two (three) parts in each stage:
        1) conv transpose to upsample the feature maps of the stage below it (or the bottleneck in case of the first stage)
        2) n_conv_per_stage conv blocks to let the two inputs get to know each other and merge
        3) (optional if deep_supervision=True) a segmentation output Todo: enable upsample logits?
        :param encoder:
        :param num_classes:
        :param n_conv_per_stage:
        :param deep_supervision:
        """
        super().__init__()
        self.deep_supervision = deep_supervision
        self.encoder = encoder
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, "n_conv_per_stage must have as many entries as we have " \
                                                          "resolution stages - 1 (n_stages in encoder - 1), " \
                                                          "here: %d" % n_stages_encoder

        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = encoder.dropout_op_kwargs if dropout_op_kwargs is None else dropout_op_kwargs
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs


        # we start with the bottleneck and work out way up
        stages = []
        transpconvs = []
        seg_layers = []
        mono_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(transpconv_op(
                input_features_below, input_features_skip, stride_for_transpconv, stride_for_transpconv,
                bias=conv_bias
            ))
            # input features to conv is 2x input_features_skip (concat input_features_skip with transpconv output)
            mono_layer = Mono2DV2(in_channels=2 * input_features_skip, norm=None, **mono_layer_kwargs)
            input_channels = mono_layer.out_channels
            mono_layers.append(
                nn.Sequential(
                    mono_layer,
                    nn.InstanceNorm2d(input_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
                )
            )
            stages.append(StackedConvBlocks(
                n_conv_per_stage[s-1], encoder.conv_op, input_channels, input_features_skip,
                encoder.kernel_sizes[-(s + 1)], 1,
                conv_bias,
                norm_op,
                norm_op_kwargs,
                dropout_op,
                dropout_op_kwargs,
                nonlin,
                nonlin_kwargs,
                nonlin_first
            ))

            # we always build the deep supervision outputs so that we can always load parameters. If we don't do this
            # then a model trained with deep_supervision=True could not easily be loaded at inference time where
            # deep supervision is not needed. It's just a convenience thing
            seg_layers.append(encoder.conv_op(input_features_skip, num_classes, 1, 1, 0, bias=True))

        self.stages = nn.ModuleList(stages)
        self.mono_layers = nn.ModuleList(mono_layers)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        """
        we expect to get the skips in the order they were computed, so the bottleneck should be the last entry
        :param skips:
        :return:
        """
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s+2)]), 1)
            x = self.mono_layers[s](x)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == (len(self.stages) - 1):
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x

        # invert seg outputs so that the largest segmentation prediction is returned first
        seg_outputs = seg_outputs[::-1]

        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r

    def compute_conv_feature_map_size(self, input_size):
        """
        IMPORTANT: input_size is the input_size of the encoder!
        :param input_size:
        :return:
        """
        # first we need to compute the skip sizes. Skip bottleneck because all output feature maps of our ops will at
        # least have the size of the skip above that (therefore -1)
        skip_sizes = []
        for s in range(len(self.encoder.strides) - 1):
            skip_sizes.append([i // j for i, j in zip(input_size, self.encoder.strides[s])])
            input_size = skip_sizes[-1]
        # print(skip_sizes)

        assert len(skip_sizes) == len(self.stages)

        # our ops are the other way around, so let's match things up
        output = np.int64(0)
        for s in range(len(self.stages)):
            # print(skip_sizes[-(s+1)], self.encoder.output_channels[-(s+2)])
            # conv blocks
            output += self.stages[s].compute_conv_feature_map_size(skip_sizes[-(s+1)])
            # trans conv
            output += np.prod([self.encoder.output_channels[-(s+2)], *skip_sizes[-(s+1)]], dtype=np.int64)
            # segmentation
            if self.deep_supervision or (s == (len(self.stages) - 1)):
                output += np.prod([self.num_classes, *skip_sizes[-(s+1)]], dtype=np.int64)
        return output


