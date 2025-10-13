from MonaiTrainers import nnUNetTrainerNoDeepSupervision
from typing import Union, Tuple, List
from torch import nn
import monai.networks.nets as nets
from monai.networks.blocks.segresnet_block import ResBlock, get_conv_layer, get_upsample_layer


class SegResNetTiny1Trainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return nets.SegResNet(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            init_filters=1,
            norm=("GROUP", {"num_groups": 1}),
        )


class SegResNetTiny2Trainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return nets.SegResNet(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            init_filters=2,
            norm=("GROUP", {"num_groups": 2}),
        )


class SegResNetTiny4Trainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return nets.SegResNet(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            init_filters=4,
            norm=("GROUP", {"num_groups": 4})
        )


class SegResNetXTiny1Trainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return SegResNetXTiny1(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            init_filters=1,
            norm=("GROUP", {"num_groups": 1}),
        )


class SegResNetXTiny2Trainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return SegResNetXTiny2(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            init_filters=1,
            norm=("GROUP", {"num_groups": 1}),
        )


class SegResNetXTiny4Trainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return SegResNetXTiny4(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            init_filters=1,
            norm=("GROUP", {"num_groups": 1}),
        )


class SegResNetXTiny1(nets.SegResNet):
    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = min(filters * 2**i, 1)
            print(f"layer_in_channels: {layer_in_channels}")
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers
    
    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = min(filters * 2 ** (n_up - i), 1)
            print(f"sample_in_channels: {sample_in_channels}")
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples


class SegResNetXTiny2(nets.SegResNet):
    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = min(filters * 2**i, 2)
            print(f"layer_in_channels: {layer_in_channels}")
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers
    
    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = min(filters * 2 ** (n_up - i), 2)
            print(f"sample_in_channels: {sample_in_channels}")
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples


class SegResNetXTiny4(nets.SegResNet):
    def _make_down_layers(self):
        down_layers = nn.ModuleList()
        blocks_down, spatial_dims, filters, norm = (self.blocks_down, self.spatial_dims, self.init_filters, self.norm)
        for i, item in enumerate(blocks_down):
            layer_in_channels = min(filters * 2**i, 4)
            print(f"layer_in_channels: {layer_in_channels}")
            pre_conv = (
                get_conv_layer(spatial_dims, layer_in_channels // 2, layer_in_channels, stride=2)
                if i > 0
                else nn.Identity()
            )
            down_layer = nn.Sequential(
                pre_conv, *[ResBlock(spatial_dims, layer_in_channels, norm=norm, act=self.act) for _ in range(item)]
            )
            down_layers.append(down_layer)
        return down_layers
    
    def _make_up_layers(self):
        up_layers, up_samples = nn.ModuleList(), nn.ModuleList()
        upsample_mode, blocks_up, spatial_dims, filters, norm = (
            self.upsample_mode,
            self.blocks_up,
            self.spatial_dims,
            self.init_filters,
            self.norm,
        )
        n_up = len(blocks_up)
        for i in range(n_up):
            sample_in_channels = min(filters * 2 ** (n_up - i), 4)
            print(f"sample_in_channels: {sample_in_channels}")
            up_layers.append(
                nn.Sequential(
                    *[
                        ResBlock(spatial_dims, sample_in_channels // 2, norm=norm, act=self.act)
                        for _ in range(blocks_up[i])
                    ]
                )
            )
            up_samples.append(
                nn.Sequential(
                    *[
                        get_conv_layer(spatial_dims, sample_in_channels, sample_in_channels // 2, kernel_size=1),
                        get_upsample_layer(spatial_dims, sample_in_channels // 2, upsample_mode=upsample_mode),
                    ]
                )
            )
        return up_layers, up_samples