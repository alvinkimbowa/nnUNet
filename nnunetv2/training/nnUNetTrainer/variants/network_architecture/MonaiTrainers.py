import monai.networks.nets as nets
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from typing import Union, Tuple, List
from torch import nn
from torch._dynamo import OptimizedModule


class nnUNetTrainerNoDeepSupervision(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = False

    def set_deep_supervision_enabled(self, enabled: bool):
            """
            This function is specific for the default architecture in nnU-Net. If you change the architecture, there are
            chances you need to change this as well!
            """
            if self.is_ddp:
                mod = self.network.module
            else:
                mod = self.network
            if isinstance(mod, OptimizedModule):
                mod = mod._orig_mod
            
            mod.deep_supervision = enabled


class AttentionUNetTrainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   patch_size: List[int],
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return nets.AttentionUnet(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            channels=arch_init_kwargs['features_per_stage'],
            strides=arch_init_kwargs['strides'],
        )


class UNetPlusPlusTrainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   patch_size: List[int],
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return nets.BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
        )


class SegResNetTrainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   patch_size: List[int],
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return nets.SegResNet(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
        )


class UNETRTrainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   patch_size: List[int],
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return nets.UNETR(
            img_size=patch_size,
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
        )


class SwinUNETRTrainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   patch_size: List[int],
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return nets.SwinUNETR(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
        )

