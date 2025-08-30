from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from typing import Union, Tuple, List
from torch import nn
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.unext.archs import UNext
import torch
from torch._dynamo import OptimizedModule


class UNeXtTrainer(nnUNetTrainer):
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

    
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        print("num_input_channels: ", num_input_channels)
        print("num_output_channels: ", num_output_channels)
        print("enable_deep_supervision: ", enable_deep_supervision)
        print("arch_init_kwargs: ", arch_init_kwargs)

        return UNext(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=enable_deep_supervision,
        )
    
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