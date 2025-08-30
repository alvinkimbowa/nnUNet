from typing import Union, Tuple, List
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.network_architecture.unext.archs import UNext_S
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.UNeXtTrainer import UNeXtTrainer


class UNeXtTrainer_S(UNeXtTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        
        return UNext_S(
            num_classes=num_output_channels,
            input_channels=num_input_channels,
            deep_supervision=enable_deep_supervision,
        )