import monai.networks.nets as nets
import torch
from mono2D import Mono2D
from torch.nn.parallel import DistributedDataParallel as DDP
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from dynamic_network_architectures.initialization.weight_init import InitWeights_He
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from typing import Tuple, Union, List

class nnUNetTrainerCustom(nnUNetTrainer):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        epochs: int,
        save_every: int,
        model_name: str,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, epochs, save_every,
                         model_name, unpack_dataset, device
                         )
        self.enable_deep_supervision = False

    @staticmethod
    def build_network_architecture(model_name: str, patch_size,
                                   architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> torch.nn.Module:
        """
        This is where you build the architecture according to the plans. There is no obligation to use
        get_network_from_plans, this is just a utility we use for the nnU-Net default architectures. You can do what
        you want. Even ignore the plans and just return something static (as long as it can process the requested
        patch size)
        but don't bug us with your bugs arising from fiddling with this :-P
        This is the function that is called in inference as well! This is needed so that all network architecture
        variants can be loaded at inference time (inference will use the same nnUNetTrainer that was used for
        training, so if you change the network architecture during training by deriving a new trainer class then
        inference will know about it).

        If you need to know how many segmentation outputs your custom architecture needs to have, use the following snippet:
        > label_manager = plans_manager.get_label_manager(dataset_json)
        > label_manager.num_segmentation_heads
        (why so complicated? -> We can have either classical training (classes) or regions. If we have regions,
        the number of outputs is != the number of classes. Also there is the ignore label for which no output
        should be generated. label_manager takes care of all that for you.)

        """
        
        if "monogenic" in model_name:
            model_name = model_name.replace("monogenic", "")
            model_name = model_name.replace("_", "")
            mono = True
            nscale = arch_init_kwargs["nscale"]
            return_phase = arch_init_kwargs["return_phase"]
            return_phase_asym = arch_init_kwargs["return_phase_asym"]
            trainable = arch_init_kwargs["trainable"]
            monogenic_kwargs = ["nscale", "return_phase", "return_phase_asym", "trainable"]
            arch_init_kwargs = {k:v for k,v in arch_init_kwargs.items() if k not in monogenic_kwargs}
        else:
            mono = False
        
        if model_name == "nnunet":
            net = get_network_from_plans(
                architecture_class_name,
                arch_init_kwargs,
                arch_init_kwargs_req_import,
                num_input_channels,
                num_output_channels,
                allow_init=True,
                deep_supervision=enable_deep_supervision
            )
        elif model_name == "UNETR":
            net = nets.UNETR(
                img_size=patch_size,
                spatial_dims=2,
                in_channels=num_input_channels,
                out_channels=num_output_channels,
            )
        elif model_name == "UNetplusplus":
            net = nets.BasicUNetPlusPlus(
                spatial_dims=2,
                in_channels=num_input_channels,
                out_channels=num_output_channels,
            )
        elif model_name == "UNet":
            net = nets.UNet(
                spatial_dims=2,
                in_channels=num_input_channels,
                out_channels=num_output_channels,
            )
        elif model_name == "SegResNet":
            net = nets.SegResNet(
                spatial_dims=2,
                in_channels=num_input_channels,
                out_channels=num_output_channels,
            )
        elif model_name == "SwinUNETR":
            net = nets.SwinUNETR(
                img_size=patch_size,
                spatial_dims=2,
                in_channels=num_input_channels,
                out_channels=num_output_channels,
            )
        else:
            raise ValueError(f"Unknown model name: {model_name}. Supported models are: nnunet, UNETR, UNetplusplus, UNet, SegResNet, and SwinUNETR.")

        if model_name == "nnunet":
            net.monogenic = False
        else:
            net = OrigModel(net)
        
        if mono:
            return MonoModel(net, nscale, return_phase, return_phase_asym, trainable)
        else:
            return net


    def initialize(self):
        if not self.was_initialized:
            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.model_name,
                self.configuration_manager.patch_size,
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.enable_deep_supervision).to(self.device)
            # compile network for free speedup
            if self._do_i_compile():
                self.print_to_log_file('Using torch.compile...')
                self.network = torch.compile(self.network)

            self.optimizer, self.lr_scheduler = self.configure_optimizers()
            # if ddp, wrap in DDP wrapper
            if self.is_ddp:
                self.network = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.network)
                self.network = DDP(self.network, device_ids=[self.local_rank])

            self.loss = self._build_loss()
            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")
        

class MonoModel(torch.nn.Module):
    def __init__(self, model, nscale=1, return_phase=True, return_phase_asym=False, trainable=True):
        super(MonoModel, self).__init__()
        self.model = model
        self.mono = Mono2D(nscale=nscale, return_phase=return_phase, return_phase_asym=return_phase_asym, trainable=trainable)
        self.monogenic = True
    
    def forward(self, x):
        x = self.mono(x)
        x = self.model(x)
        return x


class OrigModel(torch.nn.Module):
    def __init__(self, model):
        super(OrigModel, self).__init__()
        self.model = model
        self.model.apply(self.initialize)
        self.monogenic = False
    
    def forward(self, x):
        x = self.model(x)
        if isinstance(x, list):
            x = x[0]
        return x

    @staticmethod
    def initialize(module):
        InitWeights_He(1e-2)(module)
