import monai.networks.nets as nets
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from typing import Union, Tuple, List
from torch import nn
from torch._dynamo import OptimizedModule
from torch import autocast
from nnunetv2.utilities.helpers import dummy_context
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from torch.nn.parallel import DistributedDataParallel as DDP


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
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return nets.AttentionUnet(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            channels=arch_init_kwargs['features_per_stage'],
            strides=arch_init_kwargs['strides'],
        )


class UNetPlusPlusTrainerNoDeepSupervision(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return nets.BasicUNetPlusPlus(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
            deep_supervision=enable_deep_supervision,
        )
    
    def train_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        self.optimizer.zero_grad(set_to_none=True)
        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            # del data
            if self.enable_deep_supervision:
                target = [target[0] for _ in range(len(output))]
            else:
                output = output[0]
            l = self.loss(output, target)

        if self.grad_scaler is not None:
            self.grad_scaler.scale(l).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            l.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        return {'loss': l.detach().cpu().numpy()}
    
    def validation_step(self, batch: dict) -> dict:
        data = batch['data']
        target = batch['target']

        data = data.to(self.device, non_blocking=True)
        if isinstance(target, list):
            target = [i.to(self.device, non_blocking=True) for i in target]
        else:
            target = target.to(self.device, non_blocking=True)

        # Autocast can be annoying
        # If the device_type is 'cpu' then it's slow as heck and needs to be disabled.
        # If the device_type is 'mps' then it will complain that mps is not implemented, even if enabled=False is set. Whyyyyyyy. (this is why we don't make use of enabled=False)
        # So autocast will only be active if we have a cuda device.
        with autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            output = self.network(data)
            del data
            if self.enable_deep_supervision:
                target = [target[0] for _ in range(len(output))]
            else:
                output = output[0]
            l = self.loss(output, target)

        # we only need the output with the highest output resolution (if DS enabled)
        if self.enable_deep_supervision:
            output = output[0]
            target = target[0]

        # the following is needed for online evaluation. Fake dice (green line)
        axes = [0] + list(range(2, output.ndim))

        if self.label_manager.has_regions:
            predicted_segmentation_onehot = (torch.sigmoid(output) > 0.5).long()
        else:
            # no need for softmax
            output_seg = output.argmax(1)[:, None]
            predicted_segmentation_onehot = torch.zeros(output.shape, device=output.device, dtype=torch.float32)
            predicted_segmentation_onehot.scatter_(1, output_seg, 1)
            del output_seg

        if self.label_manager.has_ignore_label:
            if not self.label_manager.has_regions:
                mask = (target != self.label_manager.ignore_label).float()
                # CAREFUL that you don't rely on target after this line!
                target[target == self.label_manager.ignore_label] = 0
            else:
                if target.dtype == torch.bool:
                    mask = ~target[:, -1:]
                else:
                    mask = 1 - target[:, -1:]
                # CAREFUL that you don't rely on target after this line!
                target = target[:, :-1]
        else:
            mask = None

        tp, fp, fn, _ = get_tp_fp_fn_tn(predicted_segmentation_onehot, target, axes=axes, mask=mask)

        tp_hard = tp.detach().cpu().numpy()
        fp_hard = fp.detach().cpu().numpy()
        fn_hard = fn.detach().cpu().numpy()
        if not self.label_manager.has_regions:
            # if we train with regions all segmentation heads predict some kind of foreground. In conventional
            # (softmax training) there needs tobe one output for the background. We are not interested in the
            # background Dice
            # [1:] in order to remove background
            tp_hard = tp_hard[1:]
            fp_hard = fp_hard[1:]
            fn_hard = fn_hard[1:]

        return {'loss': l.detach().cpu().numpy(), 'tp_hard': tp_hard, 'fp_hard': fp_hard, 'fn_hard': fn_hard}


class UNetPlusPlusTrainer(UNetPlusPlusTrainerNoDeepSupervision):
    def __init__(self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.enable_deep_supervision = True


class SegResNetTrainer(nnUNetTrainerNoDeepSupervision):
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
        )

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
        )


class SegResNetXTiny4Trainer(nnUNetTrainerNoDeepSupervision):
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
    
    def initialize(self):
        if not self.was_initialized:
            ## DDP batch size and oversampling can differ between workers and needs adaptation
            # we need to change the batch size in DDP because we don't use any of those distributed samplers
            self._set_batch_size_and_oversample()

            self.num_input_channels = determine_num_input_channels(self.plans_manager, self.configuration_manager,
                                                                   self.dataset_json)

            self.network = self.build_network_architecture(
                self.configuration_manager.network_arch_class_name,
                self.configuration_manager.network_arch_init_kwargs,
                self.configuration_manager.network_arch_init_kwargs_req_import,
                self.num_input_channels,
                self.label_manager.num_segmentation_heads,
                self.configuration_manager.patch_size,
                self.enable_deep_supervision
            ).to(self.device)
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

            self.dataset_class = infer_dataset_class(self.preprocessed_dataset_folder)

            # torch 2.2.2 crashes upon compiling CE loss
            # if self._do_i_compile():
            #     self.loss = torch.compile(self.loss)
            self.was_initialized = True
        else:
            raise RuntimeError("You have called self.initialize even though the trainer was already initialized. "
                               "That should not happen.")


class SwinUNETRTrainer(nnUNetTrainerNoDeepSupervision):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return nets.SwinUNETR(
            spatial_dims=2,
            in_channels=num_input_channels,
            out_channels=num_output_channels,
        )

