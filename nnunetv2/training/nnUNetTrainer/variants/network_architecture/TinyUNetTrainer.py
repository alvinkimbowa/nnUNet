import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from typing import Union, Tuple, List
from torch import nn
from torch._dynamo import OptimizedModule
import math
import torch.nn.functional as F
# from thop import clever_format, profile


class TinyUNetTrainer(nnUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   patch_size: List[int],
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return TinyUNet(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            out_filters=[64, 128, 256, 512],
            deep_supervision=enable_deep_supervision
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
        
        mod.deep_supervision = enabled


class TinyUNetTrainer_S32(TinyUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   patch_size: List[int],
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return TinyUNet(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            out_filters=[32, 64, 128, 256],
            deep_supervision=enable_deep_supervision
        )


class TinyUNetTrainer_S16(TinyUNetTrainer):
    @staticmethod
    def build_network_architecture(architecture_class_name: str,
                                   arch_init_kwargs: dict,
                                   arch_init_kwargs_req_import: Union[List[str], Tuple[str, ...]],
                                   num_input_channels: int,
                                   num_output_channels: int,
                                   patch_size: List[int],
                                   enable_deep_supervision: bool = True) -> nn.Module:

        return TinyUNet(
            in_channels=num_input_channels,
            num_classes=num_output_channels,
            out_filters=[16, 32, 64, 128],
            deep_supervision=enable_deep_supervision
        )


def autopad(k, p=None, d=1):  
    '''
    k: kernel
    p: padding
    d: dilation
    '''
    if d > 1:
        # actual kernel-size
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k] # actual kernel-size
    if p is None:
        # auto-pad
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, paddin g, groups, dilation, activation)."""
    default_act = nn.GELU()
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03, affine=True, track_running_stats=True)
        self.act    = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution with args(ch_in, ch_out, kernel, stride, dilation, activation)."""
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

    
# Lightweight Cascade Multi-Receptive Fields Module
class CMRF(nn.Module):
    """CMRF Module with args(ch_in, ch_out, number, shortcut, groups, expansion)."""
    def __init__(self, c1, c2, N=8, shortcut=True, g=1, e=0.5):
        super().__init__()
        
        self.N         = N
        self.c         = int(c2 * e / self.N)
        self.add       = shortcut and c1 == c2
        
        self.pwconv1   = Conv(c1, c2//self.N, 1, 1)
        self.pwconv2   = Conv(c2//2, c2, 1, 1)
        self.m         = nn.ModuleList(DWConv(self.c, self.c, k=3, act=False) for _ in range(N-1))

    def forward(self, x):
        """Forward pass through CMRF Module."""
        x_residual = x
        x          = self.pwconv1(x)

        x          = [x[:, 0::2, :, :], x[:, 1::2, :, :]]
        x.extend(m(x[-1]) for m in self.m)
        x[0]       = x[0] +  x[1] 
        x.pop(1)
        
        y          = torch.cat(x, dim=1) 
        y          = self.pwconv2(y)
        return x_residual + y if self.add else y


'''
U-shape/U-like Model
'''
# Encoder in TinyU-Net
class UNetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetEncoder, self).__init__()
        self.cmrf       = CMRF(in_channels, out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.cmrf(x)
        return self.downsample(x), x
    

# Decoder in TinyU-Net
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.cmrf      = CMRF(in_channels, out_channels)
        self.upsample  = F.interpolate
        
    def forward(self, x, skip_connection):
        x = self.upsample(x, scale_factor=2, mode='bicubic', align_corners=False)
        x = torch.cat([x, skip_connection], dim=1)
        x = self.cmrf(x)
        return x
    

# TinyU-Net
class TinyUNet(nn.Module):
    """TinyU-Net with args(in_channels, num_classes)."""
    '''
    in_channels: The number of input channels
    num_classes: The number of segmentation classes
    '''
    def __init__(self, in_channels=3, num_classes=2, out_filters=[64, 128, 256, 512], deep_supervision=True):
        super(TinyUNet, self).__init__()
        in_filters = [o * 3 for o in out_filters]
        in_filters[-1] = out_filters[-1] * 2
        
        self.encoder1   = UNetEncoder(in_channels, out_filters[0])
        self.encoder2   = UNetEncoder(out_filters[0], out_filters[1])
        self.encoder3   = UNetEncoder(out_filters[1], out_filters[2])
        self.encoder4   = UNetEncoder(out_filters[2], out_filters[3])

        self.decoder4   = UNetDecoder(in_filters[3], out_filters[3])
        self.decoder3   = UNetDecoder(in_filters[2], out_filters[2])
        self.decoder2   = UNetDecoder(in_filters[1], out_filters[1])
        self.decoder1   = UNetDecoder(in_filters[0], out_filters[0])
        self.final_conv = nn.Conv2d(out_filters[0], num_classes, kernel_size=1)
        self.deep_supervision = deep_supervision
        
    def forward(self, x):
        x, skip1 = self.encoder1(x)
        x, skip2 = self.encoder2(x)
        x, skip3 = self.encoder3(x)
        x, skip4 = self.encoder4(x)

        d4       = self.decoder4(x, skip4)
        d3       = self.decoder3(d4, skip3)
        d2       = self.decoder2(d3, skip2)
        d1       = self.decoder1(d2, skip1)
        d1       = self.final_conv(d1)

        if self.deep_supervision:
            return [d1, d2, d3, d4]
        else:
            return d1


if __name__ == '__main__':
    model         = TinyUNet(in_channels=3, num_classes=2)

    device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model         = model.to(device)

    # summary(model, (3, 256, 256))
        
    dummy_input   = torch.randn(1, 3, 256, 256).to(device)
    flops, params = profile(model, (dummy_input, ), verbose=False)
    #-------------------------------------------------------------------------------#
    #   flops * 2 because profile does not consider convolution as two operations.
    #-------------------------------------------------------------------------------#
    flops         = flops * 2
    flops, params = clever_format([flops, params], "%.4f")
    print(f'Total GFLOPs: {flops}')
    print(f'Total Params: {params}')