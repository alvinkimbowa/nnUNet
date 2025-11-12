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


