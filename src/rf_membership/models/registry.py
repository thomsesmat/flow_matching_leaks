"""Single entry point to build a velocity network: ``build_backbone(name, shape, size)``.

name:  'transformer' | 'unet' | 'linear'
shape: latent shape without batch dim, (C, T) for sequences or (C, H, W) for images (dispatches on its length).
size:  Transformer size ('small' | 'medium' | 'large'); ignored by 'unet' and 'linear'.
"""

from rf_membership.config import ARCHITECTURES, UNETS

MODELS = ("transformer", "unet", "linear")


def build_backbone(name, shape, size="medium"):
    c, image = shape[0], len(shape) == 3
    if name == "transformer":
        from rf_membership.models.transformer import Transformer
        from rf_membership.models.transformer_image import ImageTransformer

        arch = ARCHITECTURES[size]
        if image:
            return ImageTransformer(in_channels=c, patch_size=2, image_size=shape[1], pos_enc="RoPE", **arch)
        return Transformer(input_size=c, pos_enc="RoPE", **arch)
    if name == "unet":
        if image:
            from rf_membership.models.unet_2d import UNet2D

            return UNet2D(in_channels=c, **UNETS["2d"])
        from rf_membership.models.unet_1d import UNet1D

        return UNet1D(in_channels=c, **UNETS["1d"])
    if name == "linear":
        from rf_membership.models.linear import Linear

        return Linear(input_size=c)
    raise ValueError(f"Unknown model '{name}', choose from {MODELS}")
