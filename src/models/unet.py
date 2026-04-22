"""
src/models/unet.py
------------------
U-Net wrapper using segmentation-models-pytorch.
Supports pretrained encoder backbones from ImageNet.

Supported encoders:
  resnet34, resnet50, resnet101
  efficientnet-b0 through efficientnet-b7
  ...and many more (full list: https://smp.readthedocs.io)
"""

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_unet(
    encoder_name: str = "resnet50",
    pretrained: bool = True,
    in_channels: int = 3,
    num_classes: int = 1,
) -> nn.Module:
    """
    Build a U-Net with a pretrained encoder backbone.

    Architecture:
      Encoder (pretrained)  →  Skip connections  →  Decoder  →  Segmentation head
      Standard U-Net skip connections preserved at 5 resolution levels.

    Args:
        encoder_name: Backbone name (e.g., 'resnet50', 'efficientnet-b4').
        pretrained:   Load ImageNet weights for the encoder.
        in_channels:  Input channels (3 for RGB satellite imagery).
        num_classes:  Output channels (1 for binary road mask).

    Returns:
        nn.Module — outputs raw logits [B, 1, H, W].
    """
    encoder_weights = "imagenet" if pretrained else None

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,          # Raw logits — sigmoid applied in loss/metrics
        decoder_use_batchnorm=True,
    )

    # Log total parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[U-Net ({encoder_name})] "
          f"Total params: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")

    return model
