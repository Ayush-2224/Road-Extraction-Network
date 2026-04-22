"""
src/models/deeplabv3plus.py
---------------------------
DeepLabV3+ wrapper using segmentation-models-pytorch.

Why DeepLabV3+ for urban roads?
  - ASPP (Atrous Spatial Pyramid Pooling): captures multi-scale context
    in a single pass using different dilation rates.
  - Better at combining wide-field context (large road structures)
    with fine-grained detail (thin lanes, markings).
  - Stronger for high-resolution satellite imagery than plain DeepLabV3.

Recommended encoders:
  - resnet50 / resnet101 — strong baseline
  - efficientnet-b4 / efficientnet-b5 — best accuracy / compute tradeoff
"""

import segmentation_models_pytorch as smp
import torch.nn as nn


def build_deeplabv3plus(
    encoder_name: str = "resnet50",
    pretrained: bool = True,
    in_channels: int = 3,
    num_classes: int = 1,
) -> nn.Module:
    """
    Build DeepLabV3+ with pretrained encoder backbone.

    Architecture:
      Encoder (pretrained ASPP) → Decoder (4× upsampling) → Segmentation head
      ASPP uses dilation rates [6, 12, 18] for multi-scale context.

    Args:
        encoder_name: Backbone (e.g., 'resnet101', 'efficientnet-b5').
        pretrained:   Load ImageNet weights.
        in_channels:  Input channels (3 for RGB).
        num_classes:  Output channels (1 for binary road mask).

    Returns:
        nn.Module — raw logits [B, 1, H, W].
    """
    encoder_weights = "imagenet" if pretrained else None

    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        encoder_output_stride=16,    # 16 → better spatial resolution than 32
        decoder_channels=256,
        decoder_atrous_rates=(6, 12, 18),
        in_channels=in_channels,
        classes=num_classes,
        activation=None,             # Raw logits
    )

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[DeepLabV3+ ({encoder_name})] "
          f"Total params: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")

    return model
