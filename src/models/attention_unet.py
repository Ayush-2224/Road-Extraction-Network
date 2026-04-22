"""
src/models/attention_unet.py
----------------------------
Attention U-Net: U-Net augmented with learnable Attention Gates on skip connections.

Why it helps for urban roads:
  - Attention Gates learn to suppress irrelevant background features
  - Focus decoder on activated road regions (intersections, lanes)
  - Especially effective for thin/narrow roads in dense urban scenes

Architecture:
  - Pretrained SMP encoder extracts multi-scale features
  - Custom decoder with Attention Gates before each skip connection merge
  - Attention Gate uses gating signal (g) from upsampled path
    and skip feature (x) from encoder to produce spatial attention map

Reference:
  Oktay et al. (2018) — "Attention U-Net: Learning Where to Look for the Pancreas"
  https://arxiv.org/abs/1804.03999
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_encoder
from typing import List


# ──────────────────────────────────────────────────────────
#  Attention Gate
# ──────────────────────────────────────────────────────────

class AttentionGate(nn.Module):
    """
    Additive Attention Gate.

    Computes a spatial attention map α ∈ [0, 1] over the skip connection x,
    conditioned on the gating signal g from the upsampled decoder path.

    α = σ( ψ( ReLU( Wg·g + Wx·x ) ) )
    out = α ⊙ x

    Args:
        F_g:   Number of channels in gating signal g (from decoder).
        F_x:   Number of channels in skip connection x (from encoder).
        F_int: Intermediate dimension (usually F_x // 2).
    """

    def __init__(self, F_g: int, F_x: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, bias=False),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            g: Gating signal [B, F_g, H', W'] — from decoder (upsampled)
            x: Skip connection [B, F_x, H, W]  — from encoder
        Returns:
            Attended skip features [B, F_x, H, W]
        """
        # Upsample g to match x spatial size if needed
        if g.shape[2:] != x.shape[2:]:
            g = F.interpolate(g, size=x.shape[2:], mode="bilinear", align_corners=False)

        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


# ──────────────────────────────────────────────────────────
#  Decoder Block
# ──────────────────────────────────────────────────────────

class AttentionDecoderBlock(nn.Module):
    """
    Single decoder block: Upsample + Attention Gate + Concat + Conv Conv.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.attn_gate = AttentionGate(
            F_g=in_channels,
            F_x=skip_channels,
            F_int=max(skip_channels // 2, 16),
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        skip_attended = self.attn_gate(g=x, x=skip)
        x = torch.cat([x, skip_attended], dim=1)
        return self.conv(x)


# ──────────────────────────────────────────────────────────
#  Attention U-Net
# ──────────────────────────────────────────────────────────

class AttentionUNet(nn.Module):
    """
    Attention U-Net with pretrained SMP encoder.

    Encoder: Pretrained ResNet / EfficientNet (from segmentation-models-pytorch)
    Decoder: Custom 5-level decoder with Attention Gates on all skip connections
    Head:    1×1 Conv → binary road logit

    Args:
        encoder_name:    Backbone name (e.g., 'resnet50', 'efficientnet-b4').
        encoder_weights: 'imagenet' or None.
        in_channels:     Input channels (3 for RGB).
        num_classes:     Output channels (1 for binary road mask).
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 1,
    ):
        super().__init__()

        # ── Encoder ──
        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=5,
            weights=encoder_weights,
        )
        # out_channels is a tuple: (in_ch, s1, s2, s3, s4, s5)
        enc_channels = self.encoder.out_channels   # e.g. (3, 64, 256, 512, 1024, 2048) for resnet50

        # ── Bottleneck conv (s5 → bottleneck) ──
        self.bottleneck = nn.Sequential(
            nn.Conv2d(enc_channels[-1], enc_channels[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(enc_channels[-1]),
            nn.ReLU(inplace=True),
        )

        # ── Decoder blocks (from deepest to shallowest) ──
        # We use 4 decoder blocks for a depth-5 encoder
        # Channels: s5→s4, s4→s3, s3→s2, s2→s1
        decoder_out = [256, 128, 64, 32]
        self.decoder_blocks = nn.ModuleList()

        in_ch = enc_channels[-1]   # bottleneck output channels
        skip_chs = list(reversed(enc_channels[1:-1]))  # [s4, s3, s2, s1]

        for i, (skip_ch, out_ch) in enumerate(zip(skip_chs, decoder_out)):
            self.decoder_blocks.append(
                AttentionDecoderBlock(in_channels=in_ch, skip_channels=skip_ch, out_channels=out_ch)
            )
            in_ch = out_ch

        # Final upsample to reach input resolution (no skip for last level)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, decoder_out[-1], 3, padding=1, bias=False),
            nn.BatchNorm2d(decoder_out[-1]),
            nn.ReLU(inplace=True),
        )

        # ── Segmentation Head ──
        self.seg_head = nn.Conv2d(decoder_out[-1], num_classes, kernel_size=1)

        # Log params
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Attention U-Net ({encoder_name})] "
              f"Total params: {total/1e6:.1f}M | Trainable: {trainable/1e6:.1f}M")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input size for later upsampling reference
        input_size = x.shape[2:]

        # Encoder: features[0]=input, features[1..5]=s1..s5
        features = self.encoder(x)
        enc_skips = list(reversed(features[1:-1]))  # [s4, s3, s2, s1]
        bottleneck = self.bottleneck(features[-1])

        # Decoder with attention gates
        d = bottleneck
        for block, skip in zip(self.decoder_blocks, enc_skips):
            d = block(d, skip)

        d = self.final_upsample(d)

        # Ensure output matches input resolution
        if d.shape[2:] != input_size:
            d = F.interpolate(d, size=input_size, mode="bilinear", align_corners=False)

        return self.seg_head(d)


def build_attention_unet(
    encoder_name: str = "resnet50",
    pretrained: bool = True,
    in_channels: int = 3,
    num_classes: int = 1,
) -> nn.Module:
    """Factory function for Attention U-Net."""
    encoder_weights = "imagenet" if pretrained else None
    return AttentionUNet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        num_classes=num_classes,
    )
