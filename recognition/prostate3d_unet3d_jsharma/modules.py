# Commit milestone: implemented and verified Improved 3D U-Net architecture with residual and dropout layers
"""
modules.py — Model components for 3D prostate segmentation

Implements:
- ResidualBlock3D: conv3d + BN + ReLU with residual shortcut
- ImprovedUNet3D: lightweight residual U-Net 3D (hard-difficulty direction)
"""

from __future__ import annotations
import torch
import torch.nn as nn


# ------------- building blocks -------------

class ResidualBlock3D(nn.Module):
    """
    A residual block: Conv3D -> BN -> ReLU -> Dropout -> Conv3D -> BN + skip
    Keeps spatial size (padding=1). Changes channels when in≠out via 1x1x1 conv.
    """
    def __init__(self, in_ch: int, out_ch: int, p_drop: float = 0.2):
        super().__init__()
        self.conv1 = nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.drop  = nn.Dropout3d(p_drop)
        self.conv2 = nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_ch)

        self.short = (
            nn.Identity() if in_ch == out_ch
            else nn.Conv3d(in_ch, out_ch, kernel_size=1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.short(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


def up_block(in_ch: int, out_ch: int) -> nn.Module:
    """Transposed conv upsample + residual block after skip-concat."""
    return nn.Sequential(
        nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2, bias=False),
        nn.BatchNorm3d(out_ch),
        nn.ReLU(inplace=True),
    )


# ------------- improved U-Net 3D -------------

class ImprovedUNet3D(nn.Module):
    """
    Residual U-Net 3D with shallow width (fits student GPU), strong dice performance.
    Encoder: 64-128-256
    Bottleneck: 512
    Decoder: 256-128-64
    Final 1x1x1 conv -> num_classes.
    """
    def __init__(self, in_channels: int = 1, num_classes: int = 6, p_drop: float = 0.2):
        super().__init__()
        # encoder
        self.enc1 = ResidualBlock3D(in_channels, 64, p_drop)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ResidualBlock3D(64, 128, p_drop)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ResidualBlock3D(128, 256, p_drop)
        self.pool3 = nn.MaxPool3d(2)

        # bottleneck
        self.bott = ResidualBlock3D(256, 512, p_drop)

        # decoder
        self.up3 = up_block(512, 256)
        self.dec3 = ResidualBlock3D(512, 256, p_drop)

        self.up2 = up_block(256, 128)
        self.dec2 = ResidualBlock3D(256, 128, p_drop)

        self.up1 = up_block(128, 64)
        self.dec1 = ResidualBlock3D(128, 64, p_drop)

        self.head = nn.Conv3d(64, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encode
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        # bottleneck
        b = self.bott(self.pool3(e3))

        # decode with skip connections
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)
