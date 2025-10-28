# modules.py
# Author: Janvhi Sharma (COMP3710 Project 7)
# Improved UNet3D with residual conv blocks, batch norm, and dropout

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_channels, out_channels):
    return nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)


class ConvBlock(nn.Module):
    """Conv → BN → ReLU → Dropout → Conv → BN → ReLU with residual skip"""
    def __init__(self, in_c, out_c, p_drop=0.1):
        super().__init__()
        self.conv1 = conv3x3x3(in_c, out_c)
        self.bn1 = nn.BatchNorm3d(out_c)
        self.conv2 = conv3x3x3(out_c, out_c)
        self.bn2 = nn.BatchNorm3d(out_c)
        self.drop = nn.Dropout3d(p_drop)
        self.res = nn.Conv3d(in_c, out_c, kernel_size=1) if in_c != out_c else nn.Identity()

    def forward(self, x):
        identity = self.res(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop(x)
        x = F.relu(self.bn2(self.conv2(x)) + identity)
        return x


class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool = nn.MaxPool3d(2)
        self.block = ConvBlock(in_c, out_c)

    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = ConvBlock(in_c, out_c)

    def forward(self, x_dec, x_enc):
        x_dec = F.interpolate(x_dec, scale_factor=2, mode="trilinear", align_corners=False)
        diffD = x_enc.size(2) - x_dec.size(2)
        diffH = x_enc.size(3) - x_dec.size(3)
        diffW = x_enc.size(4) - x_dec.size(4)
        x_dec = F.pad(x_dec, [diffW // 2, diffW - diffW // 2,
                              diffH // 2, diffH - diffH // 2,
                              diffD // 2, diffD - diffD // 2])
        x = torch.cat([x_enc, x_dec], dim=1)
        return self.block(x)


class UNet3D(nn.Module):
    """Improved UNet3D Architecture"""
    def __init__(self, in_channels=1, out_channels=2, base_c=16, p_drop=0.1):
        super().__init__()
        self.enc1 = ConvBlock(in_channels, base_c, p_drop)
        self.enc2 = Down(base_c, base_c * 2)
        self.enc3 = Down(base_c * 2, base_c * 4)
        self.enc4 = Down(base_c * 4, base_c * 8)

        self.bottleneck = ConvBlock(base_c * 8, base_c * 16)

        self.up1 = Up(base_c * 16 + base_c * 8, base_c * 8)
        self.up2 = Up(base_c * 8 + base_c * 4, base_c * 4)
        self.up3 = Up(base_c * 4 + base_c * 2, base_c * 2)
        self.up4 = Up(base_c * 2 + base_c, base_c)

        self.out_conv = nn.Conv3d(base_c, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        d1 = self.up1(b, e4)
        d2 = self.up2(d1, e3)
        d3 = self.up3(d2, e2)
        d4 = self.up4(d3, e1)
        return self.out_conv(d4)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        dims = (0, 2, 3, 4)
        intersection = torch.sum(probs * targets, dims)
        denom = torch.sum(probs + targets, dims)
        dice = (2. * intersection + self.smooth) / (denom + self.smooth)
        return 1 - dice.mean()
