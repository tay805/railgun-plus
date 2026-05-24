"""The map-based policy network.

This is the RAILGUN backbone: a 5-level U-Net that maps an (k, H, W) feature
stack to (5, H, W) per-cell action logits. Because it is fully convolutional,
it handles maps of varying size (given H, W divisible by 16).

DESIGN SEAM: the rest of the project only depends on the `MapPolicy` protocol
(a module taking (B,k,H,W) -> (B,5,H,W)). If you later want to try a non-U-Net
map-based backbone (e.g. a CNN with a transformer bottleneck for global
coordination), implement the same signature and the data pipeline, training
loop, corrector, and eval all work unchanged. See research plan Phase 2/3.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _double_conv(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class RailgunUNet(nn.Module):
    """5-level U-Net. ~30M params at base=64 (matches RAILGUN's stated size).

    Args:
        in_channels: k, number of input feature channels (default 6).
        num_actions: output channels (5: wait/up/down/left/right).
        base: channel width of the first level (64 in the paper).
    """

    def __init__(self, in_channels: int = 6, num_actions: int = 5, base: int = 64):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]  # 64..1024

        # Encoder
        self.enc1 = _double_conv(in_channels, c[0])
        self.enc2 = _double_conv(c[0], c[1])
        self.enc3 = _double_conv(c[1], c[2])
        self.enc4 = _double_conv(c[2], c[3])
        self.bottleneck = _double_conv(c[3], c[4])
        self.pool = nn.MaxPool2d(2)

        # Decoder (transposed conv upsampling, as in original U-Net — RAILGUN
        # explicitly avoids bilinear interpolation).
        self.up4 = nn.ConvTranspose2d(c[4], c[3], 2, stride=2)
        self.dec4 = _double_conv(c[4], c[3])
        self.up3 = nn.ConvTranspose2d(c[3], c[2], 2, stride=2)
        self.dec3 = _double_conv(c[3], c[2])
        self.up2 = nn.ConvTranspose2d(c[2], c[1], 2, stride=2)
        self.dec2 = _double_conv(c[2], c[1])
        self.up1 = nn.ConvTranspose2d(c[1], c[0], 2, stride=2)
        self.dec1 = _double_conv(c[1], c[0])

        self.head = nn.Conv2d(c[0], num_actions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)  # (B, 5, H, W) logits

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
