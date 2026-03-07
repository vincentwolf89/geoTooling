"""U-Net model met attention gates voor dijk-segmentatie."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_unet(
    in_channels: int = 2,
    num_classes: int = 6,
    base_features: int = 32,
) -> nn.Module:
    """Bouw een U-Net met attention gates voor segmentatie van dijkonderdelen.

    Parameters
    ----------
    in_channels : int
        Aantal inputkanalen (bijv. 5 = DTM + slope + R + G + B).
    num_classes : int
        Aantal outputklassen.
    base_features : int
        Aantal features in de eerste laag (verdubbelt per encoder-blok).
    """
    return AttentionUNet(in_channels, num_classes, base_features)


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AttentionGate(nn.Module):
    """Attention gate: leert welke skip-connection features relevant zijn."""

    def __init__(self, gate_ch: int, skip_ch: int, inter_ch: int):
        super().__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.W_skip = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, 1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(inter_ch, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        g = self.W_gate(gate)
        s = self.W_skip(skip)
        attn = self.psi(self.relu(g + s))
        return skip * attn


class AttentionUNet(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, base: int = 32):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4, dropout=0.1)
        self.enc4 = ConvBlock(base * 4, base * 8, dropout=0.2)

        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(base * 8, base * 16, dropout=0.3)

        # Decoder met attention gates
        self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
        self.attn4 = AttentionGate(base * 8, base * 8, base * 4)
        self.dec4 = ConvBlock(base * 16, base * 8, dropout=0.2)

        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.attn3 = AttentionGate(base * 4, base * 4, base * 2)
        self.dec3 = ConvBlock(base * 8, base * 4, dropout=0.1)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.attn2 = AttentionGate(base * 2, base * 2, base)
        self.dec2 = ConvBlock(base * 4, base * 2)

        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.attn1 = AttentionGate(base, base, base // 2)
        self.dec1 = ConvBlock(base * 2, base)

        # Deep supervision: auxiliary outputs van diepere lagen
        self.aux3 = nn.Conv2d(base * 4, num_classes, 1)
        self.aux2 = nn.Conv2d(base * 2, num_classes, 1)

        self.final = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder met attention
        u4 = self.up4(b)
        a4 = self.attn4(u4, e4)
        d4 = self.dec4(torch.cat([u4, a4], dim=1))

        u3 = self.up3(d4)
        a3 = self.attn3(u3, e3)
        d3 = self.dec3(torch.cat([u3, a3], dim=1))

        u2 = self.up2(d3)
        a2 = self.attn2(u2, e2)
        d2 = self.dec2(torch.cat([u2, a2], dim=1))

        u1 = self.up1(d2)
        a1 = self.attn1(u1, e1)
        d1 = self.dec1(torch.cat([u1, a1], dim=1))

        out = self.final(d1)

        if self.training:
            # Deep supervision: return aux outputs voor extra loss
            aux3_out = F.interpolate(self.aux3(d3), size=x.shape[2:], mode="bilinear", align_corners=False)
            aux2_out = F.interpolate(self.aux2(d2), size=x.shape[2:], mode="bilinear", align_corners=False)
            return out, aux3_out, aux2_out

        return out
