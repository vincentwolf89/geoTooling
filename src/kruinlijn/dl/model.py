"""Segmentatiemodel voor dijk-classificatie.

Primair: segmentation_models_pytorch met pretrained ResNet34 encoder.
Fallback: custom Attention U-Net met ASPP + SE-blokken.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_unet(
    in_channels: int = 9,
    num_classes: int = 10,
    base_features: int = 32,
    encoder_name: str = "resnet34",
    use_pretrained: bool = True,
) -> nn.Module:
    """Bouw een segmentatiemodel.

    Probeert eerst smp.UnetPlusPlus met pretrained encoder.
    Valt terug op custom AttentionUNet als smp niet beschikbaar is.

    Parameters
    ----------
    in_channels : int
        Aantal inputkanalen (9 = DTM + slope + aspect + curvature + TPI + nDSM + R + G + B).
    num_classes : int
        Aantal outputklassen.
    base_features : int
        Aantal features in de eerste laag (alleen voor fallback model).
    encoder_name : str
        Encoder backbone voor smp (bijv. 'resnet34', 'resnet50', 'efficientnet-b3').
    use_pretrained : bool
        Gebruik ImageNet pretrained weights voor encoder.
    """
    try:
        import segmentation_models_pytorch as smp

        weights = "imagenet" if use_pretrained else None
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=num_classes,
            decoder_attention_type="scse",  # Spatial + Channel SE attention
        )
        print(f"  Model: smp.UnetPlusPlus + {encoder_name} (pretrained={use_pretrained})")
        return model

    except ImportError:
        print("  Model: custom AttentionUNet (smp niet beschikbaar)")
        return AttentionUNet(in_channels, num_classes, base_features)


# ---- Fallback: custom Attention U-Net ----

class SEBlock(nn.Module):
    """Squeeze-Excitation block: leert per-kanaal gewichten."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1, 1)
        return x * w


class ConvBlock(nn.Module):
    """Conv-BN-ReLU blok met optionele SE-attention en residual connection."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0, use_se: bool = True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Conv2d(in_ch, out_ch, 1, bias=False) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        out = self.se(out)
        out = self.dropout(out)
        return out + self.shortcut(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling -- multi-scale feature extraction."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv3x3_r6 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=6, dilation=6, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv3x3_r12 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=12, dilation=12, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * 4, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        feat1 = self.conv1x1(x)
        feat2 = self.conv3x3_r6(x)
        feat3 = self.conv3x3_r12(x)
        feat4 = F.interpolate(self.pool(x), size=(h, w), mode="bilinear", align_corners=False)
        return self.project(torch.cat([feat1, feat2, feat3, feat4], dim=1))


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
        self.enc1 = ConvBlock(in_channels, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4, dropout=0.1)
        self.enc4 = ConvBlock(base * 4, base * 8, dropout=0.2)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck_conv = ConvBlock(base * 8, base * 16, dropout=0.3)
        self.aspp = ASPP(base * 16, base * 16)

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

        self.aux3 = nn.Conv2d(base * 4, num_classes, 1)
        self.aux2 = nn.Conv2d(base * 2, num_classes, 1)

        self.final = nn.Conv2d(base, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck_conv(self.pool(e4))
        b = self.aspp(b)

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
            aux3_out = F.interpolate(self.aux3(d3), size=x.shape[2:], mode="bilinear", align_corners=False)
            aux2_out = F.interpolate(self.aux2(d2), size=x.shape[2:], mode="bilinear", align_corners=False)
            return out, aux3_out, aux2_out

        return out
