"""FilaNet: a tiny, thin-structure-aware U-Net for solar filaments.

    input (3ch, full res) ----------------------------------------> DetailStem (1/1, 1/2)  -----+
           |                                                                                    |
           +--> pretrained tiny encoder (ConvNeXt-Atto/Femto, MobileNetV4, ...) 1/4..1/32        |
                         |                                                                      |
                 StripPooling context (long thin structures need long, narrow receptive fields)  |
                         |                                                                      |
                 light U-Net decoder with scSE gates  1/32 -> 1/4  --------------------------> 1/2 -> 1/1
                                                                                                |
                                                          heads: filament mask, filament spine  <+

Design choices:
* Patchify encoders (ConvNeXt) start at stride 4; filament barbs are 1-3 px wide, so a 2-layer
  full-resolution DetailStem feeds the last two decoder stages (cost: ~10k params).
* Strip pooling (Hou et al., CVPR 2020) pools along whole rows / columns: a cheap, anisotropic
  global context that matches elongated filaments.
* The spine head is supervised with MAGFiLO's annotated spines (unused by most entries); it
  regularises topology (continuity) and is a strong instance-level feature for rescoring.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def cba(cin, cout, k=3, s=1, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, k, s, k // 2, groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.GELU())


class DSConv(nn.Sequential):
    """Depthwise-separable 3x3 conv block."""

    def __init__(self, cin, cout):
        super().__init__(nn.Conv2d(cin, cin, 3, 1, 1, groups=cin, bias=False), nn.BatchNorm2d(cin), nn.GELU(),
                         nn.Conv2d(cin, cout, 1, bias=False), nn.BatchNorm2d(cout), nn.GELU())


class SCSE(nn.Module):
    def __init__(self, c, r=4):
        super().__init__()
        self.c = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(c // r, 8), 1), nn.GELU(),
                               nn.Conv2d(max(c // r, 8), c, 1), nn.Sigmoid())
        self.s = nn.Sequential(nn.Conv2d(c, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.c(x) + x * self.s(x)


class StripPooling(nn.Module):
    def __init__(self, c):
        super().__init__()
        mid = max(c // 4, 16)
        self.reduce = cba(c, mid, 1)
        self.h = nn.Sequential(nn.Conv2d(mid, mid, (1, 3), padding=(0, 1), bias=False), nn.BatchNorm2d(mid))
        self.v = nn.Sequential(nn.Conv2d(mid, mid, (3, 1), padding=(1, 0), bias=False), nn.BatchNorm2d(mid))
        self.fuse = nn.Sequential(nn.GELU(), nn.Conv2d(mid, c, 1, bias=False), nn.BatchNorm2d(c))

    def forward(self, x):
        z = self.reduce(x)
        h, w = z.shape[-2:]
        a = self.h(F.adaptive_avg_pool2d(z, (1, w))).expand(-1, -1, h, w)
        b = self.v(F.adaptive_avg_pool2d(z, (h, 1))).expand(-1, -1, h, w)
        return F.gelu(x + self.fuse(a + b))


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout, light=False):
        super().__init__()
        conv = DSConv if light else cba
        self.b1 = conv(cin + cskip, cout)
        self.b2 = conv(cout, cout)
        self.att = SCSE(cout)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        if skip is not None:
            if skip.shape[-2:] != x.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], 1)
        return self.att(self.b2(self.b1(x)))


class FilaNet(nn.Module):
    def __init__(self, encoder: str = "convnext_atto.d2_in1k", pretrained: bool = True, in_chans: int = 3,
                 dec_channels=(128, 80, 48, 24, 16), stem_channels=(16, 24), n_out: int = 2,
                 strip_pool: bool = True, drop_path: float = 0.05):
        super().__init__()
        kw = dict(features_only=True, pretrained=pretrained, in_chans=in_chans)
        try:
            self.encoder = timm.create_model(encoder, drop_path_rate=drop_path, **kw)
        except TypeError:
            self.encoder = timm.create_model(encoder, **kw)
        red = self.encoder.feature_info.reduction()
        chs = self.encoder.feature_info.channels()
        # keep stages with stride >= 4, up to 4 of them
        self.keep = [i for i, r in enumerate(red) if r >= 4][-4:]
        enc_ch = [chs[i] for i in self.keep]                      # strides 4, 8, 16, 32
        self.stem1 = cba(in_chans, stem_channels[0])                # 1/1
        self.stem2 = cba(stem_channels[0], stem_channels[1], s=2)   # 1/2
        self.context = StripPooling(enc_ch[-1]) if strip_pool else nn.Identity()
        d = dec_channels
        self.dec = nn.ModuleList([
            DecoderBlock(enc_ch[3], enc_ch[2], d[0]),               # -> 1/16
            DecoderBlock(d[0], enc_ch[1], d[1]),                    # -> 1/8
            DecoderBlock(d[1], enc_ch[0], d[2]),                    # -> 1/4
            DecoderBlock(d[2], stem_channels[1], d[3], light=True),  # -> 1/2
            DecoderBlock(d[3], stem_channels[0], d[4], light=True),  # -> 1/1
        ])
        self.head = nn.Conv2d(d[4], n_out, 1)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, x):
        s1 = self.stem1(x)
        s2 = self.stem2(s1)
        feats = self.encoder(x)
        f = [feats[i] for i in self.keep]
        y = self.context(f[3])
        y = self.dec[0](y, f[2])
        y = self.dec[1](y, f[1])
        y = self.dec[2](y, f[0])
        y = self.dec[3](y, s2)
        y = self.dec[4](y, s1)
        return self.head(y)


def build_model(cfg: dict | None = None, pretrained: bool = True) -> FilaNet:
    cfg = dict(cfg or {})
    cfg.pop("pretrained", None)
    return FilaNet(pretrained=pretrained, **cfg)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def save_checkpoint(path: str, model: nn.Module, cfg: dict, extra: dict | None = None, half: bool = True):
    sd = {k: (v.half() if half and v.is_floating_point() else v) for k, v in model.state_dict().items()}
    torch.save({"model": sd, "cfg": cfg, **(extra or {})}, path)


def load_checkpoint(path: str, device: str = "cpu") -> tuple[FilaNet, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ck["cfg"], pretrained=False)
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ck["model"].items()})
    return model.eval().to(device), ck
