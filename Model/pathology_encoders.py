"""Encoder builders for Synthetic, Real, w/o Pre-train, RetCCL, and CTransPath."""
from __future__ import annotations
from typing import Optional, Sequence, List
import torch
import torch.nn as nn
import torchvision.models as models

def _resnet(name: str) -> nn.Module:
    constructor = getattr(models, name)
    try:
        return constructor(weights=None)
    except TypeError:
        return constructor(pretrained=False)

def _strip_prefix(key: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in ("module.", "encoder_q.", "feature_extractor.", "backbone.", "model."):
            if key.startswith(prefix):
                key = key[len(prefix):]
                changed = True
    return key

def load_checkpoint_flex(model: nn.Module, checkpoint_path: Optional[str], strict: bool = False) -> nn.Module:
    if not checkpoint_path:
        return model
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    own = model.state_dict()
    cleaned = {}
    for key, value in state.items():
        clean = _strip_prefix(key)
        candidates = [clean]
        if clean.startswith("resnet."):
            candidates.append(clean[len("resnet."):])
        for candidate in candidates:
            if candidate in own and own[candidate].shape == value.shape:
                cleaned[candidate] = value
                break
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return model

class ResNetFeatureEncoder(nn.Module):
    def __init__(self, architecture: str = "resnet34", checkpoint_path: Optional[str] = None):
        super().__init__()
        model = _resnet(architecture)
        load_checkpoint_flex(model, checkpoint_path)
        self.stem = nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = model.layer1, model.layer2, model.layer3, model.layer4
        self.out_channels = [64, 128, 256, 512] if architecture in {"resnet18", "resnet34"} else [256, 512, 1024, 2048]
        self.classifier_channels = self.out_channels[-1]
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        return [f1, f2, f3, f4]
    def forward(self, x):
        return self.forward_features(x)[-1]

class TimmFeatureEncoder(nn.Module):
    def __init__(self, model_name: str, checkpoint_path: Optional[str] = None):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("CTransPath support requires timm.") from exc
        self.native_features = True
        try:
            self.model = timm.create_model(model_name, pretrained=False, features_only=True, out_indices=(0, 1, 2, 3))
            self.out_channels = list(self.model.feature_info.channels())
        except Exception:
            self.native_features = False
            self.model = timm.create_model(model_name, pretrained=False, num_classes=0, global_pool="")
            self.out_channels = [getattr(self.model, "num_features", 768)] * 4
        load_checkpoint_flex(self.model, checkpoint_path)
        self.classifier_channels = self.out_channels[-1]

    def _channels_first(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.ndim == 3:
            tokens = feat.shape[1]
            side = int(tokens ** 0.5)
            if side * side != tokens:
                feat = feat[:, 1:, :]
                side = int(feat.shape[1] ** 0.5)
            feat = feat.transpose(1, 2).reshape(feat.shape[0], feat.shape[-1], side, side)
        if feat.ndim == 4 and feat.shape[1] not in self.out_channels and feat.shape[-1] in self.out_channels:
            feat = feat.permute(0, 3, 1, 2).contiguous()
        return feat

    def forward_features(self, x):
        if self.native_features:
            return [self._channels_first(feat) for feat in self.model(x)]
        feat = self._channels_first(self.model.forward_features(x))
        sizes = [max(1, feat.shape[-1] * scale) for scale in (8, 4, 2, 1)]
        return [torch.nn.functional.interpolate(feat, size=(s, s), mode="bilinear", align_corners=False) for s in sizes]

    def forward(self, x):
        return self.forward_features(x)[-1]

def build_encoder(name: str, checkpoint_path: Optional[str] = None) -> nn.Module:
    normalized = name.lower().replace("_", "-")
    if normalized in {"synthetic", "real", "resnet34"}:
        return ResNetFeatureEncoder("resnet34", checkpoint_path)
    if normalized in {"without-pretrain", "wo-pretrain", "w/o-pretrain"}:
        return ResNetFeatureEncoder("resnet34", None)
    if normalized == "retccl":
        return ResNetFeatureEncoder("resnet50", checkpoint_path)
    if normalized == "ctranspath":
        return TimmFeatureEncoder("swin_tiny_patch4_window7_224", checkpoint_path)
    raise ValueError(f"Unsupported encoder: {name}")

class EncoderClassifier(nn.Module):
    def __init__(self, encoder: nn.Module, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.encoder = encoder
        channels = getattr(encoder, "classifier_channels", getattr(encoder, "out_channels", [512])[-1])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels, num_classes))
    def forward(self, x):
        return self.head(self.pool(self.encoder(x)))

class FeaturePyramidUNet(nn.Module):
    def __init__(self, encoder: nn.Module, out_channels: int = 1, decoder_channels: Sequence[int] = (256, 128, 64)):
        super().__init__()
        self.encoder = encoder
        enc = list(getattr(encoder, "out_channels"))
        self.proj = nn.ModuleList([nn.Conv2d(c, c, 1) for c in enc])
        blocks = []
        in_ch = enc[-1]
        for skip_ch, dec_ch in zip(reversed(enc[:-1]), decoder_channels):
            blocks.append(nn.Sequential(
                nn.Conv2d(in_ch + skip_ch, dec_ch, 3, padding=1),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(dec_ch, dec_ch, 3, padding=1),
                nn.BatchNorm2d(dec_ch),
                nn.ReLU(inplace=True),
            ))
            in_ch = dec_ch
        self.blocks = nn.ModuleList(blocks)
        self.final = nn.Conv2d(in_ch, out_channels, 1)
    def forward(self, x):
        size = x.shape[-2:]
        feats = [p(f) for p, f in zip(self.proj, self.encoder.forward_features(x))]
        y = feats[-1]
        for block, skip in zip(self.blocks, reversed(feats[:-1])):
            y = torch.nn.functional.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = block(torch.cat([y, skip], 1))
        return torch.nn.functional.interpolate(self.final(y), size=size, mode="bilinear", align_corners=False)
