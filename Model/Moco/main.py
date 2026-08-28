#!/usr/bin/env python3
"""Inspect MoCo checkpoints against the legacy ResNet34 backbone."""
from __future__ import annotations
import argparse
import torch
import torch.nn as nn
import torchvision.models as models

try:
    from torchvision.models import ResNet34_Weights
except Exception:
    ResNet34_Weights = None

def get_resnet34():
    if ResNet34_Weights is not None:
        model = models.resnet34(weights=None)
    else:
        model = models.resnet34(pretrained=False)
    model.fc = nn.Identity(); model.avgpool = nn.Identity(); return model

class moco_backbone(nn.Module):
    def __init__(self):
        super().__init__(); self.resnet = get_resnet34()
        self.layer0 = nn.Sequential(self.resnet.conv1, self.resnet.bn1, self.resnet.relu, self.resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = self.resnet.layer1, self.resnet.layer2, self.resnet.layer3, self.resnet.layer4
        self.bottleneck = nn.Sequential(nn.Conv2d(512, 1024, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(1024), nn.Conv2d(1024, 1024, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(1024), nn.MaxPool2d(2, 2))
    def forward(self, x):
        x = self.layer0(x); x = self.layer1(x); x = self.layer2(x); x = self.layer3(x); x = self.layer4(x); return self.bottleneck(x)

def extract_backbone_state(state_dict):
    out = {}
    for key, value in state_dict.items():
        clean_key = key[len('module.'):] if key.startswith('module.') else key
        for prefix in ['encoder_q.feature_extractor.resnet', 'encoder_q.feature_extractor.layer', 'encoder_q.feature_extractor.bottleneck']:
            if prefix in clean_key:
                out[clean_key.split('encoder_q.feature_extractor.')[1]] = value
                break
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--strict', action='store_true')
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    state_dict = checkpoint.get('state_dict', checkpoint)
    backbone = moco_backbone()
    extracted = extract_backbone_state(state_dict)
    missing, unexpected = backbone.load_state_dict(extracted, strict=False)
    print(f'extracted_keys={len(extracted)} missing={len(missing)} unexpected={len(unexpected)}')
    if args.strict and (missing or unexpected):
        raise SystemExit(1)
if __name__ == '__main__':
    main()
