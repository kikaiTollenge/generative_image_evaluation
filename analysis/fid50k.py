#!/usr/bin/env python3
"""Compute FID between real and generated image directories.

For manuscript-scale evaluation, use InceptionV3 features with 50,000 generated
images. For offline smoke tests, use ``--feature-extractor color`` or omit
``--pretrained`` to avoid downloading pretrained weights.
"""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
from scipy import linalg

EXTS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def image_paths(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in EXTS)


def color_features(path: Path, size: int) -> np.ndarray:
    arr = np.asarray(Image.open(path).convert("RGB").resize((size, size)), dtype=np.float32) / 255.0
    pixels = arr.reshape(-1, 3)
    quantiles = np.quantile(pixels, [0.1, 0.5, 0.9], axis=0).reshape(-1)
    return np.concatenate([pixels.mean(axis=0), pixels.std(axis=0), quantiles])


def load_color_features(root: Path, size: int, limit: Optional[int]) -> np.ndarray:
    paths = image_paths(root)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"No images found under {root}")
    return np.stack([color_features(path, size) for path in paths], axis=0)


def build_inception(pretrained: bool, device: str):
    import torch
    import torch.nn as nn
    import torchvision.models as models
    try:
        from torchvision.models import Inception_V3_Weights
        weights = Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.inception_v3(weights=weights, aux_logits=True, init_weights=False)
    except Exception:
        model = models.inception_v3(pretrained=pretrained, aux_logits=True, init_weights=False)
    model.fc = nn.Identity()
    model.AuxLogits = None
    model.aux_logits = False
    model.eval().to(device)
    return model


def load_inception_features(root: Path, image_size: int, limit: Optional[int], batch_size: int, device: str, pretrained: bool) -> np.ndarray:
    import torch
    from torchvision import transforms
    paths = image_paths(root)
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise SystemExit(f"No images found under {root}")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    model = build_inception(pretrained, device)
    feats = []
    with torch.no_grad():
        for start in range(0, len(paths), batch_size):
            images = [transform(Image.open(path).convert("RGB")) for path in paths[start:start + batch_size]]
            batch = torch.stack(images).to(device)
            out = model(batch)
            feats.append(out.detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def stable_sqrtm(matrix: np.ndarray) -> np.ndarray:
    try:
        root, _ = linalg.sqrtm(matrix, disp=False)
        return root
    except TypeError:
        return linalg.sqrtm(matrix)


def fid_from_features(a: np.ndarray, b: np.ndarray) -> float:
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    cov_a = np.atleast_2d(np.cov(a, rowvar=False))
    cov_b = np.atleast_2d(np.cov(b, rowvar=False))
    eps = 1e-6
    covmean = stable_sqrtm(cov_a.dot(cov_b))
    if not np.isfinite(covmean).all():
        eye = np.eye(cov_a.shape[0]) * eps
        covmean = stable_sqrtm((cov_a + eye).dot(cov_b + eye))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    if not np.isfinite(covmean).all():
        covmean = np.diag(np.sqrt(np.maximum(np.diag(cov_a), 0.0) * np.maximum(np.diag(cov_b), 0.0)))
    value = np.sum((mu_a - mu_b) ** 2) + np.trace(cov_a + cov_b - 2.0 * covmean)
    return float(max(value, 0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--image-size", type=int, default=299)
    parser.add_argument("--limit", type=int, default=None, help="Optional image limit for smoke tests")
    parser.add_argument("--feature-extractor", choices=["inception", "color"], default="inception")
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet-pretrained InceptionV3 weights")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.feature_extractor == "color":
        real = load_color_features(Path(args.real_dir), args.image_size, args.limit)
        generated = load_color_features(Path(args.generated_dir), args.image_size, args.limit)
        metric = "fid_color_stat"
    else:
        real = load_inception_features(Path(args.real_dir), args.image_size, args.limit, args.batch_size, args.device, args.pretrained)
        generated = load_inception_features(Path(args.generated_dir), args.image_size, args.limit, args.batch_size, args.device, args.pretrained)
        metric = "fid_inception_v3"
    row = {"metric": metric, "real_images": real.shape[0], "generated_images": generated.shape[0], "image_size": args.image_size, "fid": fid_from_features(real, generated)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)


if __name__ == "__main__":
    main()
