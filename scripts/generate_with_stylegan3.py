#!/usr/bin/env python3
"""Thin wrapper around the official NVIDIA StyleGAN3 gen_images.py."""
from __future__ import annotations
import argparse, subprocess, sys
from pathlib import Path
import numpy as np
from PIL import Image


def parse_seeds(seeds: str) -> list[int]:
    parsed = []
    for item in seeds.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            start, end = item.split("-", 1)
            parsed.extend(range(int(start), int(end) + 1))
        else:
            parsed.append(int(item))
    if not parsed:
        raise ValueError("No seeds were provided")
    return parsed


def write_smoke_image(outdir: Path, seeds: str, image_size: int) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for seed in parse_seeds(seeds):
        rng = np.random.default_rng(seed)
        image = rng.integers(0, 256, size=(image_size, image_size, 3), dtype=np.uint8)
        Image.fromarray(image).save(outdir / f"seed{seed:04d}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stylegan3-repo", default=None, help="Path to official NVIDIA/stylegan3 repository")
    parser.add_argument("--network", default=None, help="Path or URL to trained StyleGAN3 generator .pkl")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--seeds", required=True, help="e.g. 1-70000 or 0,1,2")
    parser.add_argument("--trunc", default="1")
    parser.add_argument("--smoke-test", action="store_true", help="Write one random RGB image to verify output plumbing without generator weights")
    parser.add_argument("--image-size", type=int, default=1024, help="Smoke-test image size")
    parser.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args forwarded to gen_images.py")
    args = parser.parse_args()
    if args.smoke_test:
        write_smoke_image(Path(args.outdir), args.seeds, args.image_size)
        return
    if not args.stylegan3_repo:
        raise SystemExit("--stylegan3-repo is required unless --smoke-test is used")
    if not args.network:
        raise SystemExit("--network is required unless --smoke-test is used")
    gen = Path(args.stylegan3_repo) / "gen_images.py"
    if not gen.exists():
        raise SystemExit(f"Could not find official StyleGAN3 gen_images.py at {gen}")
    cmd = [sys.executable, str(gen), "--outdir", args.outdir, "--seeds", args.seeds, "--network", args.network, "--trunc", str(args.trunc)]
    if args.extra:
        cmd.extend(args.extra)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
