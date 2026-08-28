#!/usr/bin/env python3
"""Create tiny random datasets for smoke tests only."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from PIL import Image

def save_rgb(path: Path, shape, rng):
    path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray(rng.integers(0, 256, size=(shape[1], shape[0], 3), dtype=np.uint8)).save(path)
def save_mask(path: Path, shape, rng):
    path.parent.mkdir(parents=True, exist_ok=True); Image.fromarray((rng.integers(0, 2, size=(shape[1], shape[0]), dtype=np.uint8) * 255)).save(path)
def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--root", required=True); parser.add_argument("--samples", type=int, default=4); parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(); root = Path(args.root); rng = np.random.default_rng(args.seed)
    for split in ["train", "test"]:
        for idx in range(args.samples):
            save_rgb(root / "warwick_qu" / split / "image" / f"{split}_{idx}.bmp", (775, 522), rng); save_mask(root / "warwick_qu" / split / "mask" / f"{split}_{idx}_anno.bmp", (775, 522), rng)
            save_rgb(root / "monuseg" / split / "images" / f"monuseg_{split}_{idx}.png", (1000, 1000), rng); save_mask(root / "monuseg" / split / "masks" / f"monuseg_{split}_{idx}.png", (1000, 1000), rng)
    classes = ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma", "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]
    for cls in classes:
        for idx in range(args.samples): save_rgb(root / "breakhis" / cls / "40X" / f"{cls}_{idx}.png", (700, 460), rng)
    print(root)
if __name__ == "__main__": main()
