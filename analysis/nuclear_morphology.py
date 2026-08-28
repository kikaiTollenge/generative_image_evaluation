#!/usr/bin/env python3
"""Compute nuclear morphology from instance masks; HoVer-Net inference is external."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
from statistics import median
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
try:
    import cv2
except ImportError:
    cv2 = None
FEATURES = ["median_nuclear_area", "mean_circularity", "nuclear_density", "median_nn_distance", "nuclear_area_cv"]

def cliff_delta(x, y):
    x = np.asarray(x); y = np.asarray(y)
    return (sum(float((xi > y).sum()) for xi in x) - sum(float((xi < y).sum()) for xi in x)) / float(len(x) * len(y))

def region_perimeter(binary: np.ndarray) -> float:
    if cv2 is not None:
        contours, _ = cv2.findContours(binary.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return float(sum(cv2.arcLength(c, True) for c in contours))
    padded = np.pad(binary.astype(bool), 1); center = padded[1:-1, 1:-1]
    edge = center & (~padded[:-2, 1:-1] | ~padded[2:, 1:-1] | ~padded[1:-1, :-2] | ~padded[1:-1, 2:])
    return float(edge.sum())

def load_instance_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    arr = np.asarray(Image.open(path))
    return arr[..., 0] if arr.ndim == 3 else arr


def features_from_mask(path: Path, group: str, microns_per_pixel=None):
    arr = load_instance_array(path)
    ids = [int(v) for v in np.unique(arr) if int(v) != 0]; areas = []; circularities = []; centroids = []
    for obj_id in ids:
        binary = arr == obj_id; area_px = float(binary.sum())
        if area_px <= 0: continue
        ys, xs = np.nonzero(binary); perim = region_perimeter(binary); circularity = 4.0 * np.pi * area_px / (perim * perim) if perim > 0 else 0.0
        areas.append(area_px); circularities.append(circularity); centroids.append((float(xs.mean()), float(ys.mean())))
    image_area_px = float(arr.shape[0] * arr.shape[1])
    area_scale = microns_per_pixel ** 2 if microns_per_pixel else 1.0; dist_scale = microns_per_pixel if microns_per_pixel else 1.0
    unit_area = "um2" if microns_per_pixel else "pixel2"; unit_dist = "um" if microns_per_pixel else "pixel"
    if len(centroids) > 1:
        distances, _ = cKDTree(np.asarray(centroids)).query(np.asarray(centroids), k=2); median_nn = float(np.median(distances[:, 1]) * dist_scale)
    else:
        median_nn = float("nan")
    areas = np.asarray(areas, dtype=float)
    return {"image": path.name, "group": group, "n_nuclei": len(areas), "area_unit": unit_area, "distance_unit": unit_dist, "median_nuclear_area": float(np.median(areas) * area_scale) if len(areas) else float("nan"), "mean_circularity": float(np.mean(circularities)) if circularities else float("nan"), "nuclear_density": float(len(areas) / (image_area_px * area_scale)) if image_area_px else float("nan"), "median_nn_distance": median_nn, "nuclear_area_cv": float(np.std(areas, ddof=1) / np.mean(areas)) if len(areas) > 1 and np.mean(areas) else float("nan")}

def summarize(rows):
    out = []; groups = sorted({r["group"] for r in rows})
    if len(groups) != 2: return out
    for feat in FEATURES:
        x = [float(r[feat]) for r in rows if r["group"] == groups[0] and np.isfinite(float(r[feat]))]; y = [float(r[feat]) for r in rows if r["group"] == groups[1] and np.isfinite(float(r[feat]))]
        if not x or not y: continue
        q1x, q3x = np.percentile(x, [25, 75]); q1y, q3y = np.percentile(y, [25, 75]); stat = mannwhitneyu(x, y, alternative="two-sided")
        out.append({"feature": feat, "first_group": groups[0], "second_group": groups[1], "median_first": median(x), "iqr_first": f"{q1x:.6g}-{q3x:.6g}", "median_second": median(y), "iqr_second": f"{q1y:.6g}-{q3y:.6g}", "mann_whitney_u": float(stat.statistic), "p_value": float(stat.pvalue), "cliffs_delta": cliff_delta(x, y)})
    return out

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-mask-dir"); parser.add_argument("--synthetic-mask-dir"); parser.add_argument("--microns-per-pixel", type=float, default=None); parser.add_argument("--features-output", required=True); parser.add_argument("--summary-output", required=True)
    args = parser.parse_args(); rows = []
    for group, folder in [("real", args.real_mask_dir), ("synthetic", args.synthetic_mask_dir)]:
        if not folder: continue
        for path in sorted(Path(folder).glob("*")):
            if path.suffix.lower() in {".npy", ".png", ".tif", ".tiff", ".bmp"}: rows.append(features_from_mask(path, group, args.microns_per_pixel))
    if not rows: raise SystemExit("No instance masks found. Provide --real-mask-dir and/or --synthetic-mask-dir.")
    fp = Path(args.features_output); fp.parent.mkdir(parents=True, exist_ok=True)
    with fp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)
    summary = summarize(rows); sp = Path(args.summary_output); sp.parent.mkdir(parents=True, exist_ok=True)
    with sp.open("w", newline="") as handle:
        fields = list(summary[0].keys()) if summary else ["feature"]; writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(summary)
if __name__ == "__main__": main()
