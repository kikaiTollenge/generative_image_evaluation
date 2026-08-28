#!/usr/bin/env python3
"""Run manuscript-defined semantic segmentation experiments with CSV output."""
from __future__ import annotations
import argparse, csv, random
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from Dataset.pathology_datasets import ImageMaskDataset, RandomSegmentationDataset
from Model.pathology_encoders import FeaturePyramidUNet, build_encoder

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def dice_iou(logits: torch.Tensor, target: torch.Tensor):
    pred = (torch.sigmoid(logits) > 0.5).float(); target = (target > 0.5).float(); dims = tuple(range(1, pred.ndim))
    inter = (pred * target).sum(dim=dims); pred_sum = pred.sum(dim=dims); target_sum = target.sum(dim=dims); union = pred_sum + target_sum - inter
    dice = ((2 * inter + 1e-7) / (pred_sum + target_sum + 1e-7)).mean().item(); iou = ((inter + 1e-7) / (union + 1e-7)).mean().item()
    return float(dice), float(iou)

def evaluate(model, loader, device):
    model.eval(); loss_fn = nn.BCEWithLogitsLoss(); losses=[]; dices=[]; ious=[]
    with torch.no_grad():
        for image, mask in loader:
            image, mask = image.to(device), mask.to(device); logits = model(image)
            losses.append(loss_fn(logits, mask).item()); d, i = dice_iou(logits, mask); dices.append(d); ious.append(i)
    return float(np.mean(losses)), float(np.mean(dices)), float(np.mean(ious))

def dataset_from_args(args, seed):
    if args.dataset in {"warwick_qu", "monuseg"}:
        return (ImageMaskDataset(args.train_images, args.train_masks, args.image_size, True, args.mask_suffix), ImageMaskDataset(args.test_images, args.test_masks, args.image_size, False, args.mask_suffix))
    if args.dataset == "random":
        ds = RandomSegmentationDataset(args.random_samples, args.image_size, seed); n_train = max(1, int(0.7 * len(ds))); n_test = len(ds) - n_train
        return random_split(ds, [n_train, n_test], generator=torch.Generator().manual_seed(seed))
    raise ValueError(args.dataset)

def train_once(args, seed):
    set_seed(seed); device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    train_ds, test_ds = dataset_from_args(args, seed)
    model = FeaturePyramidUNet(build_encoder(args.method, args.checkpoint), out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay); loss_fn = nn.BCEWithLogitsLoss()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    for _ in range(args.epochs):
        model.train()
        for image, mask in train_loader:
            image, mask = image.to(device), mask.to(device); optimizer.zero_grad(); loss = loss_fn(model(image), mask); loss.backward(); optimizer.step()
    loss, dice, iou = evaluate(model, test_loader, device)
    return {"seed": seed, "method": args.method, "dataset": args.dataset, "loss": loss, "dice": dice, "iou": iou}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["warwick_qu", "monuseg", "random"], required=True)
    parser.add_argument("--method", choices=["synthetic", "real", "without-pretrain", "retccl", "ctranspath", "resnet34"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--train-images"); parser.add_argument("--train-masks"); parser.add_argument("--test-images"); parser.add_argument("--test-masks")
    parser.add_argument("--mask-suffix", default="")
    parser.add_argument("--image-size", type=int, default=256); parser.add_argument("--epochs", type=int, default=50); parser.add_argument("--repeat", type=int, default=100); parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16); parser.add_argument("--workers", type=int, default=0); parser.add_argument("--lr", type=float, default=1e-4); parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default=None); parser.add_argument("--random-samples", type=int, default=8); parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.dataset != "random" and any(v is None for v in [args.train_images, args.train_masks, args.test_images, args.test_masks]):
        raise SystemExit("Real segmentation datasets require --train-images --train-masks --test-images --test-masks")
    rows = [train_once(args, args.seed + i) for i in range(args.repeat)]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "method", "dataset", "loss", "dice", "iou"]); writer.writeheader(); writer.writerows(rows)
if __name__ == "__main__":
    main()
