#!/usr/bin/env python3
"""Run manuscript-defined classification experiments with CSV output."""
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
from Dataset.pathology_datasets import BreakHisClassificationDataset, RandomClassificationDataset, WarwickGlaSClassificationDataset
from Model.pathology_encoders import EncoderClassifier, build_encoder

def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def macro_f1(y_true, y_pred, classes: int) -> float:
    scores = []
    for cls in range(classes):
        tp = sum((t == cls and p == cls) for t, p in zip(y_true, y_pred))
        fp = sum((t != cls and p == cls) for t, p in zip(y_true, y_pred))
        fn = sum((t == cls and p != cls) for t, p in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    return float(np.mean(scores))

def evaluate(model, loader, device, classes: int):
    model.eval(); y_true, y_pred = [], []; loss_fn = nn.CrossEntropyLoss(); total_loss = 0.0
    with torch.no_grad():
        for image, label in loader:
            image, label = image.to(device), label.to(device)
            logits = model(image); total_loss += loss_fn(logits, label).item(); pred = logits.argmax(dim=1)
            y_true.extend(label.cpu().tolist()); y_pred.extend(pred.cpu().tolist())
    accuracy = float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if y_true else 0.0
    return total_loss / max(1, len(loader)), accuracy, macro_f1(y_true, y_pred, classes)

def train_once(args, seed: int):
    set_seed(seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    if args.dataset == "warwick_qu":
        train_ds = WarwickGlaSClassificationDataset(args.csv, "train", binary=args.binary, image_root=args.data_root, image_size=args.image_size)
        test_ds = WarwickGlaSClassificationDataset(args.csv, "test", binary=args.binary, image_root=args.data_root, image_size=args.image_size)
        classes = 2 if args.binary else 5
    elif args.dataset == "breakhis":
        train_ds = BreakHisClassificationDataset(args.data_root, "train", args.split_csv, args.magnification, args.image_size, seed)
        test_ds = BreakHisClassificationDataset(args.data_root, "test", args.split_csv, args.magnification, args.image_size, seed)
        classes = 8
    elif args.dataset == "random":
        ds = RandomClassificationDataset(args.random_samples, args.num_classes, args.image_size, seed)
        n_train = max(1, int(0.7 * len(ds))); n_test = len(ds) - n_train
        train_ds, test_ds = random_split(ds, [n_train, n_test], generator=torch.Generator().manual_seed(seed)); classes = args.num_classes
    else:
        raise ValueError(args.dataset)
    model = EncoderClassifier(build_encoder(args.method, args.checkpoint), classes, args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay); loss_fn = nn.CrossEntropyLoss()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    for _ in range(args.epochs):
        model.train()
        for image, label in train_loader:
            image, label = image.to(device), label.to(device); optimizer.zero_grad(); loss = loss_fn(model(image), label); loss.backward(); optimizer.step()
    loss, accuracy, f1 = evaluate(model, test_loader, device, classes)
    return {"seed": seed, "method": args.method, "dataset": args.dataset, "loss": loss, "accuracy": accuracy, "f1": f1}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["warwick_qu", "breakhis", "random"], required=True)
    parser.add_argument("--method", choices=["synthetic", "real", "without-pretrain", "retccl", "ctranspath", "resnet34"], required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--csv", default="Dataset/glas/Grade_modified.csv")
    parser.add_argument("--split-csv", default=None, help="Fixed BreakHis split CSV with image/path,label,split columns")
    parser.add_argument("--magnification", default="40X")
    parser.add_argument("--binary", action="store_true", help="Use Warwick-QU/GlaS benign-vs-malignant labels")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--device", default=None)
    parser.add_argument("--random-samples", type=int, default=12)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [train_once(args, args.seed + i) for i in range(args.repeat)]
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "method", "dataset", "loss", "accuracy", "f1"]); writer.writeheader(); writer.writerows(rows)
if __name__ == "__main__":
    main()
