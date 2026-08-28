"""Dataset utilities for the manuscript-defined public experiments."""
from __future__ import annotations
import csv, random
from pathlib import Path
from typing import Callable, Dict, List, Optional
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
try:
    import torchvision.transforms as T
except Exception:
    T = None
IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
WARWICK_GRADE_TO_BINARY = {"benign": 0, "malignant": 1}
WARWICK_SIRINUKUN_TO_FIVE_CLASS = {"adenomatous": 0, "healthy": 1, "poorly differentiated": 2, "moderately differentiated": 3, "moderately-to-poorly differentated": 4, "moderately-to-poorly differentiated": 4}
BREAKHIS_CLASSES = ["adenosis", "fibroadenoma", "phyllodes_tumor", "tubular_adenoma", "ductal_carcinoma", "lobular_carcinoma", "mucinous_carcinoma", "papillary_carcinoma"]

def _default_transform(image_size: int, train: bool) -> Callable[[Image.Image], torch.Tensor]:
    if T is None:
        raise RuntimeError("torchvision is required for image transforms")
    ops = [T.Resize((image_size, image_size))]
    if train:
        ops.extend([T.RandomHorizontalFlip(), T.RandomVerticalFlip()])
    ops.extend([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return T.Compose(ops)

def _mask_transform(image_size: int):
    def transform(mask: Image.Image) -> torch.Tensor:
        mask = mask.resize((image_size, image_size), resample=Image.NEAREST)
        arr = (np.asarray(mask.convert("L"), dtype=np.float32) > 0).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)
    return transform

def list_images(root) -> List[Path]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)

def _read_csv(path) -> List[Dict[str, str]]:
    with open(path, newline="") as handle:
        return [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in row.items()} for row in csv.DictReader(handle)]

def _resolve_path(raw: str, root: Optional[Path]) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else ((root / path) if root is not None else path)

class WarwickGlaSClassificationDataset(Dataset):
    """Warwick-QU/GlaS classification loader; legacy code name: glas."""
    def __init__(self, csv_path, split: str, binary: bool = True, image_root=None, image_size: int = 224):
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be train, test, or all")
        self.csv_path = Path(csv_path)
        self.image_root = Path(image_root) if image_root else None
        self.transform = _default_transform(image_size, train=(split == "train"))
        self.samples = []
        for row in _read_csv(self.csv_path):
            name = row.get("name", "")
            if split != "all" and split not in name:
                continue
            if binary:
                raw = row.get("grade (GlaS)") or row.get(" grade (GlaS)")
                label = int(raw) if str(raw).isdigit() else WARWICK_GRADE_TO_BINARY[str(raw).strip().lower()]
            else:
                raw = row.get("grade (Sirinukunwattana et al. 2015)") or row.get(" grade (Sirinukunwattana et al. 2015)")
                label = int(raw) if str(raw).isdigit() else WARWICK_SIRINUKUN_TO_FIVE_CLASS[str(raw).strip().lower()]
            self.samples.append((_resolve_path(name, self.image_root), label))
        if not self.samples:
            raise ValueError(f"No Warwick-QU/GlaS samples found for split={split}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert("RGB")), torch.tensor(label, dtype=torch.long)

class ImageMaskDataset(Dataset):
    """Generic image/mask dataset used for Warwick-QU and MoNuSeg segmentation."""
    def __init__(self, image_dir, mask_dir, image_size: int = 256, train: bool = True, mask_suffix: str = ""):
        self.image_transform = _default_transform(image_size, train=train)
        self.mask_transform = _mask_transform(image_size)
        self.samples = self._match_pairs(Path(image_dir), Path(mask_dir), mask_suffix)
        if not self.samples:
            raise ValueError(f"No image/mask pairs found in {image_dir} and {mask_dir}")
    @staticmethod
    def _stem_key(path: Path, mask_suffix: str) -> str:
        stem = path.stem
        if mask_suffix and stem.endswith(mask_suffix):
            stem = stem[:-len(mask_suffix)]
        if stem.endswith("_anno"):
            stem = stem[:-5]
        return stem
    def _match_pairs(self, image_dir: Path, mask_dir: Path, mask_suffix: str):
        masks = {self._stem_key(p, mask_suffix): p for p in list_images(mask_dir)}
        pairs = []
        for image in list_images(image_dir):
            mask = masks.get(self._stem_key(image, ""))
            if mask is not None:
                pairs.append((image, mask))
        return pairs
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        return self.image_transform(Image.open(image_path).convert("RGB")), self.mask_transform(Image.open(mask_path).convert("L"))

class BreakHisClassificationDataset(Dataset):
    """BreakHis 40X classification loader.

    Use split_csv for the exact fixed split. Without it, a deterministic balanced
    7:3 split is reconstructed from the directory tree for testing/reuse.
    """
    def __init__(self, root, split: str, split_csv=None, magnification: str = "40X", image_size: int = 224, seed: int = 0):
        if split not in {"train", "test", "all"}:
            raise ValueError("split must be train, test, or all")
        self.root = Path(root)
        self.transform = _default_transform(image_size, train=(split == "train"))
        self.samples = self._from_split_csv(Path(split_csv), split) if split_csv else self._from_directory(split, magnification, seed)
        if not self.samples:
            raise ValueError(f"No BreakHis samples found for split={split} under {root}")
    def _class_from_path(self, path: Path):
        normalized = str(path).lower().replace("-", "_").replace(" ", "_")
        for idx, cls in enumerate(BREAKHIS_CLASSES):
            if cls in normalized:
                return idx
        return None
    def _from_split_csv(self, csv_path: Path, split: str):
        samples = []
        for row in _read_csv(csv_path):
            if split != "all" and row.get("split", "all").lower() != split:
                continue
            image = row.get("path") or row.get("image") or row.get("filename")
            label = row.get("label") or row.get("class")
            if image is None or label is None:
                raise ValueError("BreakHis split CSV needs image/path and label/class columns")
            label_id = int(label) if str(label).isdigit() else BREAKHIS_CLASSES.index(str(label).strip().lower().replace(" ", "_"))
            samples.append((_resolve_path(image, self.root), label_id))
        return samples
    def _from_directory(self, split: str, magnification: str, seed: int):
        by_class = {i: [] for i in range(len(BREAKHIS_CLASSES))}
        for path in list_images(self.root):
            if magnification and magnification.lower() not in str(path).lower():
                continue
            label = self._class_from_path(path)
            if label is not None:
                by_class[label].append(path)
        rng = random.Random(seed)
        non_empty = [v for v in by_class.values() if v]
        if not non_empty:
            return []
        per_class = min(len(v) for v in non_empty)
        selected = []
        for label, paths in by_class.items():
            if not paths:
                continue
            paths = sorted(paths)
            rng.shuffle(paths)
            paths = paths[:per_class]
            cut = int(round(0.7 * len(paths)))
            chosen = paths[:cut] if split == "train" else paths[cut:] if split == "test" else paths
            selected.extend((p, label) for p in chosen)
        rng.shuffle(selected)
        return selected
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        path, label = self.samples[index]
        return self.transform(Image.open(path).convert("RGB")), torch.tensor(label, dtype=torch.long)

class RandomClassificationDataset(Dataset):
    def __init__(self, samples: int, classes: int, image_size: int = 224, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.images = rng.integers(0, 256, size=(samples, image_size, image_size, 3), dtype=np.uint8)
        self.labels = rng.integers(0, classes, size=(samples,), dtype=np.int64)
        self.transform = _default_transform(image_size, True)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index])), torch.tensor(int(self.labels[index]), dtype=torch.long)

class RandomSegmentationDataset(Dataset):
    def __init__(self, samples: int, image_size: int = 256, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.images = rng.integers(0, 256, size=(samples, image_size, image_size, 3), dtype=np.uint8)
        self.masks = rng.integers(0, 2, size=(samples, image_size, image_size), dtype=np.uint8)
        self.image_transform = _default_transform(image_size, True)
        self.mask_transform = _mask_transform(image_size)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, index):
        return self.image_transform(Image.fromarray(self.images[index])), self.mask_transform(Image.fromarray(self.masks[index] * 255))
