# Dataset Inventory

| Dataset | Manuscript task | Existing code | Loader | Split recovered | Metadata | README | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Warwick-QU/GlaS | classification | `Dataset/glas/Grade.csv`, `Grade_modified.csv`, legacy `glas_classify_Dataset` | `WarwickGlaSClassificationDataset` | yes, via train/test filename convention in metadata | yes | yes | COMPLETE except images externalized |
| Warwick-QU/GlaS | semantic segmentation | legacy `PathologyData` image/mask pairing | `ImageMaskDataset` | yes, via train/test dirs | masks externalized | yes | COMPLETE except images/masks externalized |
| BreakHis 40X | classification | no history found for BreakHis code | `BreakHisClassificationDataset` | exact split not recovered; fixed CSV supported; deterministic balanced 7:3 available | external or user-provided CSV | yes | PARTIAL: MISSING exact split metadata |
| MoNuSeg | semantic segmentation | no history found for MoNuSeg code | `ImageMaskDataset` | official split expected from dataset layout | external | yes | PARTIAL: data external |
| Synthetic CRC | MoCo pre-training/generation | legacy README command, MoCo code | image-folder input to `Model/Moco/main_moco.py` | seed range 1-70000 from manuscript | external image folder | yes | PARTIAL: dataset external |
| Real CRC | MoCo pre-training/morphology | legacy `Dataset/true_image` examples were removed from Git | image-folder input to `Model/Moco/main_moco.py` | manuscript says 70,000 for pre-training and 1,500 morphology sample | external image folder/masks | yes | PARTIAL: dataset external |
