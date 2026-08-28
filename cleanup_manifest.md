# Cleanup Manifest

| Path | Action | Reason | Manuscript relevance | Replacement |
| --- | --- | --- | --- | --- |
| `.idea/` | deleted from Git | IDE metadata | unrelated | `.gitignore` |
| `__pycache__/`, `*/__pycache__/` | deleted from Git | generated Python cache | unrelated | regenerated locally |
| `output.log`, `Model/output.log`, `Model/Moco/output.log` | deleted from Git | historical runtime logs with local paths | unrelated | machine-readable CSV outputs |
| `try_param`, `best_epochs.txt` | deleted from Git | ad hoc hyperparameter notes/results | not a stable manuscript artifact | README and CSV outputs |
| `image.png`, `label.png`, `output.png`, `output.png` | deleted from Git | transient visualization outputs | not manuscript data | regenerate from scripts if needed |
| `Dataset/glas/train/`, `Dataset/glas/test/` images/masks | deleted from Git | public repository should not ship full dataset images/masks | Warwick-QU/GlaS remains required | metadata and path-based loaders |
| `Dataset/true_image/` | deleted from Git | example/full image data should not be tracked | not the manuscript CRC pre-training corpus | external data directory |
| `weight/` | deleted from Git | model outputs/curves/checkpoint area | weights are external artifacts | checkpoint paths via CLI |
| `temp.py` | deleted from Git | one-off utility | unrelated | none |
| `main.py` | deleted from Git | one-off GlaS metadata converter with misleading name | relevant only as metadata preparation | `Dataset/prepare_glas_metadata.py` |

Preserved after audit: `Model/Moco/` for legacy MoCo pre-training, `Model/Unet/` for legacy segmentation models, `Model/classify_model.py` for legacy ResNet34 classification, and `train.py` as the historical entry point.
