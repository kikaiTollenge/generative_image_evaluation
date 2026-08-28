# generative_image_evaluation

`generative_image_evaluation` is the public code repository for the paper **Synthetic Colorectal Cancer Histologic Images via StyleGAN3: Enhancing Database Diversity and Model Ability**. It provides training, evaluation, and analysis scripts for synthetic colorectal cancer histopathology image experiments, including StyleGAN3 image generation, MoCo pre-training, downstream classification and segmentation, HoVer-Net instance-output morphology analysis, FID-style image-quality comparison, and repeated-run statistical analysis.

## What You Can Run

- Generate synthetic CRC images through an external NVIDIA StyleGAN3 checkout.
- Pre-train MoCo encoders on synthetic or real CRC image folders.
- Fine-tune/evaluate classification models on Warwick-QU/GlaS and BreakHis.
- Fine-tune/evaluate semantic segmentation models on Warwick-QU/GlaS and MoNuSeg.
- Compare downstream encoders: Synthetic, Real, w/o Pre-train, RetCCL, and CTransPath.
- Compute nuclear morphology features from HoVer-Net `inst` outputs.
- Compute FID-style image-quality comparisons.
- Run Mann-Whitney U, Holm correction, and Cliff's delta on repeated-result CSVs.

## Repository Layout

```text
Dataset/
  dataset.py
  pathology_datasets.py
  prepare_glas_metadata.py
Model/
  Moco/
  Unet/
  classify_model.py
  pathology_encoders.py
analysis/
  nuclear_morphology.py
  fid50k.py
  statistical_comparison.py
scripts/
  generate_with_stylegan3.py
  make_dummy_data.py
  run_classification.py
  run_segmentation.py
```

## Installation

Install PyTorch for your CUDA version, then install the Python dependencies:

```bash
pip install -r requirements.txt
```

CTransPath requires `timm`. RetCCL, CTransPath, Synthetic, and Real encoder runs can load checkpoints through `--checkpoint`.

## Data Layout

Keep datasets, checkpoints, generated images, HoVer-Net outputs, and StyleGAN3 code outside Git-tracked source files. The scripts accept explicit path arguments, so any equivalent local layout is fine. One convenient layout is:

```text
data/
  warwick_qu/
    train/image/
    train/mask/
    test/image/
    test/mask/
    Grade.csv
    Grade_modified.csv
  breakhis/
    adenosis/40X/*.png
    fibroadenoma/40X/*.png
    phyllodes_tumor/40X/*.png
    tubular_adenoma/40X/*.png
    ductal_carcinoma/40X/*.png
    lobular_carcinoma/40X/*.png
    mucinous_carcinoma/40X/*.png
    papillary_carcinoma/40X/*.png
  monuseg/
    train/images/
    train/masks/
    test/images/
    test/masks/
  crc_synthetic_pretrain/
    class0/*.png
  crc_real_pretrain/
    class0/*.png
  hovernet_output/
    real/inst/*_inst.npy
    synthetic/inst/*_inst.npy
```

Warwick-QU and GlaS refer to the same benchmark in this codebase. The legacy folder name is `glas`.

If only `Grade.csv` is available for Warwick-QU/GlaS, rebuild the helper metadata:

```bash
python Dataset/prepare_glas_metadata.py \
  --input /path/to/warwick_qu/Grade.csv \
  --output /path/to/warwick_qu/Grade_modified.csv
```

## StyleGAN3 Image Generation

Keep the official NVIDIA StyleGAN3 repository outside this project:

```bash
git clone https://github.com/NVlabs/stylegan3.git /path/to/stylegan3
```

Run the project wrapper with a trained generator checkpoint:

Generator weights: [Google Drive](https://drive.google.com/file/d/1Oms4N_vgLwkMt9RnW4XSCGtDEozCyNA7/view?usp=drive_link)

```bash
python scripts/generate_with_stylegan3.py \
  --stylegan3-repo /path/to/stylegan3 \
  --network /path/to/downloaded_crc_stylegan3_generator.pkl \
  --outdir /path/to/generated_crc_images \
  --seeds 1
```

Use any seed list accepted by StyleGAN3, such as `1`, `1,2,3`, or `1-10`.

For a dependency-free plumbing check without a generator checkpoint:

```bash
python scripts/generate_with_stylegan3.py \
  --smoke-test \
  --outdir /tmp/stylegan3_smoke \
  --seeds 1 \
  --image-size 1024
```

The `--smoke-test` mode writes random RGB images and is only for checking file output.

## MoCo Pre-Training

`Model/Moco/main_moco.py` uses `torchvision.datasets.ImageFolder`. Put images under at least one class subfolder, for example:

```text
/path/to/crc_synthetic_pretrain/class0/*.png
/path/to/crc_real_pretrain/class0/*.png
```

Class labels are ignored by MoCo; the subfolder is only required by `ImageFolder`.

Run synthetic-source pre-training:

```bash
mkdir -p /path/to/runs/synthetic_moco
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python Model/Moco/main_moco.py \
  -a moco_encoder \
  --lr 0.015 \
  --batch-size 256 \
  --epochs 400 \
  --dist-url tcp://localhost:10001 \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --mlp \
  --moco-k 65536 \
  --moco-dim 256 \
  --moco-t 0.07 \
  --moco-m 0.999 \
  --aug-plus \
  --cos \
  --output-dir /path/to/runs/synthetic_moco \
  /path/to/crc_synthetic_pretrain \
  > /path/to/runs/synthetic_moco/output.log 2>&1 &
```

Run real-source pre-training by changing the data and output paths:

```bash
mkdir -p /path/to/runs/real_moco
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python Model/Moco/main_moco.py \
  -a moco_encoder \
  --lr 0.015 \
  --batch-size 256 \
  --epochs 400 \
  --dist-url tcp://localhost:10002 \
  --multiprocessing-distributed \
  --world-size 1 \
  --rank 0 \
  --mlp \
  --moco-k 65536 \
  --moco-dim 256 \
  --moco-t 0.07 \
  --moco-m 0.999 \
  --aug-plus \
  --cos \
  --output-dir /path/to/runs/real_moco \
  /path/to/crc_real_pretrain \
  > /path/to/runs/real_moco/output.log 2>&1 &
```

The command above follows the paper training settings. For code checks, reduce `--epochs`, `--batch-size`, and `--moco-k`.

## Classification

The classification script writes:

```text
seed,method,dataset,loss,accuracy,f1
```

Warwick-QU/GlaS:

```bash
python scripts/run_classification.py \
  --dataset warwick_qu \
  --method synthetic \
  --checkpoint /path/to/synthetic_moco_checkpoint.pth.tar \
  --csv /path/to/warwick_qu/Grade_modified.csv \
  --data-root /path/to/data_root \
  --binary \
  --epochs 50 \
  --repeat 100 \
  --batch-size 32 \
  --output results/classification_warwick_synthetic.csv
```

BreakHis 40X:

```bash
python scripts/run_classification.py \
  --dataset breakhis \
  --method real \
  --checkpoint /path/to/real_moco_checkpoint.pth.tar \
  --data-root /path/to/breakhis \
  --split-csv /path/to/breakhis_40x_split.csv \
  --epochs 50 \
  --repeat 100 \
  --batch-size 128 \
  --output results/classification_breakhis_real.csv
```

Supported methods are `synthetic`, `real`, `without-pretrain`, `retccl`, `ctranspath`, and `resnet34`. If `--split-csv` is omitted for BreakHis, the loader creates a deterministic balanced split from the directory tree.

## Semantic Segmentation

The segmentation script writes:

```text
seed,method,dataset,loss,dice,iou
```

Warwick-QU/GlaS:

```bash
python scripts/run_segmentation.py \
  --dataset warwick_qu \
  --method synthetic \
  --checkpoint /path/to/synthetic_moco_checkpoint.pth.tar \
  --train-images /path/to/warwick_qu/train/image \
  --train-masks /path/to/warwick_qu/train/mask \
  --test-images /path/to/warwick_qu/test/image \
  --test-masks /path/to/warwick_qu/test/mask \
  --epochs 50 \
  --repeat 100 \
  --batch-size 16 \
  --lr 1e-4 \
  --output results/segmentation_warwick_synthetic.csv
```

MoNuSeg:

```bash
python scripts/run_segmentation.py \
  --dataset monuseg \
  --method real \
  --checkpoint /path/to/real_moco_checkpoint.pth.tar \
  --train-images /path/to/monuseg/train/images \
  --train-masks /path/to/monuseg/train/masks \
  --test-images /path/to/monuseg/test/images \
  --test-masks /path/to/monuseg/test/masks \
  --epochs 50 \
  --repeat 100 \
  --batch-size 16 \
  --lr 5e-4 \
  --output results/segmentation_monuseg_real.csv
```

Supported methods are the same as classification.

## Nuclear Morphology

Run HoVer-Net separately, then pass this script the contents of HoVer-Net's `inst` output folders. The expected input is instance-id arrays such as `*_inst.npy`; original pathology images are not inputs to this script.

```bash
python analysis/nuclear_morphology.py \
  --real-mask-dir /path/to/hovernet_output/real/inst \
  --synthetic-mask-dir /path/to/hovernet_output/synthetic/inst \
  --features-output results/nuclear_morphology_features.csv \
  --summary-output results/nuclear_morphology_summary.csv
```

The features CSV contains:

```text
image,group,n_nuclei,area_unit,distance_unit,median_nuclear_area,mean_circularity,nuclear_density,median_nn_distance,nuclear_area_cv
```

Use `--microns-per-pixel` to report area and distance in physical units.

## FID / Image Quality

Use `analysis/fid50k.py` to compare a real-reference image folder with a generated-image folder:

```bash
python analysis/fid50k.py \
  --real-dir /path/to/real_crc_reference_images \
  --generated-dir /path/to/generated_images \
  --feature-extractor inception \
  --pretrained \
  --limit 50000 \
  --output results/fid_generated.csv
```

For offline code checks, use the lightweight color-statistics path:

```bash
python analysis/fid50k.py \
  --real-dir /path/to/real_images \
  --generated-dir /path/to/generated_images \
  --feature-extractor color \
  --limit 2 \
  --output /tmp/fid_smoke.csv
```

The output CSV contains:

```text
metric,real_images,generated_images,image_size,fid
```

## Statistical Comparison

Use `analysis/statistical_comparison.py` on repeated-run CSVs:

```bash
python analysis/statistical_comparison.py \
  --input results/classification_warwick_all_methods.csv \
  --metric accuracy \
  --group-col method \
  --output results/classification_warwick_accuracy_stats.csv
```

The output includes medians, IQRs, median differences, Mann-Whitney U, Holm-adjusted P values, and Cliff's delta.
