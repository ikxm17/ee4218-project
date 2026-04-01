"""
dataloader.py — Ultralytics-native DataLoader for PTQ and QAT
--------------------------------------------------------------
Loads all arguments directly from an existing args.yaml file produced
by a TinyissimoYOLO training run, ensuring preprocessing, augmentation,
and hyperparameters are identical to training.

Usage:
    from dataloader import build_ptq_calibration_loader, build_qat_loaders

    # PTQ calibration loader
    calib_loader = build_ptq_calibration_loader(
        args_yaml="results/exp2/args.yaml",
        n_calib=200,
    )
    for batch in calib_loader:
        images = batch["img"]   # [B, 3, 256, 256] float32 in [0,1]

    # QAT train + val loaders
    train_loader, val_loader = build_qat_loaders(
        args_yaml="results/exp2/args.yaml",
    )
    for batch in train_loader:
        images  = batch["img"]        # [B, 3, 256, 256]
        bboxes  = batch["bboxes"]     # [N, 4]  normalised xywh
        cls     = batch["cls"]        # [N, 1]  class ids
        idx     = batch["batch_idx"]  # [N]     image index in batch
"""

import random
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader, Subset

# ── Ensure local ultralytics package takes priority ───────────────────────────
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ultralytics.cfg import get_cfg               # noqa: E402
from ultralytics.data.dataset import YOLODataset  # noqa: E402
from ultralytics.utils import DEFAULT_CFG         # noqa: E402


# ── Collate wrapper ───────────────────────────────────────────────────────────

def _collate_float32(batch):
    """
    Wraps YOLODataset.collate_fn and converts 'img' from uint8 [0, 255]
    to float32 [0.0, 1.0].

    Ultralytics' collate_fn returns uint8 because the Ultralytics trainer
    applies normalisation internally before passing images to the model.
    Since the quantization pipeline bypasses the trainer entirely, this
    wrapper must perform that conversion so the model always receives
    float32 input in the expected range.

    build_dataloader() does not expose a collate_fn parameter, so all
    loaders are built with plain DataLoader() to allow this override.
    """
    result = YOLODataset.collate_fn(batch)
    result["img"] = result["img"].float() / 255.0
    return result


# ── Internal helpers ──────────────────────────────────────────────────────────

def _load_args(args_yaml: str):
    """
    Load args.yaml into a validated Ultralytics config namespace.
    Uses get_cfg so all type coercion and defaults are applied identically
    to the training pipeline.
    """
    with open(args_yaml) as f:
        overrides = yaml.safe_load(f)
    args = get_cfg(DEFAULT_CFG, overrides)
    return args


def _resolve_img_path(args, split_key: str) -> str:
    """
    Resolve the image directory for a given split key ("train" or "val"),
    mirroring how Ultralytics DetectionTrainer.build_dataset() resolves paths.

    Handles both:
      - Direct directory entries:  train: images/train2017
      - Image-list .txt entries:   train: train2017.txt
    """
    with open(args.data) as f:
        data_cfg = yaml.safe_load(f)

    root  = Path(data_cfg.get("path", "."))
    value = data_cfg[split_key]
    p     = Path(value)

    if not p.is_absolute():
        p = root / p

    if p.suffix == ".txt":
        with open(p) as f:
            first = f.readline().strip()
        return str(Path(first).parent)

    return str(p)


def _load_data_dict(data_yaml: str) -> dict:
    """Parse data YAML into the dict Ultralytics dataset constructors expect."""
    with open(data_yaml) as f:
        data = yaml.safe_load(f)
    if isinstance(data.get("names"), list):
        data["names"] = {i: n for i, n in enumerate(data["names"])}
    data.setdefault("nc", len(data["names"]))
    return data


def _build_yolo_dataset(
    args,
    split_key: str,
    augment:   bool,
) -> YOLODataset:
    """
    Construct YOLODataset using values pulled directly from args.yaml.
    Matches TinyissimoYOLO's DetectionTrainer.build_dataset() exactly.
    """
    img_path = _resolve_img_path(args, split_key)
    data     = _load_data_dict(args.data)

    args_copy         = get_cfg(DEFAULT_CFG, vars(args))
    args_copy.augment = augment
    args_copy.mosaic  = args.mosaic if augment else 0.0

    dataset = YOLODataset(
        img_path    = img_path,
        imgsz       = args.imgsz,
        batch_size  = args.batch,
        augment     = augment,
        hyp         = args_copy,
        rect        = False,
        cache       = args.cache,
        single_cls  = args.single_cls,
        stride      = 32,
        pad         = 0.5,
        data        = data,
        task        = "detect",
    )
    return dataset


# ── Public API ────────────────────────────────────────────────────────────────

def build_ptq_calibration_loader(
    args_yaml:  str,
    n_calib:    int = 200,
    batch_size: int = 1,
    split_key:  str = "val",
) -> DataLoader:
    """
    Build a calibration DataLoader for Post-Training Quantization (PTQ).

    Pulls img_size, data path, cache, workers directly from args.yaml.
    Uses the val split with NO augmentation and a fixed random subset
    of n_calib images for reproducibility.

    Args:
        args_yaml  : path to args.yaml from a training run
        n_calib    : number of calibration images (200-500 typical)
        batch_size : images per batch — default 1 for PTQ calibration
        split_key  : which split to sample from (default "val")

    Returns:
        DataLoader yielding dicts with keys:
            "img"      [B, 3, H, W]  float32 in [0.0, 1.0]
            "bboxes"   [N, 4]        normalised xywh
            "cls"      [N, 1]        class indices
            "im_file"  list[str]     source image paths
    """
    args    = _load_args(args_yaml)
    dataset = _build_yolo_dataset(args, split_key, augment=False)

    rng     = random.Random(42)
    n       = min(n_calib, len(dataset))
    indices = rng.sample(range(len(dataset)), n)
    subset  = Subset(dataset, indices)

    print(f"PTQ calibration")
    print(f"  args.yaml  : {args_yaml}")
    print(f"  split      : {split_key}")
    print(f"  images     : {n}/{len(dataset)}")
    print(f"  img_size   : {args.imgsz}")
    print(f"  batch_size : {batch_size}")

    return DataLoader(
        subset,
        batch_size  = batch_size,
        shuffle     = False,
        num_workers = min(args.workers, 2),
        collate_fn  = _collate_float32,
        pin_memory  = False,
    )


def build_qat_loaders(
    args_yaml:  str,
    batch_size: int  = None,
    workers:    int  = None,
) -> tuple:
    """
    Build train and val DataLoaders for Quantization-Aware Training (QAT).

    All hyperparameters (img_size, augmentation, mosaic, hsv_*, etc.) are
    pulled from args.yaml so QAT fine-tuning uses identical preprocessing
    to the original training run.

    build_dataloader() from Ultralytics does not expose a collate_fn
    parameter, so both loaders are constructed with plain DataLoader()
    to allow _collate_float32 to be injected. All other settings
    (batch size, workers, drop_last) match what build_dataloader would
    have produced.

    Args:
        args_yaml  : path to args.yaml from a training run
        batch_size : override batch size (default: value in args.yaml)
        workers    : override worker count (default: value in args.yaml)

    Returns:
        (train_loader, val_loader)
    """
    args = _load_args(args_yaml)

    if batch_size is not None:
        args.batch   = batch_size
    if workers is not None:
        args.workers = workers

    train_dataset = _build_yolo_dataset(args, "train", augment=True)
    val_dataset   = _build_yolo_dataset(args, "val",   augment=False)

    # Plain DataLoader — NOT build_dataloader — so collate_fn can be set.
    # pin_memory=False avoids thread teardown errors when the CUDA toolkit
    # is absent (driver present but nvcc not installed).
    train_loader = DataLoader(
        train_dataset,
        batch_size  = args.batch,
        shuffle     = True,
        num_workers = args.workers,
        collate_fn  = _collate_float32,
        pin_memory  = False,
        drop_last   = True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size  = args.batch,
        shuffle     = False,
        num_workers = args.workers * 2,
        collate_fn  = _collate_float32,
        pin_memory  = False,
        drop_last   = False,
    )

    print(f"QAT loaders")
    print(f"  args.yaml  : {args_yaml}")
    print(f"  img_size   : {args.imgsz}")
    print(f"  batch_size : {args.batch}")
    print(f"  workers    : {args.workers}")
    print(f"  train      : {len(train_dataset)} images  augment=True  mosaic={args.mosaic}")
    print(f"  val        : {len(val_dataset)} images  augment=False")

    return train_loader, val_loader


# ── Smoke test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--args-yaml", type=str, default="results/exp2/args.yaml",
                        help="Path to args.yaml from a TinyissimoYOLO training run")
    parser.add_argument("--n-calib", type=int, default=50)
    args = parser.parse_args()

    _loaders_to_close = []

    print("\n── PTQ calibration loader ───────────────────────────────")
    calib_loader = build_ptq_calibration_loader(
        args_yaml  = args.args_yaml,
        n_calib    = args.n_calib,
        batch_size = 1,
    )
    _loaders_to_close.append(calib_loader)
    batch = next(iter(calib_loader))
    print(f"  img       : {batch['img'].shape}  dtype={batch['img'].dtype}  "
          f"range=[{batch['img'].min():.3f}, {batch['img'].max():.3f}]")
    print(f"  bboxes    : {batch['bboxes'].shape}")
    print(f"  cls       : {batch['cls'].shape}")
    print(f"  im_file   : {batch['im_file'][0]}")

    print("\n── QAT train/val loaders ────────────────────────────────")
    train_loader, val_loader = build_qat_loaders(
        args_yaml  = args.args_yaml,
        batch_size = 4,
    )
    _loaders_to_close.extend([train_loader, val_loader])
    batch = next(iter(train_loader))
    print(f"  [train] img       : {batch['img'].shape}  dtype={batch['img'].dtype}  "
          f"range=[{batch['img'].min():.3f}, {batch['img'].max():.3f}]")
    print(f"  [train] bboxes    : {batch['bboxes'].shape}")
    print(f"  [train] cls       : {batch['cls'].shape}")
    print(f"  [train] batch_idx : {batch['batch_idx'].shape}")

    batch = next(iter(val_loader))
    print(f"  [val]   img       : {batch['img'].shape}  dtype={batch['img'].dtype}  "
          f"range=[{batch['img'].min():.3f}, {batch['img'].max():.3f}]")

    print("\nSmoke test passed.")

    # Explicitly release iterators before exit to prevent worker thread
    # teardown races that print ConnectionResetError / ConnectionRefusedError.
    # for loader in _loaders_to_close:
    #     loader._iterator = None