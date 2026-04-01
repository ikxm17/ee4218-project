"""
evaluate_map.py
===============
Evaluate TinyissimoYOLO using ap_per_class() from the TinyissimoYOLO
Ultralytics fork — bypassing DetMetrics entirely.

ap_per_class() is the stable core function that all Ultralytics versions
call internally. It is present in all forks and has a consistent signature:

    ap_per_class(tp, conf, pred_cls, target_cls,
                 plot=False, save_dir=Path('.'), names={})
    returns: (tp, fp, p, r, f1, ap, unique_classes)
        tp : [nc, 10]  true positives at each IoU threshold
        ap : [nc, 10]  average precision at each IoU threshold
        p  : [nc]      precision at best F1 threshold
        r  : [nc]      recall at best F1 threshold

This approach is version-agnostic — it works identically regardless of
which Ultralytics fork or version is installed.

Config values (from args.yaml + tinyissimo-v1-small.yaml):
    nc       : 3
    imgsz    : 256
    conf     : null  -> 0.001 (Ultralytics validator default)
    iou      : 0.7
    max_det  : 300
"""

from __future__ import annotations

import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Optional
from tqdm import tqdm

from ultralytics.utils.metrics import ap_per_class, box_iou
from ultralytics.utils.ops import non_max_suppression, xywhn2xyxy

# ── Constants from args.yaml / model yaml ─────────────────────────────────────
NC        = 3
IMGSZ     = 256
CONF_NULL = 0.001    # Ultralytics DetectionValidator default when conf: null
IOU_THRES = 0.7      # from args.yaml
MAX_DET   = 300

# Standard COCO mAP IoU sweep — 0.50 to 0.95 in 10 steps
IOUV = torch.linspace(0.5, 0.95, 10)
NIOU = IOUV.numel()  # 10


# ── Confidence threshold resolver ─────────────────────────────────────────────

def _resolve_conf(args_yaml: Optional[str]) -> float:
    """
    Read conf from args.yaml.
    conf: null / conf: false / key absent  ->  0.001
    conf: 0.25                             ->  0.25
    """
    if args_yaml is None:
        return CONF_NULL
    with open(args_yaml) as f:
        cfg = yaml.safe_load(f)
    val = cfg.get("conf", None)
    if val is None or val is False:
        return CONF_NULL
    try:
        return float(val)
    except (TypeError, ValueError):
        return CONF_NULL


# ── Per-image prediction matcher ──────────────────────────────────────────────

def _match_predictions(
    pred_cls: torch.Tensor,
    true_cls: torch.Tensor,
    iou:      torch.Tensor,
    iouv:     torch.Tensor,
) -> torch.Tensor:
    """
    Greedy IoU matching — same algorithm as Ultralytics DetectionValidator.

    Args:
        pred_cls : [D]    predicted class indices (int64)
        true_cls : [M]    ground truth class indices (int64)
        iou      : [M, D] pairwise IoU matrix
        iouv     : [10]   IoU thresholds 0.50 -> 0.95

    Returns:
        correct : [D, 10] bool
    """
    D = pred_cls.shape[0]
    correct = torch.zeros(D, iouv.shape[0],
                          dtype=torch.bool, device=pred_cls.device)
    cls_match = true_cls.unsqueeze(1) == pred_cls.unsqueeze(0)  # [M, D]

    for t_idx, threshold in enumerate(iouv):
        tp_mask = (iou >= threshold) & cls_match
        if tp_mask.any():
            iou_masked       = iou * tp_mask.float()
            best_pred_per_gt = iou_masked.argmax(dim=1)
            best_iou_per_gt  = iou_masked.max(dim=1).values
            valid_gt         = best_iou_per_gt >= threshold
            matched_preds    = best_pred_per_gt[valid_gt].unique()
            correct[matched_preds, t_idx] = True

    return correct


# ── Core evaluate_map ─────────────────────────────────────────────────────────

def evaluate_map(
    model:      nn.Module,
    val_loader: torch.utils.data.DataLoader,
    nc:         int   = NC,
    imgsz:      int   = IMGSZ,
    conf:       Optional[float] = None,
    iou:        float = IOU_THRES,
    max_det:    int   = MAX_DET,
    args_yaml:  Optional[str]   = None,
    device:     Optional[torch.device] = None,
    verbose:    bool  = True,
) -> dict:
    """
    Evaluate TinyissimoYOLO detection performance.

    Uses ap_per_class() directly — bypassing DetMetrics entirely — so the
    function works with any Ultralytics version or fork, including the
    custom TinyissimoYOLO fork, without needing to match the DetMetrics API.

    ap_per_class() is the stable numerical core that all versions share.

    Args:
        model      : quantized or FP32 TinyissimoYOLO nn.Module
        val_loader : Ultralytics DataLoader yielding batch dicts
        nc         : number of classes (default 3)
        imgsz      : model input resolution (default 256)
        conf       : NMS confidence threshold (None -> read from args_yaml)
        iou        : NMS IoU threshold (default 0.7, from args.yaml)
        max_det    : max detections per image (default 300)
        args_yaml  : path to args.yaml for conf resolution
        device     : inference device (autodetected if None)
        verbose    : print results table

    Returns:
        dict with keys:
            map       — mAP@0.5:0.95
            map50     — mAP@0.5
            map75     — mAP@0.75
            precision — mean precision
            recall    — mean recall
            maps      — per-class mAP@0.5:0.95  np.ndarray [nc]
    """
    if device is None:
        device = next(model.parameters()).device

    if conf is None:
        conf = _resolve_conf(args_yaml)
        src = 'from args.yaml' if args_yaml else 'default'
        print(f"[evaluate_map] conf={conf} ({src})")

    iouv_dev = IOUV.to(device)
    names    = {i: str(i) for i in range(nc)}

    # ── Accumulators (flat arrays across entire val set) ──────────────────────
    all_tp       = []   # [D, 10] bool per image
    all_conf     = []   # [D]     float per image
    all_pred_cls = []   # [D]     int per image
    all_tgt_cls  = []   # [M]     int per image (all GT, including no-det images)
    seen         = 0

    # ── Force batch_size=1 via a dedicated single-image loader ──────────────
    # The onnx2torch-converted model contains OnnxShape nodes inside the
    # Detect head that call torch.tensor() on CUDA tensor shapes.  This
    # raises "CUDA illegal memory access" for batch_size > 1 because
    # torch.tensor() cannot accept CUDA-backed shape tuples directly.
    # Running with batch_size=1 avoids all dynamic shape computations.
    from torch.utils.data import DataLoader
    from ultralytics.data.dataset import YOLODataset

    def _collate_f32(batch):
        result = YOLODataset.collate_fn(batch)
        result["img"] = result["img"].float() / 255.0
        return result

    single_loader = DataLoader(
        val_loader.dataset,
        batch_size  = 1,
        shuffle     = False,
        num_workers = 0,
        collate_fn  = _collate_f32,
        pin_memory  = False,
    )

    model.eval()

    with torch.no_grad():
        for batch in tqdm(single_loader, desc="Evaluating mAP", leave=False):

            imgs      = batch["img"].to(device)
            gt_bboxes = batch["bboxes"].to(device)    # [N,4] xywh normalised
            gt_cls    = batch["cls"].to(device)        # [N,1] int
            gt_bidx   = batch["batch_idx"].to(device)  # [N] int
            B         = imgs.shape[0]
            seen     += B

            # ── Forward pass ────────────────────────────────────────────────
            # BackboneWrapper returns (decoded, raw_feat):
            #   output[0]: [B, 4+nc, num_preds] = [B, 7, 64]
            #   output[1]: [B, 67, 8, 8] raw feat for QAT loss
            #
            # Pass output[0] directly to non_max_suppression.
            # NMS (from TinyissimoYOLO fork) handles internally:
            #   - transpose [B, 4+nc, N] -> [B, N, 4+nc]
            #   - cxcywh -> xyxy via xywh2xyxy()
            #   - confidence = max(class_scores) > conf_thres
            #   - no separate objectness column (nc = shape[1] - 4)
            raw       = model(imgs)
            preds_raw = raw[0]    # [B, 7, 64] = [B, 4+nc, num_preds]

            # ── NMS ──────────────────────────────────────────────────────────
            # Output: list of B tensors [D, 6]: x1,y1,x2,y2,conf,cls (xyxy)
            det_list = non_max_suppression(
                preds_raw,
                conf_thres = conf,
                iou_thres  = iou,
                max_det    = max_det,
            )

            # ── Per-image matching ────────────────────────────────────────────
            for img_i, dets in enumerate(det_list):

                mask       = (gt_bidx == img_i)
                gt_boxes_n = gt_bboxes[mask]
                gt_labels  = gt_cls[mask].squeeze(1).long()
                nl         = gt_labels.shape[0]

                # Always record GT labels for this image
                all_tgt_cls.append(gt_labels.cpu())

                # Convert GT to absolute pixel xyxy
                gt_boxes_abs = xywhn2xyxy(
                    gt_boxes_n, w=imgsz, h=imgsz
                ).to(device)

                no_dets = (dets is None or dets.shape[0] == 0)
                if no_dets:
                    # No predictions — nothing to record on pred side
                    continue

                pred_boxes = dets[:, :4]
                pred_conf  = dets[:, 4]
                pred_cls   = dets[:, 5].long()

                if nl == 0:
                    # No GT for this image — all preds are FP
                    correct = torch.zeros(dets.shape[0], NIOU,
                                          dtype=torch.bool, device=device)
                else:
                    iou_matrix = box_iou(gt_boxes_abs, pred_boxes)  # [M, D]
                    correct    = _match_predictions(
                        pred_cls, gt_labels, iou_matrix, iouv_dev
                    )

                all_tp.append(correct.cpu())
                all_conf.append(pred_conf.cpu())
                all_pred_cls.append(pred_cls.cpu())

    # ── Handle empty detection case ───────────────────────────────────────────
    if len(all_tp) == 0:
        print("[evaluate_map] No detections — check conf threshold or model.")
        return {"map": 0.0, "map50": 0.0, "map75": 0.0,
                "precision": 0.0, "recall": 0.0,
                "maps": np.zeros(nc)}

    # ── Concatenate all per-image results into flat arrays ────────────────────
    tp_np       = torch.cat(all_tp,       dim=0).numpy()   # [total_dets, 10]
    conf_np     = torch.cat(all_conf,     dim=0).numpy()   # [total_dets]
    pred_cls_np = torch.cat(all_pred_cls, dim=0).numpy()   # [total_dets]
    tgt_cls_np  = torch.cat(all_tgt_cls,  dim=0).numpy()   # [total_gt]

    # ── Call ap_per_class — stable across all Ultralytics versions ────────────
    # Signature:
    #   ap_per_class(tp, conf, pred_cls, target_cls,
    #                plot=False, save_dir=Path('.'), names={})
    # Returns: (tp_out, fp, p, r, f1, ap, unique_classes)
    #   ap  : [nc_seen, 10]  AP at each IoU threshold per class
    #   p   : [nc_seen]      precision at best F1
    #   r   : [nc_seen]      recall at best F1
    results = ap_per_class(
        tp_np,
        conf_np,
        pred_cls_np,
        tgt_cls_np,
        plot    = False,
        save_dir= Path("."),
        names   = names,
    )

    # TinyissimoYOLO fork returns 12 values (confirmed from source):
    #   [0]  tp             [nc]
    #   [1]  fp             [nc]
    #   [2]  p              [nc]      precision at best F1
    #   [3]  r              [nc]      recall at best F1
    #   [4]  f1             [nc]
    #   [5]  ap             [nc, 10]  AP at 10 IoU thresholds
    #   [6]  unique_classes [nc]      int
    #   [7]  p_curve        [nc, 1000]  — not needed
    #   [8]  r_curve        [nc, 1000]  — not needed
    #   [9]  f1_curve       [nc, 1000]  — not needed
    #   [10] x              [1000]      — not needed
    #   [11] prec_values    [nc, 1000]  — not needed
    (_, _, p, r, f1, ap, unique_classes,
     _, _, _, _, _) = results

    # ap shape: [nc_seen, 10] — columns are IoU thresholds 0.50..0.95
    map50    = float(ap[:, 0].mean())        # IoU=0.50 column
    map75_idx = 5                            # IoU=0.75 is index 5
    map75    = float(ap[:, map75_idx].mean()) if ap.shape[1] > map75_idx else 0.0
    map_all  = float(ap.mean())             # mean over all 10 thresholds
    mp       = float(p.mean())
    mr       = float(r.mean())

    # Per-class mAP@0.5:0.95 — fill a [nc] array (unseen classes get 0)
    maps = np.zeros(nc)
    for i, cls_idx in enumerate(unique_classes.astype(int)):
        if cls_idx < nc:
            maps[cls_idx] = float(ap[i].mean())

    if verbose:
        _print_results(p, r, ap, unique_classes, names, seen, nc)

    return {
        "map":       map_all,
        "map50":     map50,
        "map75":     map75,
        "precision": mp,
        "recall":    mr,
        "maps":      maps,
    }


# ── Results printer ───────────────────────────────────────────────────────────

def _print_results(
    p:              np.ndarray,
    r:              np.ndarray,
    ap:             np.ndarray,
    unique_classes: np.ndarray,
    names:          dict,
    seen:           int,
    nc:             int,
):
    header = (f"\n{'Class':<20} {'Images':>8} {'P':>8} "
              f"{'R':>8} {'mAP50':>10} {'mAP50-95':>12}")
    sep    = "-" * 70
    print(sep)
    print(header)
    print(sep)

    # All-class summary
    map50   = float(ap[:, 0].mean()) if ap.size else 0.0
    map_all = float(ap.mean())       if ap.size else 0.0
    mp      = float(p.mean())        if p.size  else 0.0
    mr      = float(r.mean())        if r.size  else 0.0
    print(f"{'all':<20} {seen:>8} {mp:>8.3f} "
          f"{mr:>8.3f} {map50:>10.4f} {map_all:>12.4f}")

    # Per-class rows
    for i, cls_idx in enumerate(unique_classes.astype(int)):
        name    = names.get(cls_idx, str(cls_idx))
        ap50_i  = float(ap[i, 0])
        ap_i    = float(ap[i].mean())
        p_i     = float(p[i]) if i < len(p) else 0.0
        r_i     = float(r[i]) if i < len(r) else 0.0
        print(f"  {name:<18} {'':>8} {p_i:>8.3f} "
              f"{r_i:>8.3f} {ap50_i:>10.4f} {ap_i:>12.4f}")

    print(sep)