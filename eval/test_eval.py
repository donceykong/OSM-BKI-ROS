"""
Memory-efficient eval: streams files into a 9x9 confusion matrix instead of
holding ~250M-row gt/pred arrays + per-class boolean masks in RAM.

Why the notebook crashes:
  1. np.loadtxt + list-append + np.concatenate peaks at multiple GB of
     transient memory for ~300M int32 rows.
  2. The per-class section then allocates ~250M-element boolean masks
     repeatedly, and jaccard_score materializes large internal arrays.

This script keeps memory bounded: each file is parsed, accumulated into
a 9x9 confusion matrix, and discarded. Math is identical to
sklearn.metrics.jaccard_score with labels=[1..8] applied to the
gt!=0-filtered arrays.
"""

import argparse
import glob
import os
import time

import numpy as np
import pandas as pd

COMMON_CLASSES = [
    "unlabeled",   # 0
    "road",        # 1
    "sidewalk",    # 2
    "parking",     # 3
    "building",    # 4
    "fence",       # 5
    "vegetation",  # 6
    "vehicle",     # 7
    "terrain",     # 8
]
NUM_CLASSES = len(COMMON_CLASSES)


def accumulate_confusion(eval_dir: str) -> tuple[np.ndarray, int]:
    files = sorted(glob.glob(os.path.join(eval_dir, "*.txt")))
    print(f"Found {len(files)} evaluation files in {eval_dir}")
    if not files:
        raise FileNotFoundError(f"No .txt files in {eval_dir}")

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total = 0
    t0 = time.time()
    for i, fpath in enumerate(files):
        arr = pd.read_csv(
            fpath, sep=r"\s+", header=None, dtype=np.int16, engine="c",
        ).values
        gt = arr[:, 0].astype(np.int64, copy=False)
        pr = arr[:, 1].astype(np.int64, copy=False)
        idx = gt * NUM_CLASSES + pr
        cm += np.bincount(
            idx, minlength=NUM_CLASSES * NUM_CLASSES
        ).reshape(NUM_CLASSES, NUM_CLASSES)
        total += arr.shape[0]
        if (i + 1) % 200 == 0 or i + 1 == len(files):
            print(f"  [{i+1}/{len(files)}] points={total:>12d}  "
                  f"elapsed={time.time()-t0:6.1f}s")
    return cm, total


def report(cm_full: np.ndarray, total: int) -> None:
    # Drop GT=0 rows only (matches `mask = gt_all != 0` in the notebook).
    # Keep all prediction columns so points with gt!=0, pred=0 still count
    # as FN for the GT class — this is what jaccard_score(labels=[1..8])
    # does on the masked arrays.
    semantic_classes = np.arange(1, NUM_CLASSES)
    cm = cm_full[1:, :]                            # shape (8, 9)

    gt_counts = cm.sum(axis=1)                     # includes pred=0
    tp = cm[np.arange(NUM_CLASSES - 1), semantic_classes].astype(np.float64)
    pred_counts = cm[:, 1:].sum(axis=0)            # gt in 1..8, pred=c

    with np.errstate(divide="ignore", invalid="ignore"):
        denom_iou = gt_counts + pred_counts - tp
        iou = np.where(denom_iou > 0, tp / denom_iou, 0.0)
        recall = np.where(gt_counts > 0, tp / gt_counts, 0.0)
        precision = np.where(pred_counts > 0, tp / pred_counts, 0.0)

    eval_total = int(gt_counts.sum())
    print(f"\nTotal points: {total}")
    print(f"Points after removing unlabeled (GT class 0): {eval_total}")

    header = (f"\n{'Class':>3}  {'Name':<15s}  {'IoU':>8s}  "
              f"{'Accuracy':>10s}  {'Precision':>10s}  "
              f"{'GT count':>12s}  {'Pred count':>12s}")
    print(header)
    print("-" * (len(header) - 1))
    for i, cls_id in enumerate(semantic_classes):
        print(f"{cls_id:>3d}  {COMMON_CLASSES[cls_id]:<15s}  "
              f"{iou[i]:>8.4f}  {recall[i]:>10.4f}  {precision[i]:>10.4f}  "
              f"{int(gt_counts[i]):>12d}  {int(pred_counts[i]):>12d}")

    present = gt_counts > 0
    miou = iou[present].mean() if present.any() else 0.0
    mrec = recall[present].mean() if present.any() else 0.0
    mpre = precision[present].mean() if present.any() else 0.0
    oacc = tp.sum() / eval_total if eval_total > 0 else 0.0

    print(f"\nmIoU (over {int(present.sum())} present classes): {miou:.4f}")
    print(f"Mean accuracy (recall):                {mrec:.4f}")
    print(f"Mean precision:                        {mpre:.4f}")
    print(f"Overall accuracy:                      {oacc:.4f}")

    print(
        f"\n& {100*iou[0]:.4f} & {100*iou[1]:.4f} & {100*iou[2]:.4f} "
        f"& {100*iou[3]:.4f} & {100*iou[4]:.4f} & {100*iou[7]:.4f} "
        f"& {100*iou[5]:.4f} & {100*iou[6]:.4f} & {100*miou:.4f} "
        f"& {100*mrec:.4f} \\\\"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/media/donceykong/doncey_ssd_02/datasets/")
    p.add_argument("--dataset", default="kitti360")
    p.add_argument("--sequence", default="2013_05_28_drive_0000_sync")
    p.add_argument("--prefix", default="in_domain/vanilla")
    args = p.parse_args()

    eval_dir = os.path.join(
        args.data_dir, args.dataset, args.sequence, "evaluations", args.prefix
    )
    cm, total = accumulate_confusion(eval_dir)
    report(cm, total)


if __name__ == "__main__":
    main()
