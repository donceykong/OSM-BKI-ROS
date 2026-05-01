"""
Memory-efficient baseline eval: raw per-scan softmax argmax vs GT, both
mapped through the common 9-class taxonomy.

Mirrors baseline_evaluation.ipynb but streams each scan into a 9x9
confusion matrix instead of concatenating ~250M-row arrays. Math is
identical to sklearn.metrics.jaccard_score with labels=[1..8] applied
to the gt!=0-filtered arrays.
"""

import argparse
import glob
import os
import time

import numpy as np
import yaml

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


def build_luts(labels_common_path: str, labels_inferred_path: str,
               inferred_key: str, gt_key: str) -> tuple[np.ndarray, np.ndarray]:
    with open(labels_common_path, "r") as f:
        common_cfg = yaml.safe_load(f)
    with open(labels_inferred_path, "r") as f:
        inferred_cfg = yaml.safe_load(f)

    inferred_to_common = {int(k): int(v) for k, v in common_cfg[f"{inferred_key}_to_common"].items()}
    gt_to_common = {int(k): int(v) for k, v in common_cfg[f"{gt_key}_to_common"].items()}
    learning_map_inv = {int(k): int(v) for k, v in inferred_cfg["learning_map_inv"].items()}

    max_ch = max(learning_map_inv.keys()) + 1
    channel_to_common = np.zeros(max_ch, dtype=np.int32)
    for ch, raw in learning_map_inv.items():
        channel_to_common[ch] = inferred_to_common.get(raw, 0)

    gt_lut_size = max(max(gt_to_common.keys()) + 1, 65536)
    gt_lut = np.zeros(gt_lut_size, dtype=np.int32)
    for k, v in gt_to_common.items():
        if 0 <= k < gt_lut_size:
            gt_lut[k] = v

    return channel_to_common, gt_lut


def accumulate_confusion(scan_dir: str, softmax_dir: str, gt_dir: str,
                         eval_ref_dir: str, channel_to_common: np.ndarray,
                         gt_lut: np.ndarray,
                         max_scans: int | None) -> tuple[np.ndarray, int, int]:
    eval_ref_files = sorted(glob.glob(os.path.join(eval_ref_dir, "*.txt")))
    ref_ids = [os.path.splitext(os.path.basename(p))[0] for p in eval_ref_files]
    print(f"Found {len(ref_ids)} .txt files in {eval_ref_dir}")
    if not ref_ids:
        raise FileNotFoundError(f"No .txt files in {eval_ref_dir}")

    ok_ids = []
    for sid in ref_ids:
        if (os.path.exists(os.path.join(scan_dir, sid + ".bin"))
                and os.path.exists(os.path.join(softmax_dir, sid + ".bin"))
                and os.path.exists(os.path.join(gt_dir, sid + ".bin"))):
            ok_ids.append(sid)
    print(f"Scans with all files present: {len(ok_ids)} / {len(ref_ids)}")
    if not ok_ids:
        raise RuntimeError("No scans to evaluate.")

    if max_scans is not None:
        ok_ids = ok_ids[:max_scans]
        print(f"Using first {len(ok_ids)} for evaluation")

    cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    total = 0
    skipped = 0
    t0 = time.time()
    for i, sid in enumerate(ok_ids):
        scan_path = os.path.join(scan_dir, sid + ".bin")
        sm_path = os.path.join(softmax_dir, sid + ".bin")
        gt_path = os.path.join(gt_dir, sid + ".bin")

        scan_bytes = os.path.getsize(scan_path)
        n_points = scan_bytes // (4 * 4)
        if n_points == 0:
            skipped += 1
            continue

        sm_bytes = os.path.getsize(sm_path)
        n_classes_file = sm_bytes // (2 * n_points)
        if n_classes_file * 2 * n_points != sm_bytes:
            print(f"[skip] {sid}: softmax size mismatch")
            skipped += 1
            continue

        softmax = np.fromfile(sm_path, dtype=np.float16).reshape(n_points, n_classes_file)
        pred_channel = np.argmax(softmax, axis=1).astype(np.int32)
        pred_channel = np.where(pred_channel < channel_to_common.shape[0], pred_channel, 0)
        pred_common = channel_to_common[pred_channel]

        gt_raw = np.fromfile(gt_path, dtype=np.uint32)
        if gt_raw.shape[0] != n_points:
            m = min(gt_raw.shape[0], n_points)
            gt_raw = gt_raw[:m]
            pred_common = pred_common[:m]
        gt_raw_clipped = np.where(gt_raw < gt_lut.shape[0], gt_raw, 0)
        gt_common = gt_lut[gt_raw_clipped]

        idx = gt_common.astype(np.int64) * NUM_CLASSES + pred_common.astype(np.int64)
        cm += np.bincount(idx, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
            NUM_CLASSES, NUM_CLASSES
        )
        total += gt_common.shape[0]

        if (i + 1) % 200 == 0 or i + 1 == len(ok_ids):
            print(f"  [{i+1}/{len(ok_ids)}] points={total:>12d}  "
                  f"elapsed={time.time()-t0:6.1f}s")

    return cm, total, skipped


def report(cm_full: np.ndarray, total: int) -> None:
    semantic_classes = np.arange(1, NUM_CLASSES)
    cm = cm_full[1:, :]                            # drop GT=0 rows only
    gt_counts = cm.sum(axis=1)                     # includes pred=0
    tp = cm[np.arange(NUM_CLASSES - 1), semantic_classes].astype(np.float64)
    pred_counts = cm[:, 1:].sum(axis=0)            # gt in 1..8, pred=c

    with np.errstate(divide="ignore", invalid="ignore"):
        denom_iou = gt_counts + pred_counts - tp
        iou_scores = np.where(denom_iou > 0, tp / denom_iou, 0.0)
        per_class_acc = np.where(gt_counts > 0, tp / gt_counts, 0.0)
        per_class_prec = np.where(pred_counts > 0, tp / pred_counts, 0.0)

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
              f"{iou_scores[i]:>8.4f}  {per_class_acc[i]:>10.4f}  "
              f"{per_class_prec[i]:>10.4f}  "
              f"{int(gt_counts[i]):>12d}  {int(pred_counts[i]):>12d}")

    present = gt_counts > 0
    miou = iou_scores[present].mean() if present.any() else 0.0
    mean_acc = per_class_acc[present].mean() if present.any() else 0.0
    mean_prec = per_class_prec[present].mean() if present.any() else 0.0
    overall_acc = tp.sum() / eval_total if eval_total > 0 else 0.0

    print(f"\nmIoU (over {int(present.sum())} present classes): {miou:.4f}")
    print(f"Mean accuracy (recall):                {mean_acc:.4f}")
    print(f"Mean precision:                        {mean_prec:.4f}")
    print(f"Overall accuracy:                      {overall_acc:.4f}")

    print(f"& {100*iou_scores[0]:.4f} & {100*iou_scores[1]:.4f} & {100*iou_scores[2]:.4f} & {100*iou_scores[3]:.4f} & {100*iou_scores[4]:.4f} & {100*iou_scores[7]:.4f} & {100*iou_scores[5]:.4f} & {100*iou_scores[6]:.4f} & {100*miou:.4f} & {100*mean_acc:.4f} \\\\")


def main() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="/media/donceykong/doncey_ssd_02/datasets/")
    p.add_argument("--dataset", default="kitti360")
    p.add_argument("--sequence", default="2013_05_28_drive_0000_sync")
    p.add_argument("--scan-suffix", default="velodyne_points/data")
    p.add_argument("--softmax-suffix", default="inferred_labels/cenet_kitti360_softmax")
    p.add_argument("--gt-suffix", default="gt_labels")  # "gt_labels_terrain" for MCD
    p.add_argument("--eval-ref-suffix", default="evaluations/in_domain/vanilla")
    p.add_argument("--inferred-key", default="kitti360")
    p.add_argument("--gt-key", default="kitti360")
    p.add_argument("--labels-common", default=os.path.join(here, "../config/datasets/labels_common.yaml"))
    p.add_argument("--labels-inferred", default=os.path.join(here, "../config/datasets/labels_kitti360.yaml"))
    p.add_argument("--max-scans", type=int, default=None)
    args = p.parse_args()

    sequence_dir = os.path.join(args.data_dir, args.dataset, args.sequence)
    scan_dir = os.path.join(sequence_dir, args.scan_suffix)
    softmax_dir = os.path.join(sequence_dir, args.softmax_suffix)
    gt_dir = os.path.join(sequence_dir, args.gt_suffix)
    eval_ref_dir = os.path.join(sequence_dir, args.eval_ref_suffix)

    print(f"Scan dir:    {scan_dir}")
    print(f"Softmax dir: {softmax_dir}")
    print(f"GT dir:      {gt_dir}")
    print(f"Eval ref:    {eval_ref_dir}")

    channel_to_common, gt_lut = build_luts(
        args.labels_common, args.labels_inferred, args.inferred_key, args.gt_key
    )
    print(f"channel_to_common LUT size: {channel_to_common.shape}")
    print(f"gt_lut size: {gt_lut.shape}")

    cm, total, skipped = accumulate_confusion(
        scan_dir, softmax_dir, gt_dir, eval_ref_dir,
        channel_to_common, gt_lut, args.max_scans,
    )
    print(f"Scans skipped: {skipped}")
    report(cm, total)


if __name__ == "__main__":
    main()
