#!/usr/bin/env python3
"""Evaluate ablation: compute per-class IoU, mIoU, overall accuracy,
and full confusion matrix from query_scan() output files.

Each output file has lines: gt_label pred_label

Usage:
    python3 evaluate_ablation.py --input DIR --label NAME [--output DIR]
    python3 evaluate_ablation.py --compare DIR_A LABEL_A DIR_B LABEL_B
"""

import argparse
import csv
import os
import sys

import numpy as np

# Common taxonomy (classes 1-12; 0 = unlabeled, excluded from metrics)
CLASS_NAMES = [
    "unlabeled",      # 0 — excluded
    "road",           # 1
    "sidewalk",       # 2
    "parking",        # 3
    "other-ground",   # 4
    "building",       # 5
    "fence",          # 6
    "pole",           # 7
    "traffic-sign",   # 8
    "vegetation",     # 9
    "two-wheeler",    # 10
    "vehicle",        # 11
    "other-object",   # 12
]
NUM_CLASSES = 13  # 0-12


def load_predictions(eval_dir):
    """Load all per-scan evaluation files and return (gt, pred) arrays."""
    all_gt = []
    all_pred = []
    scan_count = 0

    for fname in sorted(os.listdir(eval_dir)):
        if not fname.endswith(".txt"):
            continue
        fpath = os.path.join(eval_dir, fname)
        with open(fpath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                gt, pred = int(parts[0]), int(parts[1])
                all_gt.append(gt)
                all_pred.append(pred)
        scan_count += 1

    return np.array(all_gt, dtype=np.int32), np.array(all_pred, dtype=np.int32), scan_count


def compute_confusion_matrix(gt, pred, num_classes):
    """Compute confusion matrix C[gt][pred]."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for g, p in zip(gt, pred):
        if 0 <= g < num_classes and 0 <= p < num_classes:
            cm[g, p] += 1
    return cm


def compute_metrics(cm):
    """Compute per-class IoU, mIoU, overall accuracy from confusion matrix.
    Excludes class 0 (unlabeled).
    """
    per_class_iou = {}
    valid_classes = []

    for k in range(1, cm.shape[0]):  # skip class 0
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp  # predicted as k but not k
        fn = cm[k, :].sum() - tp  # actually k but predicted as something else
        denom = tp + fp + fn
        if cm[k, :].sum() == 0:
            # No GT points for this class
            per_class_iou[k] = float("nan")
            continue
        iou = tp / denom if denom > 0 else 0.0
        per_class_iou[k] = iou
        valid_classes.append(k)

    miou = np.nanmean([per_class_iou[k] for k in valid_classes]) if valid_classes else 0.0

    # Overall accuracy: correct / total, excluding class 0
    mask = np.ones(cm.shape[0], dtype=bool)
    mask[0] = False
    cm_no_unlabeled = cm[np.ix_(mask, mask)]
    total = cm_no_unlabeled.sum()
    correct = np.diag(cm_no_unlabeled).sum()
    overall_acc = correct / total if total > 0 else 0.0

    return per_class_iou, miou, overall_acc


def print_results(label, per_class_iou, miou, overall_acc, scan_count, total_points):
    """Print formatted results."""
    print(f"\n{'='*80}")
    print(f"  {label}")
    print(f"  Scans: {scan_count}, Points: {total_points:,}")
    print(f"{'='*80}")
    print(f"  {'Class':<15} {'IoU':>8}")
    print(f"  {'-'*23}")
    for k in range(1, NUM_CLASSES):
        iou = per_class_iou.get(k, float("nan"))
        iou_str = f"{iou*100:.2f}%" if not np.isnan(iou) else "N/A"
        print(f"  {CLASS_NAMES[k]:<15} {iou_str:>8}")
    print(f"  {'-'*23}")
    print(f"  {'mIoU':<15} {miou*100:.2f}%")
    print(f"  {'Overall Acc':<15} {overall_acc*100:.2f}%")


def save_confusion_matrix(cm, output_path):
    """Save 12x12 confusion matrix (excluding class 0) as CSV."""
    names = CLASS_NAMES[1:]
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["gt\\pred"] + names)
        for i, name in enumerate(names):
            row = cm[i + 1, 1:].tolist()
            writer.writerow([name] + row)
    print(f"  Saved confusion matrix to {output_path}")


def save_results_csv(label, per_class_iou, miou, overall_acc, output_path):
    """Save one-row CSV matching paper Table I format."""
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["method"] + [CLASS_NAMES[k] for k in range(1, NUM_CLASSES)] + ["mIoU", "overall_acc"]
        writer.writerow(header)
        row = [label]
        for k in range(1, NUM_CLASSES):
            iou = per_class_iou.get(k, float("nan"))
            row.append(f"{iou*100:.2f}" if not np.isnan(iou) else "N/A")
        row.append(f"{miou*100:.2f}")
        row.append(f"{overall_acc*100:.2f}")
        writer.writerow(row)
    print(f"  Saved results CSV to {output_path}")


def run_single(eval_dir, label, output_dir=None):
    """Evaluate a single run."""
    if output_dir is None:
        output_dir = eval_dir

    gt, pred, scan_count = load_predictions(eval_dir)
    if len(gt) == 0:
        print(f"ERROR: No evaluation data found in {eval_dir}")
        return None

    cm = compute_confusion_matrix(gt, pred, NUM_CLASSES)
    per_class_iou, miou, overall_acc = compute_metrics(cm)
    print_results(label, per_class_iou, miou, overall_acc, scan_count, len(gt))
    save_confusion_matrix(cm, os.path.join(output_dir, "confusion_matrix.csv"))
    save_results_csv(label, per_class_iou, miou, overall_acc,
                     os.path.join(output_dir, "results.csv"))

    return {
        "label": label,
        "cm": cm,
        "per_class_iou": per_class_iou,
        "miou": miou,
        "overall_acc": overall_acc,
        "scan_count": scan_count,
        "total_points": len(gt),
    }


def run_compare(dir_a, label_a, dir_b, label_b, output_dir=None):
    """Compare two runs side-by-side."""
    res_a = run_single(dir_a, label_a)
    res_b = run_single(dir_b, label_b)
    if res_a is None or res_b is None:
        return

    if output_dir is None:
        output_dir = os.path.dirname(dir_a.rstrip("/"))

    # Print comparison table
    print(f"\n{'='*90}")
    print(f"  COMPARISON: {label_a} vs {label_b}")
    print(f"{'='*90}")
    print(f"  {'Class':<15} {label_a:>12} {label_b:>12} {'Delta':>10}")
    print(f"  {'-'*49}")
    for k in range(1, NUM_CLASSES):
        iou_a = res_a["per_class_iou"].get(k, float("nan"))
        iou_b = res_b["per_class_iou"].get(k, float("nan"))
        if np.isnan(iou_a) or np.isnan(iou_b):
            delta_str = "N/A"
        else:
            delta = (iou_b - iou_a) * 100
            sign = "+" if delta >= 0 else ""
            delta_str = f"{sign}{delta:.2f}%"
        a_str = f"{iou_a*100:.2f}%" if not np.isnan(iou_a) else "N/A"
        b_str = f"{iou_b*100:.2f}%" if not np.isnan(iou_b) else "N/A"
        print(f"  {CLASS_NAMES[k]:<15} {a_str:>12} {b_str:>12} {delta_str:>10}")
    print(f"  {'-'*49}")

    miou_delta = (res_b["miou"] - res_a["miou"]) * 100
    acc_delta = (res_b["overall_acc"] - res_a["overall_acc"]) * 100
    sign_m = "+" if miou_delta >= 0 else ""
    sign_a = "+" if acc_delta >= 0 else ""
    print(f"  {'mIoU':<15} {res_a['miou']*100:>11.2f}% {res_b['miou']*100:>11.2f}% {sign_m}{miou_delta:>9.2f}%")
    print(f"  {'Overall Acc':<15} {res_a['overall_acc']*100:>11.2f}% {res_b['overall_acc']*100:>11.2f}% {sign_a}{acc_delta:>9.2f}%")
    print(f"\n  Scans:  {res_a['scan_count']} vs {res_b['scan_count']}")
    print(f"  Points: {res_a['total_points']:,} vs {res_b['total_points']:,}")

    # Save combined CSV
    combined_path = os.path.join(output_dir, "ablation_comparison.csv")
    with open(combined_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["method"] + [CLASS_NAMES[k] for k in range(1, NUM_CLASSES)] + ["mIoU", "overall_acc"]
        writer.writerow(header)
        for res in [res_a, res_b]:
            row = [res["label"]]
            for k in range(1, NUM_CLASSES):
                iou = res["per_class_iou"].get(k, float("nan"))
                row.append(f"{iou*100:.2f}" if not np.isnan(iou) else "N/A")
            row.append(f"{res['miou']*100:.2f}")
            row.append(f"{res['overall_acc']*100:.2f}")
            writer.writerow(row)
        # Delta row
        row = ["Delta (B-A)"]
        for k in range(1, NUM_CLASSES):
            a = res_a["per_class_iou"].get(k, float("nan"))
            b = res_b["per_class_iou"].get(k, float("nan"))
            if np.isnan(a) or np.isnan(b):
                row.append("N/A")
            else:
                row.append(f"{(b-a)*100:+.2f}")
        row.append(f"{miou_delta:+.2f}")
        row.append(f"{acc_delta:+.2f}")
        writer.writerow(row)
    print(f"\n  Saved comparison to {combined_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OSM-BKI ablation results")
    sub = parser.add_subparsers(dest="cmd")

    single = sub.add_parser("single", help="Evaluate a single run")
    single.add_argument("--input", required=True, help="Directory with per-scan .txt files")
    single.add_argument("--label", required=True, help="Method name for display")
    single.add_argument("--output", default=None, help="Output directory (default: same as input)")

    compare = sub.add_parser("compare", help="Compare two runs")
    compare.add_argument("--dir-a", required=True, help="Run A eval directory")
    compare.add_argument("--label-a", required=True, help="Run A method name")
    compare.add_argument("--dir-b", required=True, help="Run B eval directory")
    compare.add_argument("--label-b", required=True, help="Run B method name")
    compare.add_argument("--output", default=None, help="Output directory for comparison CSV")

    args = parser.parse_args()
    if args.cmd == "single":
        run_single(args.input, args.label, args.output)
    elif args.cmd == "compare":
        run_compare(args.dir_a, args.label_a, args.dir_b, args.label_b, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
