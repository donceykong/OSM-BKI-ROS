#!/usr/bin/env python3
"""Collect per-class height statistics for MCD scans and write to JSON.

Uses the exact same transform and height-binning approach as
optimize_osm_cm_mcd.py: poses are normalized to the first pose, the reference
up axis is the first scan's lidar +z expressed in the first-pose-relative
map frame, and per-scan heights are measured upward from that scan's
bottom-most point along the up axis.

For each common class the script stores the mean, median, std, min, max,
point count, and a histogram of per-point height-above-bottom values so the
companion notebook can plot distributions.

Usage:
    python3 collect_class_heights_mcd.py [--config config/methods/mcd.yaml]
                                         [--output class_heights_mcd.json]
                                         [--max-scans N] [--keyframe-dist M]
                                         [--hist-step 0.25] [--hist-max 30.0]
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from optimize_osm_cm_mcd import (  # noqa: E402
    CLASS_NAMES,
    N_CLASSES,
    load_poses,
    load_calibration,
    load_label_mappings,
    read_scan_bin,
    read_label_bin,
)


def resolve_paths(cfg, data_dir_override):
    seq = cfg.get("sequence_name")
    if seq:
        if cfg.get("lidar_pose_suffix"):
            cfg["lidar_pose_file"] = f"{seq}/{cfg['lidar_pose_suffix']}"
        if cfg.get("input_data_suffix"):
            cfg["input_data_prefix"] = f"{seq}/{cfg['input_data_suffix']}"
        if cfg.get("gt_label_suffix"):
            cfg["gt_label_prefix"] = f"{seq}/{cfg['gt_label_suffix']}"

    if data_dir_override:
        data_dir = os.path.abspath(data_dir_override)
    else:
        data_root = (cfg.get("data_root") or "").strip()
        dataset_name = cfg.get("dataset_name", "mcd")
        if data_root:
            data_dir = os.path.join(data_root, dataset_name)
        else:
            data_dir = os.path.join(SCRIPT_DIR, "data", dataset_name)
    return data_dir


def main():
    parser = argparse.ArgumentParser(description="Collect per-class height statistics for MCD.")
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config/methods/mcd.yaml"))
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "class_heights_mcd.json"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-scans", type=int, default=50000)
    parser.add_argument("--keyframe-dist", type=float, default=5.0)
    parser.add_argument("--hist-step", type=float, default=0.5,
                        help="Histogram bin width in meters")
    parser.add_argument("--hist-max", type=float, default=30.0,
                        help="Histogram upper edge in meters (heights clipped into last bin)")
    parser.add_argument("--z-clip-low", type=float, default=0.5,
                        help="Lower percentile for scan-global z outlier clipping (for z_base)")
    parser.add_argument("--z-clip-high", type=float, default=99.5,
                        help="Upper percentile for scan-global z outlier clipping")
    parser.add_argument("--class-clip-low", type=float, default=1.0,
                        help="Lower percentile for per-class outlier clipping before stats")
    parser.add_argument("--class-clip-high", type=float, default=99.0,
                        help="Upper percentile for per-class outlier clipping before stats")
    args = parser.parse_args()

    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = raw_cfg
    if "/**" in cfg:
        cfg = cfg["/**"].get("ros__parameters", cfg)
    elif "ros__parameters" in cfg:
        cfg = cfg["ros__parameters"]

    data_dir = resolve_paths(cfg, args.data_dir)
    pose_file = os.path.join(data_dir, cfg["lidar_pose_file"])
    gt_label_dir = os.path.join(data_dir, cfg.get("gt_label_prefix", "kth_day_09/gt_labels"))
    scan_dir = os.path.join(data_dir, cfg.get("input_data_prefix", "kth_day_09/lidar_bin/data"))
    calib_file = os.path.join(data_dir, "hhs_calib.yaml")
    gt_labels_key = cfg.get("gt_labels_key", "mcd")
    ds_resolution = cfg.get("ds_resolution", 1.0)

    labels_common_path = os.path.join(SCRIPT_DIR, "config/datasets/labels_common.yaml")
    gt_mapping = load_label_mappings(labels_common_path, gt_labels_key)

    print(f"Loading poses from {pose_file}")
    poses = load_poses(pose_file)
    print(f"  Loaded {len(poses)} poses")

    first_pose = poses[0][1].copy()
    first_inv = np.linalg.inv(first_pose)
    for i in range(len(poses)):
        poses[i] = (poses[i][0], first_inv @ poses[i][1])

    body_to_lidar = load_calibration(calib_file)
    lidar_to_body = np.linalg.inv(body_to_lidar)

    # Reference up axis: +z of the first scan's lidar in the first-pose-relative map frame.
    _first_lidar_to_map = poses[0][1] @ lidar_to_body
    lidar_up_ref = _first_lidar_to_map[:3, 2].astype(np.float64)
    n_up = np.linalg.norm(lidar_up_ref)
    if n_up > 1e-6:
        lidar_up_ref /= n_up

    # Keyframe selection matches the optimizer.
    keyframe_dist = args.keyframe_dist if args.keyframe_dist is not None else cfg.get("keyframe_dist", 0.0)
    scan_list = []
    last_kf = None
    for idx, T in poses:
        sp = os.path.join(scan_dir, f"{idx:010d}.bin")
        lp = os.path.join(gt_label_dir, f"{idx:010d}.bin")
        if not (os.path.isfile(sp) and os.path.isfile(lp)):
            continue
        pos = T[:3, 3]
        if last_kf is not None and keyframe_dist > 0 and np.linalg.norm(pos - last_kf) < keyframe_dist:
            continue
        scan_list.append((idx, T, sp, lp))
        last_kf = pos
        if len(scan_list) >= args.max_scans:
            break
    print(f"Selected {len(scan_list)} scans (keyframe_dist={keyframe_dist}m)")

    # Histogram setup: bins in [0, hist_max], last bin absorbs overflow.
    step = float(args.hist_step)
    h_max = float(args.hist_max)
    num_bins = int(np.ceil(h_max / step))
    bin_edges = np.arange(num_bins + 1, dtype=np.float64) * step  # length num_bins+1

    # Running accumulators per class so we don't hold every point in RAM.
    sum_h = np.zeros(N_CLASSES, dtype=np.float64)
    sum_h2 = np.zeros(N_CLASSES, dtype=np.float64)
    count = np.zeros(N_CLASSES, dtype=np.int64)
    min_h = np.full(N_CLASSES, np.inf, dtype=np.float64)
    max_h = np.full(N_CLASSES, -np.inf, dtype=np.float64)
    hist = np.zeros((N_CLASSES, num_bins), dtype=np.int64)
    # For median, keep per-class reservoir of samples (cap to avoid blowing up memory).
    reservoir_cap = 200_000
    reservoirs = [[] for _ in range(N_CLASSES)]

    rng = np.random.default_rng(0)

    for si, (idx, T, scan_path, label_path) in enumerate(
            tqdm(scan_list, desc="scans", unit="scan")):
        pts = read_scan_bin(scan_path)
        labels_raw = read_label_bin(label_path)
        n = min(len(pts), len(labels_raw))
        if n == 0:
            continue
        pts, labels_raw = pts[:n], labels_raw[:n]

        gt_common = np.array([gt_mapping.get(int(l), 0) for l in labels_raw], dtype=np.int32)

        lidar_to_map = T @ lidar_to_body
        xyz_h = np.hstack([pts[:, :3], np.ones((n, 1), dtype=np.float32)])
        map_pts = (lidar_to_map @ xyz_h.T).T[:, :3]

        if ds_resolution > 0:
            vkeys = np.floor(map_pts / ds_resolution).astype(np.int64)
            _, uidx = np.unique(vkeys, axis=0, return_index=True)
            map_pts = map_pts[uidx]
            gt_common = gt_common[uidx]

        origin_map = lidar_to_map[:3, 3]
        # Same definition as optimizer: z_local = (map_pts - origin) · up_ref.
        z_local = (map_pts - origin_map).astype(np.float64) @ lidar_up_ref

        # Scan-global outlier clip to stabilize z_base against stray reflections.
        z_lo = float(np.percentile(z_local, args.z_clip_low))
        z_hi = float(np.percentile(z_local, args.z_clip_high))
        keep_scan = (z_local >= z_lo) & (z_local <= z_hi)
        if not keep_scan.any():
            continue
        z_local_kept = z_local[keep_scan]
        gt_kept = gt_common[keep_scan]
        z_base = float(z_local_kept.min())
        h_above = z_local_kept - z_base  # >= 0

        # Clip to histogram support (last bin absorbs overflow).
        clipped = np.clip(h_above, 0.0, h_max - 1e-9)
        bins = np.floor(clipped / step).astype(np.int32)
        np.clip(bins, 0, num_bins - 1, out=bins)

        for c in range(N_CLASSES):
            mask = gt_kept == c
            if not mask.any():
                continue
            hc = h_above[mask]
            cb = bins[mask]
            # Per-class outlier clip (drops stray points that survived the global clip).
            if hc.size >= 4:
                lo_c = float(np.percentile(hc, args.class_clip_low))
                hi_c = float(np.percentile(hc, args.class_clip_high))
                km = (hc >= lo_c) & (hc <= hi_c)
                hc = hc[km]
                cb = cb[km]
            if hc.size == 0:
                continue
            sum_h[c] += hc.sum()
            sum_h2[c] += (hc * hc).sum()
            count[c] += hc.size
            min_h[c] = min(min_h[c], float(hc.min()))
            max_h[c] = max(max_h[c], float(hc.max()))
            np.add.at(hist[c], cb, 1)

            # Reservoir sample for median estimation.
            res = reservoirs[c]
            if len(res) < reservoir_cap:
                take = min(reservoir_cap - len(res), hc.size)
                if take >= hc.size:
                    res.extend(hc.tolist())
                else:
                    pick = rng.choice(hc.size, size=take, replace=False)
                    res.extend(hc[pick].tolist())
            else:
                idxs = rng.integers(0, count[c], size=hc.size)
                keep = idxs < reservoir_cap
                for local_i, slot in zip(np.nonzero(keep)[0], idxs[keep]):
                    res[int(slot)] = float(hc[local_i])

    stats = {}
    for c in range(N_CLASSES):
        if count[c] == 0:
            stats[CLASS_NAMES[c]] = {
                "class_index": c,
                "count": 0,
                "mean": None,
                "median": None,
                "std": None,
                "min": None,
                "max": None,
                "histogram": hist[c].tolist(),
            }
            continue
        mean = sum_h[c] / count[c]
        var = max(0.0, sum_h2[c] / count[c] - mean * mean)
        std = float(np.sqrt(var))
        median = float(np.median(np.asarray(reservoirs[c]))) if reservoirs[c] else None
        stats[CLASS_NAMES[c]] = {
            "class_index": c,
            "count": int(count[c]),
            "mean": float(mean),
            "median": median,
            "std": std,
            "min": float(min_h[c]),
            "max": float(max_h[c]),
            "histogram": hist[c].tolist(),
        }

    out = {
        "dataset": "mcd",
        "sequence_name": cfg.get("sequence_name"),
        "num_scans": len(scan_list),
        "keyframe_dist_m": keyframe_dist,
        "ds_resolution_m": ds_resolution,
        "lidar_up_ref": lidar_up_ref.tolist(),
        "height_reference": "per_scan_bottom_most_point_along_first_frame_lidar_up",
        "outlier_clip": {
            "z_clip_low_pct": args.z_clip_low,
            "z_clip_high_pct": args.z_clip_high,
            "class_clip_low_pct": args.class_clip_low,
            "class_clip_high_pct": args.class_clip_high,
        },
        "histogram": {
            "step_m": step,
            "max_m": h_max,
            "num_bins": num_bins,
            "bin_edges_m": bin_edges.tolist(),
        },
        "class_names": CLASS_NAMES,
        "classes": stats,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote per-class height stats to {args.output}")
    print(f"{'class':<12} {'count':>10} {'mean':>8} {'median':>8} {'std':>8} {'min':>8} {'max':>8}")
    for name, s in stats.items():
        if s["count"] == 0:
            print(f"{name:<12} {s['count']:>10}")
        else:
            print(f"{name:<12} {s['count']:>10} {s['mean']:>8.2f} "
                  f"{(s['median'] or 0):>8.2f} {s['std']:>8.2f} "
                  f"{s['min']:>8.2f} {s['max']:>8.2f}")


if __name__ == "__main__":
    main()
