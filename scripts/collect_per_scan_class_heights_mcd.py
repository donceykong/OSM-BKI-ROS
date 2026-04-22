#!/usr/bin/env python3
"""Collect per-scan per-class mean height for MCD scans and write to JSON.

Same transform as collect_class_heights_mcd.py: poses are normalized to the
first pose, the reference up axis is the first scan's lidar +z in the
first-pose-relative map frame, and each scan's z_base is the bottom-most
projected point. Unlike the histogram version, this script records a single
mean height per class *per scan* (no binning), so downstream analysis can
see how each class' height varies across the trajectory.

Output JSON layout:
    {
      "dataset": "mcd",
      "lidar_up_ref": [...],
      "class_names": [...],
      "scans": [
        {"index": <scan_id>, "z_base": float, "pose_xyz": [x,y,z],
         "class_means": {"road": {"mean": 0.3, "count": 1234}, ...}},
        ...
      ]
    }
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
        return os.path.abspath(data_dir_override)
    data_root = (cfg.get("data_root") or "").strip()
    dataset_name = cfg.get("dataset_name", "mcd")
    if data_root:
        return os.path.join(data_root, dataset_name)
    return os.path.join(SCRIPT_DIR, "data", dataset_name)


def main():
    parser = argparse.ArgumentParser(description="Per-scan per-class mean height for MCD.")
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config/methods/mcd.yaml"))
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "per_scan_class_heights_mcd.json"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-scans", type=int, default=50000)
    parser.add_argument("--keyframe-dist", type=float, default=1.0)
    parser.add_argument("--z-clip-low", type=float, default=5.0,
                        help="Lower percentile for scan-global z outlier clipping (for z_base)")
    parser.add_argument("--z-clip-high", type=float, default=95.0,
                        help="Upper percentile for scan-global z outlier clipping")
    parser.add_argument("--class-clip-low", type=float, default=1.0,
                        help="Lower percentile for per-class outlier clipping before mean")
    parser.add_argument("--class-clip-high", type=float, default=99.0,
                        help="Upper percentile for per-class outlier clipping before mean")
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

    _first_lidar_to_map = poses[0][1] @ lidar_to_body
    lidar_up_ref = _first_lidar_to_map[:3, 2].astype(np.float64)
    n_up = np.linalg.norm(lidar_up_ref)
    if n_up > 1e-6:
        lidar_up_ref /= n_up

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

    scans_out = []
    for (idx, T, scan_path, label_path) in tqdm(scan_list, desc="scans", unit="scan"):
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
        z_local = (map_pts - origin_map).astype(np.float64) @ lidar_up_ref

        # Scan-global outlier clip to stabilize z_base (a single stray reflection
        # at very low z would otherwise drag the bottom reference down).
        z_lo = float(np.percentile(z_local, args.z_clip_low))
        z_hi = float(np.percentile(z_local, args.z_clip_high))
        keep_scan = (z_local >= z_lo) & (z_local <= z_hi)
        if not keep_scan.any():
            continue
        z_local_kept = z_local[keep_scan]
        gt_kept = gt_common[keep_scan]
        z_base = float(z_local_kept.min())
        h_above = z_local_kept - z_base  # >= 0, no binning

        class_means = {}
        for c in range(N_CLASSES):
            mask = gt_kept == c
            if not mask.any():
                continue
            hc = h_above[mask]
            # Per-class outlier clip (drops stray reflections that survived the
            # global clip but are anomalous within this class).
            if hc.size >= 4:
                lo = float(np.percentile(hc, args.class_clip_low))
                hi = float(np.percentile(hc, args.class_clip_high))
                hc = hc[(hc >= lo) & (hc <= hi)]
            if hc.size == 0:
                continue
            class_means[CLASS_NAMES[c]] = {
                "mean": float(hc.mean()),
                "count": int(hc.size),
            }

        scans_out.append({
            "index": int(idx),
            "z_base": z_base,
            "pose_xyz": [float(T[0, 3]), float(T[1, 3]), float(T[2, 3])],
            "class_means": class_means,
        })

    out = {
        "dataset": "mcd",
        "sequence_name": cfg.get("sequence_name"),
        "num_scans": len(scans_out),
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
        "class_names": CLASS_NAMES,
        "scans": scans_out,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote per-scan per-class mean heights to {args.output}")
    print(f"  scans={len(scans_out)}  classes={N_CLASSES}")


if __name__ == "__main__":
    main()
