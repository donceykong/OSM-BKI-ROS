#!/usr/bin/env python3
"""Optimize per-class Gaussian height priors (mu, tau) for MCD GT data.

The produced (mu, tau) pair is meant to be pasted directly into
`config/methods/mcd.yaml` under `height_kernel_mu` / `height_kernel_tau`
(with `height_filter_type: gaussian`).

The height reference here matches the C++ runtime
(`apply_height_kernel_to_ybars` in bkioctomap.cpp):

    h = z_map - (origin_z - sensor_mounting_height)

i.e. the height of a point above the estimated ground plane, where the
ground plane is defined as `sensor_mounting_height` meters below the
current lidar pose z. This is *not* the per-scan bottom reference used by
`collect_per_scan_class_heights_mcd.py`; we deliberately align to the
runtime semantics so the learned priors fire at the intended heights.

For each scan we:
  1. Transform points to the (first-pose-normalized) map frame.
  2. Compute per-point h using the formula above.
  3. Drop per-scan z outliers (percentile clip) to stabilize stats.
  4. Accumulate per-class height samples (with a memory cap).

After all scans we apply a per-class percentile clip to reject outliers,
then report mean (mu) and std (tau) with a floor on tau. Class 0
(unlabeled) gets a wide tau override so the kernel effectively leaves it
alone.

Output: a YAML at config/datasets/height_gaussians_mcd.yaml containing
the ready-to-paste `height_kernel_mu` / `height_kernel_tau` lists plus
per-class diagnostics (count, raw mean/std/median).
"""

import argparse
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


def reservoir_append(store, arr, limit, rng):
    """Append `arr` to the per-class sample store, capped at `limit` total samples.

    Uses a simple cap (drops overflow, optionally downsampling the last batch)
    rather than a true reservoir — callers should also bound `--max-scans` so
    the sample distribution stays representative of the dataset.
    """
    if arr.size == 0:
        return
    if limit <= 0:
        store.append(arr)
        return
    current = int(sum(a.size for a in store))
    if current >= limit:
        return
    remaining = limit - current
    if remaining >= arr.size:
        store.append(arr)
        return
    idx = rng.choice(arr.size, size=remaining, replace=False)
    store.append(arr[idx])


def robust_mu_tau(h, clip_low, clip_high, min_tau):
    """Percentile-clipped mean/std. Returns (mu, tau, n_used, raw_stats)."""
    raw_stats = {
        "count_raw": int(h.size),
        "raw_mean": float(h.mean()) if h.size else None,
        "raw_std": float(h.std()) if h.size else None,
        "raw_median": float(np.median(h)) if h.size else None,
    }
    if h.size == 0:
        return None, None, 0, raw_stats
    if h.size >= 4:
        lo = np.percentile(h, clip_low)
        hi = np.percentile(h, clip_high)
        h = h[(h >= lo) & (h <= hi)]
    if h.size == 0:
        return None, None, 0, raw_stats
    mu = float(h.mean())
    tau = float(h.std(ddof=0))
    return mu, max(tau, float(min_tau)), int(h.size), raw_stats


def main():
    parser = argparse.ArgumentParser(
        description="Optimize per-class Gaussian height priors (mu, tau) for MCD GT.")
    parser.add_argument("--config",
                        default=os.path.join(SCRIPT_DIR, "config/methods/mcd.yaml"))
    parser.add_argument("--output",
                        default=os.path.join(SCRIPT_DIR,
                                             "config/datasets/height_gaussians_mcd.yaml"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-scans", type=int, default=50000)
    parser.add_argument("--keyframe-dist", type=float, default=1.0)
    parser.add_argument("--sensor-mounting-height", type=float, default=None,
                        help="Override; defaults to cfg['sensor_mounting_height'] or 1.73.")
    parser.add_argument("--z-clip-low", type=float, default=1.0,
                        help="Per-scan percentile clip on h (stabilize against stray reflections).")
    parser.add_argument("--z-clip-high", type=float, default=99.0)
    parser.add_argument("--class-clip-low", type=float, default=1.0,
                        help="Per-class percentile clip applied before computing mu/tau.")
    parser.add_argument("--class-clip-high", type=float, default=99.0)
    parser.add_argument("--max-points-per-class", type=int, default=5_000_000,
                        help="Cap stored samples per class (0 = unlimited).")
    parser.add_argument("--min-tau", type=float, default=0.3,
                        help="Lower bound on tau (m) to avoid degenerate zero-width Gaussians.")
    parser.add_argument("--unlabeled-tau", type=float, default=100.0,
                        help="Override tau for class 0 (unlabeled). 0 disables the override.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = raw_cfg
    if "/**" in cfg:
        cfg = cfg["/**"].get("ros__parameters", cfg)
    elif "ros__parameters" in cfg:
        cfg = cfg["ros__parameters"]

    sensor_mounting_height = (args.sensor_mounting_height
                              if args.sensor_mounting_height is not None
                              else float(cfg.get("sensor_mounting_height", 1.73)))
    print(f"Using sensor_mounting_height = {sensor_mounting_height} m "
          f"(ground_z = origin_z - this)")

    data_dir = resolve_paths(cfg, args.data_dir)
    pose_file = os.path.join(data_dir, cfg["lidar_pose_file"])
    gt_label_dir = os.path.join(data_dir, cfg["gt_label_prefix"])
    scan_dir = os.path.join(data_dir, cfg["input_data_prefix"])
    calib_file = os.path.join(data_dir, "hhs_calib.yaml")
    gt_labels_key = cfg.get("gt_labels_key", "mcd")
    ds_resolution = cfg.get("ds_resolution", 1.0)

    labels_common_path = os.path.join(SCRIPT_DIR, "config/datasets/labels_common.yaml")
    gt_mapping = load_label_mappings(labels_common_path, gt_labels_key)

    print(f"Loading poses from {pose_file}")
    poses = load_poses(pose_file)
    print(f"  Loaded {len(poses)} poses")
    if not poses:
        raise RuntimeError(f"No poses parsed from {pose_file}")

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

    keyframe_dist = (args.keyframe_dist
                     if args.keyframe_dist is not None
                     else cfg.get("keyframe_dist", 0.0))
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
    print(f"Selected {len(scan_list)} scans (keyframe_dist={keyframe_dist} m)")
    if not scan_list:
        raise RuntimeError("No scans matched — check paths and keyframe_dist.")

    per_class_samples = [list() for _ in range(N_CLASSES)]

    for (idx, T, scan_path, label_path) in tqdm(scan_list, desc="scans", unit="scan"):
        pts = read_scan_bin(scan_path)
        labels_raw = read_label_bin(label_path)
        n = min(len(pts), len(labels_raw))
        if n == 0:
            continue
        pts, labels_raw = pts[:n], labels_raw[:n]
        gt_common = np.array([gt_mapping.get(int(l), 0) for l in labels_raw],
                             dtype=np.int32)

        lidar_to_map = T @ lidar_to_body
        xyz_h = np.hstack([pts[:, :3], np.ones((n, 1), dtype=np.float32)])
        map_pts = (lidar_to_map @ xyz_h.T).T[:, :3]

        if ds_resolution > 0:
            vkeys = np.floor(map_pts / ds_resolution).astype(np.int64)
            _, uidx = np.unique(vkeys, axis=0, return_index=True)
            map_pts = map_pts[uidx]
            gt_common = gt_common[uidx]

        # Runtime-matching height reference.
        origin_z = float(lidar_to_map[2, 3])
        ground_z = origin_z - sensor_mounting_height
        h = (map_pts[:, 2].astype(np.float64) - ground_z).astype(np.float32)

        if h.size >= 4:
            lo = float(np.percentile(h, args.z_clip_low))
            hi = float(np.percentile(h, args.z_clip_high))
            keep = (h >= lo) & (h <= hi)
            h = h[keep]
            gt_common = gt_common[keep]

        for c in range(N_CLASSES):
            mask = gt_common == c
            if not mask.any():
                continue
            reservoir_append(per_class_samples[c], h[mask],
                             args.max_points_per_class, rng)

    # Finalize per-class stats.
    mu = [0.0] * N_CLASSES
    tau = [float(args.min_tau)] * N_CLASSES
    diagnostics = {}

    for c in range(N_CLASSES):
        h_all = (np.concatenate(per_class_samples[c])
                 if per_class_samples[c] else np.empty(0, dtype=np.float32))
        m_val, t_val, n_used, raw_stats = robust_mu_tau(
            h_all, args.class_clip_low, args.class_clip_high, args.min_tau)

        entry = {
            **raw_stats,
            "count_used": int(n_used),
            "mu": None if m_val is None else float(m_val),
            "tau": None if t_val is None else float(t_val),
        }
        if m_val is not None:
            mu[c] = float(m_val)
            tau[c] = float(t_val)
        else:
            entry["note"] = "no GT samples — using defaults (mu=0, tau=min_tau)"
        diagnostics[CLASS_NAMES[c]] = entry

    # Unlabeled override: wide tau so the kernel is effectively a no-op there.
    if args.unlabeled_tau > 0:
        tau[0] = float(args.unlabeled_tau)
        diagnostics[CLASS_NAMES[0]]["tau"] = float(args.unlabeled_tau)
        diagnostics[CLASS_NAMES[0]]["note"] = (
            f"tau overridden via --unlabeled-tau={args.unlabeled_tau}")

    out = {
        "dataset": "mcd",
        "sequence_name": cfg.get("sequence_name"),
        "num_scans": len(scan_list),
        "keyframe_dist_m": float(keyframe_dist),
        "ds_resolution_m": float(ds_resolution),
        "sensor_mounting_height": float(sensor_mounting_height),
        "lidar_up_ref": lidar_up_ref.tolist(),
        "height_reference": "z_map - (origin_z - sensor_mounting_height)  # matches runtime",
        "outlier_clip": {
            "z_clip_low_pct": args.z_clip_low,
            "z_clip_high_pct": args.z_clip_high,
            "class_clip_low_pct": args.class_clip_low,
            "class_clip_high_pct": args.class_clip_high,
        },
        "min_tau": float(args.min_tau),
        "unlabeled_tau_override": float(args.unlabeled_tau),
        "class_names": list(CLASS_NAMES),
        # Ready-to-paste into config/methods/mcd.yaml under ros__parameters:
        "height_kernel_mu": [round(v, 4) for v in mu],
        "height_kernel_tau": [round(v, 4) for v in tau],
        "per_class": diagnostics,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print(f"\nWrote per-class Gaussian height priors to {args.output}")
    print(f"  height_kernel_mu:  {out['height_kernel_mu']}")
    print(f"  height_kernel_tau: {out['height_kernel_tau']}")
    for name, e in diagnostics.items():
        note = f"  [{e.get('note')}]" if e.get("note") else ""
        mu_s = "—" if e["mu"] is None else f"{e['mu']:+.2f}"
        tau_s = "—" if e["tau"] is None else f"{e['tau']:.2f}"
        print(f"  {name:12s} mu={mu_s}  tau={tau_s}  n={e['count_used']}{note}")


if __name__ == "__main__":
    main()
