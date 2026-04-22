#!/usr/bin/env python3
"""Estimate per-class height kernel parameters (mu, tau, dead_zone) via Dirichlet posterior.

Instead of median + MAD, we:
  1. Bin the height axis into fixed-width bins.
  2. Count per-class point hits per bin → multinomial observations.
  3. Add a symmetric Dirichlet prior (alpha_0) → posterior: alpha_k = alpha_0 + count_k.
  4. Normalize to get a posterior probability mass function p_k over height bins.

From p_k we derive per-class:
  mu        = posterior weighted mean of bin centers
  tau       = posterior weighted standard deviation
  dead_zone = half-width of the highest density region (HDR) covering `--hdr-mass` of
              the posterior mass. This is the smallest contiguous interval [mu-dz, mu+dz]
              containing the requested probability mass — i.e. where the class reliably
              appears. phi = 1 inside this interval; Gaussian decay outside.

All three parameters are per-class, data-driven, and coherent under the same posterior.

Height reference matches the C++ runtime (apply_height_kernel_to_ybars):
    h = z_map - (origin_z - sensor_mounting_height)

Output: config/datasets/height_dirichlet_mcd.yaml with ready-to-paste lists for
height_kernel_mu, height_kernel_tau, height_kernel_dead_zone (per-class vector),
plus full per-class diagnostics and a posterior plot (--plot).
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

# Force inline (flow) style for the three kernel parameter lists in the output YAML.
class FlowList(list):
    pass

def _flow_list_representer(dumper, data):
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=True)

yaml.add_representer(FlowList, _flow_list_representer)

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


def dirichlet_posterior(counts, alpha_0):
    """Return normalized posterior PMF given bin counts and symmetric prior alpha_0."""
    alpha = counts.astype(np.float64) + alpha_0
    return alpha / alpha.sum()


def hdr_half_width(bin_centers, pmf, mu, hdr_mass):
    """Compute the half-width of the smallest interval around mu covering hdr_mass.

    Expands symmetrically from mu outward, accumulating pmf mass, until the
    requested mass fraction is reached. Returns the half-width in meters.
    """
    # Sort bins by distance from mu, accumulate mass greedily.
    dists = np.abs(bin_centers - mu)
    order = np.argsort(dists)
    cumulative = 0.0
    max_dist = 0.0
    for i in order:
        cumulative += pmf[i]
        max_dist = max(max_dist, dists[i])
        if cumulative >= hdr_mass:
            break
    return float(max_dist)


def fit_class(h_all, bin_edges, bin_centers, alpha_0, hdr_mass, min_tau, min_dz):
    """Fit Dirichlet posterior and derive mu, tau, dead_zone for one class."""
    if h_all.size == 0:
        return None, None, None, {}

    counts, _ = np.histogram(h_all, bins=bin_edges)
    pmf = dirichlet_posterior(counts, alpha_0)

    mu  = float(np.sum(pmf * bin_centers))
    var = float(np.sum(pmf * (bin_centers - mu) ** 2))
    tau = max(float(np.sqrt(var)), min_tau)
    dz  = max(hdr_half_width(bin_centers, pmf, mu, hdr_mass), min_dz)

    diag = {
        "count": int(h_all.size),
        "mu": round(mu, 4),
        "tau": round(tau, 4),
        "dead_zone": round(dz, 4),
        "hdr_mass": hdr_mass,
        "alpha_0": alpha_0,
        "posterior_peak_h": round(float(bin_centers[np.argmax(pmf)]), 4),
    }
    return mu, tau, dz, diag


def main():
    parser = argparse.ArgumentParser(
        description="Per-class height kernel parameters via Dirichlet posterior.")
    parser.add_argument("--config",
                        default=os.path.join(SCRIPT_DIR, "config/methods/mcd.yaml"))
    parser.add_argument("--output",
                        default=os.path.join(SCRIPT_DIR,
                                             "config/datasets/height_dirichlet_mcd.yaml"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--max-scans", type=int, default=50000)
    parser.add_argument("--keyframe-dist", type=float, default=1.0)
    parser.add_argument("--sensor-mounting-height", type=float, default=None)
    parser.add_argument("--h-min", type=float, default=-10.0,
                        help="Minimum height (m) for histogram bins.")
    parser.add_argument("--h-max", type=float, default=15.0,
                        help="Maximum height (m) for histogram bins.")
    parser.add_argument("--bin-width", type=float, default=0.1,
                        help="Bin width (m) for height histogram.")
    parser.add_argument("--alpha-0", type=float, default=0.01,
                        help="Symmetric Dirichlet prior on each bin (smoothing). "
                             "Small = data-driven; large = uniform skepticism.")
    parser.add_argument("--hdr-mass", type=float, default=0.5,
                        help="Posterior mass fraction for the HDR dead zone (0-1). "
                             "0.5 = smallest interval covering 50%% of posterior mass.")
    parser.add_argument("--min-tau", type=float, default=0.3,
                        help="Minimum tau (m).")
    parser.add_argument("--min-dz", type=float, default=0.0,
                        help="Minimum dead zone half-width (m).")
    parser.add_argument("--unlabeled-tau", type=float, default=100.0,
                        help="Override tau for class 0 (unlabeled). 0 disables.")
    parser.add_argument("--unlabeled-dz", type=float, default=0.0,
                        help="Override dead_zone for class 0. 0 disables.")
    parser.add_argument("--max-points-per-class-per-scan", type=int, default=50000)
    parser.add_argument("--max-points-per-class", type=int, default=5_000_000)
    parser.add_argument("--z-clip-low", type=float, default=1.0)
    parser.add_argument("--z-clip-high", type=float, default=99.0)
    parser.add_argument("--plot", action="store_true",
                        help="Save per-class posterior plots to height_dirichlet_plots/.")
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
    print(f"sensor_mounting_height = {sensor_mounting_height} m")

    data_dir = resolve_paths(cfg, args.data_dir)
    pose_file    = os.path.join(data_dir, cfg["lidar_pose_file"])
    gt_label_dir = os.path.join(data_dir, cfg["gt_label_prefix"])
    scan_dir     = os.path.join(data_dir, cfg["input_data_prefix"])
    calib_file   = os.path.join(data_dir, "hhs_calib.yaml")
    gt_labels_key = cfg.get("gt_labels_key", "mcd")
    ds_resolution = cfg.get("ds_resolution", 1.0)

    labels_common_path = os.path.join(SCRIPT_DIR, "config/datasets/labels_common.yaml")
    gt_mapping = load_label_mappings(labels_common_path, gt_labels_key)

    print(f"Loading poses from {pose_file}")
    poses = load_poses(pose_file)
    print(f"  {len(poses)} poses loaded")
    if not poses:
        raise RuntimeError(f"No poses parsed from {pose_file}")

    first_inv = np.linalg.inv(poses[0][1])
    for i in range(len(poses)):
        poses[i] = (poses[i][0], first_inv @ poses[i][1])

    body_to_lidar = load_calibration(calib_file)
    lidar_to_body = np.linalg.inv(body_to_lidar)

    keyframe_dist = args.keyframe_dist
    scan_list, last_kf = [], None
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
    print(f"Selected {len(scan_list)} scans")
    if not scan_list:
        raise RuntimeError("No scans matched — check paths and keyframe_dist.")

    per_class_samples = [[] for _ in range(N_CLASSES)]

    for (idx, T, scan_path, label_path) in tqdm(scan_list, desc="scans", unit="scan"):
        pts        = read_scan_bin(scan_path)
        labels_raw = read_label_bin(label_path)
        n = min(len(pts), len(labels_raw))
        if n == 0:
            continue
        pts, labels_raw = pts[:n], labels_raw[:n]
        gt_common = np.array([gt_mapping.get(int(l), 0) for l in labels_raw], dtype=np.int32)

        lidar_to_map = T @ lidar_to_body
        xyz_h  = np.hstack([pts[:, :3], np.ones((n, 1), dtype=np.float32)])
        map_pts = (lidar_to_map @ xyz_h.T).T[:, :3]

        if ds_resolution > 0:
            vkeys = np.floor(map_pts / ds_resolution).astype(np.int64)
            _, uidx = np.unique(vkeys, axis=0, return_index=True)
            map_pts   = map_pts[uidx]
            gt_common = gt_common[uidx]

        origin_z = float(lidar_to_map[2, 3])
        ground_z = origin_z - sensor_mounting_height
        h = (map_pts[:, 2].astype(np.float64) - ground_z).astype(np.float32)

        if h.size >= 4:
            lo = float(np.percentile(h, args.z_clip_low))
            hi = float(np.percentile(h, args.z_clip_high))
            keep = (h >= lo) & (h <= hi)
            h, gt_common = h[keep], gt_common[keep]

        for c in range(N_CLASSES):
            mask = gt_common == c
            if not mask.any():
                continue
            h_c = h[mask]
            cap = args.max_points_per_class_per_scan
            if cap > 0 and h_c.size > cap:
                h_c = h_c[rng.choice(h_c.size, size=cap, replace=False)]
            reservoir_append(per_class_samples[c], h_c, args.max_points_per_class, rng)

    # Build histogram bins
    bin_edges   = np.arange(args.h_min, args.h_max + args.bin_width, args.bin_width)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    mu_list, tau_list, dz_list = [], [], []
    diagnostics = {}

    for c in range(N_CLASSES):
        h_all = (np.concatenate(per_class_samples[c])
                 if per_class_samples[c] else np.empty(0, dtype=np.float32))
        mu, tau, dz, diag = fit_class(
            h_all, bin_edges, bin_centers,
            args.alpha_0, args.hdr_mass, args.min_tau, args.min_dz)

        if mu is None:
            mu, tau, dz = 0.0, args.min_tau, 0.0
            diag = {"count": 0, "note": "no GT samples — defaults used"}

        mu_list.append(round(mu, 4))
        tau_list.append(round(tau, 4))
        dz_list.append(round(dz, 4))
        diagnostics[CLASS_NAMES[c]] = diag

    # Unlabeled overrides
    if args.unlabeled_tau > 0:
        tau_list[0] = float(args.unlabeled_tau)
        diagnostics[CLASS_NAMES[0]]["tau"] = float(args.unlabeled_tau)
        diagnostics[CLASS_NAMES[0]].setdefault("note", "")
        diagnostics[CLASS_NAMES[0]]["note"] += f" tau overridden={args.unlabeled_tau}"
    if args.unlabeled_dz > 0:
        dz_list[0] = float(args.unlabeled_dz)
        diagnostics[CLASS_NAMES[0]]["dead_zone"] = float(args.unlabeled_dz)

    out = {
        "dataset": "mcd",
        "sequence_name": cfg.get("sequence_name"),
        "num_scans": len(scan_list),
        "method": "dirichlet_posterior",
        "alpha_0": args.alpha_0,
        "hdr_mass": args.hdr_mass,
        "bin_width_m": args.bin_width,
        "h_range_m": [args.h_min, args.h_max],
        "sensor_mounting_height": sensor_mounting_height,
        "class_names": list(CLASS_NAMES),
        "height_kernel_mu":        FlowList(mu_list),
        "height_kernel_tau":       FlowList(tau_list),
        "height_kernel_dead_zone": FlowList(dz_list),
        "per_class": diagnostics,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(out, f, sort_keys=False, allow_unicode=True)

    print(f"\nWrote to {args.output}")
    print(f"  height_kernel_mu:        {mu_list}")
    print(f"  height_kernel_tau:       {tau_list}")
    print(f"  height_kernel_dead_zone: {dz_list}")
    for name, d in diagnostics.items():
        note = f"  [{d.get('note')}]" if d.get("note") else ""
        mu_s  = "—" if d.get("mu")        is None else f"{d['mu']:+.2f}"
        tau_s = "—" if d.get("tau")       is None else f"{d['tau']:.2f}"
        dz_s  = "—" if d.get("dead_zone") is None else f"{d['dead_zone']:.2f}"
        print(f"  {name:12s}  mu={mu_s}  tau={tau_s}  dz={dz_s}  n={d.get('count', 0)}{note}")

    if args.plot:
        import matplotlib.pyplot as plt
        plot_dir = os.path.join(os.path.dirname(args.output), "height_dirichlet_plots")
        os.makedirs(plot_dir, exist_ok=True)
        for c in range(N_CLASSES):
            h_all = (np.concatenate(per_class_samples[c])
                     if per_class_samples[c] else np.empty(0, dtype=np.float32))
            if h_all.size == 0:
                continue
            counts, _ = np.histogram(h_all, bins=bin_edges)
            pmf = dirichlet_posterior(counts, args.alpha_0)
            mu  = mu_list[c]
            tau = tau_list[c]
            dz  = dz_list[c]

            h_plot = np.linspace(args.h_min, args.h_max, 1000)
            phi = np.ones_like(h_plot)
            for i, hi in enumerate(h_plot):
                excess = max(0.0, abs(hi - mu) - dz)
                phi[i] = np.exp(-(excess ** 2) / (2 * tau ** 2))

            fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)

            axes[0].bar(bin_centers, pmf, width=args.bin_width * 0.9,
                        color='steelblue', alpha=0.7, label='Posterior PMF')
            axes[0].axvline(mu, color='red',   linestyle='--', linewidth=1.5, label=f'μ = {mu:.2f} m')
            axes[0].axvspan(mu - dz, mu + dz, alpha=0.15, color='green', label=f'dead zone ±{dz:.2f} m')
            axes[0].set_ylabel('Posterior probability')
            axes[0].set_title(f'{CLASS_NAMES[c]} — Dirichlet posterior  (τ={tau:.2f} m, dz=±{dz:.2f} m)')
            axes[0].legend(fontsize=8)
            axes[0].grid(True, alpha=0.3)

            axes[1].plot(h_plot, phi, color='darkorange', linewidth=2, label='φ(h)')
            axes[1].axvline(mu, color='red',   linestyle='--', linewidth=1.5)
            axes[1].axvspan(mu - dz, mu + dz, alpha=0.15, color='green')
            axes[1].set_ylabel('φ(h)  [kernel suppression]')
            axes[1].set_xlabel('Height above estimated ground (m)')
            axes[1].set_ylim(0, 1.05)
            axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            fname = os.path.join(plot_dir, f"height_posterior_{CLASS_NAMES[c]}.png")
            plt.savefig(fname, dpi=130)
            plt.close()
            print(f"  saved {fname}")


if __name__ == "__main__":
    main()
