#!/usr/bin/env python3
"""Step 4: Height consistency kernel.

Computes k_ht(h_i, y_i) for each point based on its height above ground
and semantic prediction. Points at implausible heights for their predicted
class get downweighted.

k_ht(h_i, y_i) = (1 - λ) + λ * Σ_k (y_i^k * φ_k(h_i))

where φ_k(h) = exp(-(h - μ_k)² / (2 * τ_k²))

Usage (inside container):
    python3 /ros2_ws/src/osm_bki/height_kernel/height_kernel.py
"""

import os

import numpy as np
import yaml

from dem_query import DEMQuery


# Default height priors: (μ_k, τ_k) per common-taxonomy class index
# These match the 13-class common taxonomy in labels_common.yaml
# Index 0 = unlabeled/ignore
DEFAULT_HEIGHT_PRIORS = {
    0:  {"name": "unlabeled",   "mu": 0.0,  "tau": 100.0},  # no constraint
    1:  {"name": "road",        "mu": 0.0,  "tau": 0.5},
    2:  {"name": "sidewalk",    "mu": 0.0,  "tau": 0.5},
    3:  {"name": "parking",     "mu": 0.0,  "tau": 0.5},
    4:  {"name": "other-ground","mu": 0.0,  "tau": 1.0},
    5:  {"name": "building",    "mu": 8.0,  "tau": 15.0},
    6:  {"name": "fence",       "mu": 1.0,  "tau": 1.5},
    7:  {"name": "pole",        "mu": 3.0,  "tau": 4.0},
    8:  {"name": "traffic-sign","mu": 2.5,  "tau": 3.0},
    9:  {"name": "terrain",     "mu": 0.0,  "tau": 0.5},
    10: {"name": "two-wheeler", "mu": 0.8,  "tau": 1.0},
    11: {"name": "vehicle",     "mu": 0.8,  "tau": 1.0},
    12: {"name": "other-object","mu": 1.0,  "tau": 2.0},
    13: {"name": "vegetation",  "mu": 8.0,  "tau": 5.0},
}


class HeightKernel:
    """Height consistency kernel for BKI fusion."""

    def __init__(self, dem_query, height_priors=None, lam=0.5):
        """
        Args:
            dem_query: DEMQuery instance for looking up ground elevation
            height_priors: dict mapping class_index → {"mu": float, "tau": float}
                           If None, uses DEFAULT_HEIGHT_PRIORS
            lam: λ ∈ [0,1], strength of height prior (0 = disabled, 1 = full gating)
        """
        self.dq = dem_query
        self.lam = lam
        self.priors = height_priors or DEFAULT_HEIGHT_PRIORS

        # Pre-compute arrays for vectorized computation
        max_class = max(self.priors.keys()) + 1
        self.mu = np.zeros(max_class)
        self.tau = np.ones(max_class) * 100.0  # default: no constraint
        for k, v in self.priors.items():
            self.mu[k] = v["mu"]
            self.tau[k] = v["tau"]

    def phi(self, h, class_idx):
        """Per-class height support: φ_k(h) = exp(-(h - μ_k)² / (2τ_k²))

        Args:
            h: (N,) height above ground
            class_idx: int, class index

        Returns:
            (N,) support values in [0, 1]
        """
        mu_k = self.mu[class_idx]
        tau_k = self.tau[class_idx]
        return np.exp(-((h - mu_k) ** 2) / (2 * tau_k ** 2))

    def phi_all(self, h):
        """Compute φ_k(h) for all classes at once.

        Args:
            h: (N,) height above ground

        Returns:
            (N, K) array of support values
        """
        K = len(self.mu)
        h_col = h[:, np.newaxis]  # (N, 1)
        mu = self.mu[np.newaxis, :]  # (1, K)
        tau = self.tau[np.newaxis, :]  # (1, K)
        return np.exp(-((h_col - mu) ** 2) / (2 * tau ** 2))

    def compute(self, xyz, y):
        """Compute height kernel weights for a batch of points.

        Args:
            xyz: (N, 3) points in KITTI-360 local frame
            y: (N, K) semantic predictions (softmax probabilities or one-hot)

        Returns:
            (N,) kernel weights k_ht ∈ [1-λ, 1]
        """
        result = self.dq.query(xyz)
        h = result["h_above_ground"]  # (N,)
        valid = result["valid"]  # (N,)

        N = len(xyz)
        K = y.shape[1]
        weights = np.ones(N)

        if not valid.any():
            return weights

        # For valid points, compute k_ht
        h_valid = h[valid]
        y_valid = y[valid, :min(K, len(self.mu))]

        # φ_k(h) for all classes: (N_valid, K)
        phi_all = self.phi_all(h_valid)[:, :min(K, len(self.mu))]

        # Prediction-weighted sum: Σ_k y_i^k * φ_k(h_i)
        weighted_phi = np.sum(y_valid * phi_all, axis=1)  # (N_valid,)

        # k_ht = (1 - λ) + λ * weighted_phi
        k_ht = (1.0 - self.lam) + self.lam * weighted_phi
        weights[valid] = k_ht

        return weights

    @staticmethod
    def load_config(config_path):
        """Load height priors and lambda from a YAML config file."""
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        priors = {}
        for entry in cfg.get("height_priors", []):
            priors[entry["class_index"]] = {
                "name": entry.get("name", ""),
                "mu": float(entry["mu"]),
                "tau": float(entry["tau"]),
            }
        lam = float(cfg.get("lambda", 0.5))
        return priors, lam


def main():
    """Test: compute height kernel for the first LiDAR scan."""
    SEQUENCE_DIR = "/media/sgarimella34/hercules-collect1/datasets/kitti360/2013_05_28_drive_0000_sync"
    TRANSFORM_FILE = os.path.join(SEQUENCE_DIR, "kitti360_to_utm.npz")
    DEM_PATH = "/media/sgarimella34/hercules-collect1/kitti360_DGM025/merged_dem.tif"
    DSM_PATH = "/media/sgarimella34/hercules-collect1/kitti360_DOM1/merged_dsm.tif"
    SCAN_DIR = os.path.join(SEQUENCE_DIR, "velodyne_points", "data")
    LABEL_DIR = os.path.join(SEQUENCE_DIR, "inferred_labels", "cenet_mcd_softmax")
    POSE_FILE = os.path.join(SEQUENCE_DIR, "velodyne_poses.txt")

    # Load first pose (raw) to get first_t for local frame shift
    with open(POSE_FILE) as f:
        first_line = f.readline().strip().split()
    first_floats = [float(v) for v in first_line[1:]]
    if len(first_floats) >= 16:
        first_mat = np.array(first_floats[:16]).reshape(4, 4)
    else:
        first_mat = np.eye(4)
        first_mat[:3, :] = np.array(first_floats[:12]).reshape(3, 4)
    first_t = first_mat[:3, 3]
    first_R = first_mat[:3, :3]

    # Load first scan
    scan_files = sorted(os.listdir(SCAN_DIR))
    scan_path = os.path.join(SCAN_DIR, scan_files[0])
    points = np.fromfile(scan_path, dtype=np.float32).reshape(-1, 4)[:, :3]
    print(f"Loaded scan: {scan_files[0]} ({len(points)} points)")

    # Transform points to local frame (apply first pose, then shift to origin)
    # points are in lidar frame; pose transforms lidar → world
    points_world = (first_R @ points.T).T + first_t
    points_local = points_world - first_t  # = first_R @ points.T).T

    # Load corresponding label file (softmax scores)
    label_files = sorted(os.listdir(LABEL_DIR))
    label_path = os.path.join(LABEL_DIR, label_files[0])
    # Labels are float16 softmax scores, shape (N, num_classes)
    labels_raw = np.fromfile(label_path, dtype=np.float16)
    num_points = len(points)
    num_classes = len(labels_raw) // num_points
    labels = labels_raw.reshape(num_points, num_classes).astype(np.float32)
    print(f"Loaded labels: {label_files[0]} ({num_classes} classes)")

    with DEMQuery(TRANSFORM_FILE, DEM_PATH, DSM_PATH) as dq:
        hk = HeightKernel(dq, lam=0.5)
        weights = hk.compute(points_local, labels)

        print(f"\nHeight kernel weights (λ={hk.lam}):")
        print(f"  Mean:  {weights.mean():.4f}")
        print(f"  Std:   {weights.std():.4f}")
        print(f"  Min:   {weights.min():.4f}")
        print(f"  Max:   {weights.max():.4f}")

        # Show distribution of weights
        bins = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0, 1.001]
        hist, _ = np.histogram(weights, bins=bins)
        print(f"\n  Weight distribution:")
        for i in range(len(bins) - 1):
            print(f"    [{bins[i]:.2f}, {bins[i+1]:.3f}): {hist[i]:>6d} points ({100*hist[i]/len(weights):.1f}%)")

        # Show per-class breakdown for top predicted class
        pred_class = labels.argmax(axis=1)
        print(f"\n  Per-class mean weight (top predicted class):")
        for c in range(min(13, num_classes)):
            mask = pred_class == c
            if mask.sum() > 0:
                name = DEFAULT_HEIGHT_PRIORS.get(c, {}).get("name", f"class_{c}")
                print(f"    {c:2d} ({name:>12s}): {weights[mask].mean():.4f}  (n={mask.sum():>5d})")


if __name__ == "__main__":
    main()
