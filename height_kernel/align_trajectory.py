#!/usr/bin/env python3
"""Step 2: Compute the transform from KITTI-360 local frame to UTM 32N (EPSG:25832).

KITTI-360 uses a scaled Mercator projection centered at a reference lat/lon.
The poses in velodyne_poses.txt are shifted so the first pose is at the origin.

This script:
1. Loads KITTI-360 poses
2. Reconstructs absolute Mercator coordinates (add back first-pose translation)
3. Inverts the Mercator projection to recover lat/lon for each pose
4. Converts lat/lon → UTM 32N
5. Computes and saves the affine transform from KITTI-360 local → UTM 32N
6. Plots the trajectory on top of the merged DEM for visual verification

Usage (inside container):
    python3 /ros2_ws/src/osm_bki/height_kernel/align_trajectory.py
"""

import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
from rasterio.plot import show as rioshow

# --- Config ---
DATASET_DIR = "/media/sgarimella34/hercules-collect1/datasets/kitti360"
SEQUENCE = os.environ.get("KITTI360_SEQUENCE", "2013_05_28_drive_0000_sync")
POSE_FILE = os.path.join(DATASET_DIR, SEQUENCE, "velodyne_poses.txt")

# KITTI-360 Mercator origin (from kitti360.yaml)
ORIGIN_LAT = 48.9843445
ORIGIN_LON = 8.4295857
EARTH_RADIUS = 6378137.0

DEM_PATH = "/media/sgarimella34/hercules-collect1/kitti360_DGM025/merged_dem.tif"
OUTPUT_DIR = os.path.join(DATASET_DIR, SEQUENCE)
TRANSFORM_FILE = os.path.join(OUTPUT_DIR, "kitti360_to_utm.npz")
PLOT_FILE = os.path.join(OUTPUT_DIR, "trajectory_on_dem.png")


def load_kitti360_poses(pose_file):
    """Load poses from velodyne_poses.txt. Returns (frame_indices, 4x4 matrices)."""
    frames = []
    poses = []
    with open(pose_file) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 13:
                continue
            frame_idx = int(vals[0])
            floats = [float(v) for v in vals[1:]]
            if len(floats) == 12:
                mat = np.eye(4)
                mat[:3, :] = np.array(floats).reshape(3, 4)
            elif len(floats) >= 16:
                mat = np.array(floats[:16]).reshape(4, 4)
            else:
                continue
            frames.append(frame_idx)
            poses.append(mat)
    return np.array(frames), np.array(poses)


def mercator_origin():
    """Compute the Mercator origin (mx0, my0) for the KITTI-360 reference lat/lon."""
    scale = math.cos(math.radians(ORIGIN_LAT))
    lon_rad = math.radians(ORIGIN_LON)
    lat_rad = math.radians(ORIGIN_LAT)
    mx0 = scale * lon_rad * EARTH_RADIUS
    my0 = scale * EARTH_RADIUS * math.log(math.tan(math.pi / 4.0 + lat_rad / 2.0))
    return mx0, my0, scale


def mercator_xy_to_latlon(mx, my, scale):
    """Invert the scaled Mercator projection: (mx, my) → (lat, lon)."""
    lon_rad = mx / (scale * EARTH_RADIUS)
    lat_rad = 2.0 * math.atan(math.exp(my / (scale * EARTH_RADIUS))) - math.pi / 2.0
    return math.degrees(lat_rad), math.degrees(lon_rad)


def main():
    # 1. Load poses
    print(f"Loading poses from {POSE_FILE}")
    frames, poses = load_kitti360_poses(POSE_FILE)
    print(f"  Loaded {len(frames)} poses")

    # Extract translations (these are already shifted: first pose at origin)
    translations = poses[:, :3, 3]  # (N, 3)
    first_pose_t = poses[0, :3, 3].copy()  # should be ~[0,0,0] since shifted

    # The original (unshifted) first pose translation is stored in the raw file
    # Re-read the first line to get original absolute coords
    with open(POSE_FILE) as f:
        first_line = f.readline().strip().split()
    first_floats = [float(v) for v in first_line[1:]]
    if len(first_floats) >= 16:
        orig_first = np.array(first_floats[:16]).reshape(4, 4)
    else:
        orig_first = np.eye(4)
        orig_first[:3, :] = np.array(first_floats[:12]).reshape(3, 4)
    orig_first_t = orig_first[:3, 3]
    print(f"  Original first pose translation: [{orig_first_t[0]:.3f}, {orig_first_t[1]:.3f}, {orig_first_t[2]:.3f}]")

    # NOTE: The C++ code shifts all poses by subtracting first_t.
    # The raw file has absolute Mercator coords (not shifted).
    # So the raw translations ARE the absolute Mercator coords relative to origin.
    # We just use them directly without re-adding anything.

    # 2. Compute Mercator origin
    mx0, my0, scale = mercator_origin()
    print(f"  Mercator origin: mx0={mx0:.3f}, my0={my0:.3f}, scale={scale:.6f}")

    # 3. Convert all raw poses from Mercator to lat/lon
    # Raw pose translations = (mx - mx0, my - my0, z) in Mercator frame
    # So absolute Mercator = raw_t + (mx0, my0, 0)
    # But we need to re-read ALL raw translations from file (not the C++-shifted ones)
    raw_translations = np.zeros((len(frames), 3))
    with open(POSE_FILE) as f:
        for i, line in enumerate(f):
            vals = line.strip().split()
            if len(vals) < 13:
                continue
            floats = [float(v) for v in vals[1:]]
            if len(floats) >= 16:
                mat = np.array(floats[:16]).reshape(4, 4)
            elif len(floats) == 12:
                mat = np.eye(4)
                mat[:3, :] = np.array(floats[:12]).reshape(3, 4)
            raw_translations[i] = mat[:3, 3]

    # Raw translations are in absolute Mercator: (mx - mx0, my - my0, z)
    # Recover absolute Mercator coords
    abs_mx = raw_translations[:, 0] + mx0
    abs_my = raw_translations[:, 1] + my0

    # Convert to lat/lon
    lats = np.zeros(len(frames))
    lons = np.zeros(len(frames))
    for i in range(len(frames)):
        lats[i], lons[i] = mercator_xy_to_latlon(abs_mx[i], abs_my[i], scale)

    print(f"  Lat range: [{lats.min():.6f}, {lats.max():.6f}]")
    print(f"  Lon range: [{lons.min():.6f}, {lons.max():.6f}]")

    # 4. Convert lat/lon → UTM 32N (EPSG:25832)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:25832", always_xy=True)
    utm_e, utm_n = transformer.transform(lons, lats)
    print(f"  UTM easting range:  [{utm_e.min():.2f}, {utm_e.max():.2f}]")
    print(f"  UTM northing range: [{utm_n.min():.2f}, {utm_n.max():.2f}]")

    # 5. Compute affine transform: KITTI-360 local (shifted) → UTM
    # KITTI-360 local = raw_translations - orig_first_t (this is what the C++ code uses)
    local_xyz = raw_translations - orig_first_t  # first pose at origin
    utm_xyz = np.column_stack([utm_e, utm_n, raw_translations[:, 2]])

    # Compute offset + rotation using Umeyama (least-squares similarity transform)
    # For now, try simple offset first (translation only), since the Mercator is very close to UTM at this scale
    offset = utm_xyz[:, :2].mean(axis=0) - local_xyz[:, :2].mean(axis=0)
    residuals_simple = np.linalg.norm(
        (local_xyz[:, :2] + offset) - utm_xyz[:, :2], axis=1
    )
    print(f"\n  Simple translation offset: [{offset[0]:.3f}, {offset[1]:.3f}]")
    print(f"  Residuals (translation-only): mean={residuals_simple.mean():.3f}m, max={residuals_simple.max():.3f}m")

    # Full Umeyama (rigid: rotation + translation, no scale)
    T_umeyama = umeyama_rigid(local_xyz[:, :2], utm_xyz[:, :2])
    local_hom = np.column_stack([local_xyz[:, :2], np.ones(len(local_xyz))])
    utm_pred = (T_umeyama @ local_hom.T).T
    residuals_umeyama = np.linalg.norm(utm_pred - utm_xyz[:, :2], axis=1)
    print(f"\n  Umeyama rigid transform:")
    print(f"    R = [[{T_umeyama[0,0]:.8f}, {T_umeyama[0,1]:.8f}],")
    print(f"         [{T_umeyama[1,0]:.8f}, {T_umeyama[1,1]:.8f}]]")
    print(f"    t = [{T_umeyama[0,2]:.3f}, {T_umeyama[1,2]:.3f}]")
    print(f"  Residuals (Umeyama): mean={residuals_umeyama.mean():.3f}m, max={residuals_umeyama.max():.3f}m")

    # Also store z offset: DEM elevation vs KITTI z at the first pose
    utm_first_e, utm_first_n = utm_e[0], utm_n[0]

    # Save transform
    np.savez(
        TRANSFORM_FILE,
        T_umeyama=T_umeyama,
        origin_lat=ORIGIN_LAT,
        origin_lon=ORIGIN_LON,
        utm_first_easting=utm_first_e,
        utm_first_northing=utm_first_n,
        kitti_first_z=raw_translations[0, 2],
    )
    print(f"\n  Saved transform to {TRANSFORM_FILE}")

    # 6. Plot trajectory on DEM
    plot_trajectory_on_dem(utm_e, utm_n, DEM_PATH, PLOT_FILE)


def umeyama_rigid(src, dst):
    """2D rigid (rotation + translation) alignment: dst ≈ R @ src + t.
    Returns 2x3 affine matrix [R | t].
    """
    assert src.shape == dst.shape
    n = src.shape[0]

    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Cross-covariance
    H = src_c.T @ dst_c / n

    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, np.sign(d)])
    R = Vt.T @ D @ U.T
    t = mu_dst - R @ mu_src

    T = np.zeros((2, 3))
    T[:2, :2] = R
    T[:2, 2] = t
    return T


def plot_trajectory_on_dem(utm_e, utm_n, dem_path, output_path):
    """Plot trajectory overlaid on DEM hillshade."""
    if not os.path.exists(dem_path):
        print(f"  DEM not found at {dem_path}, skipping plot.")
        return

    with rasterio.open(dem_path) as src:
        # Read a window around the trajectory for efficiency
        bounds = src.bounds
        # Check trajectory is within DEM bounds
        if utm_e.min() < bounds.left or utm_e.max() > bounds.right or \
           utm_n.min() < bounds.bottom or utm_n.max() > bounds.top:
            print("  WARNING: Trajectory partially outside DEM bounds!")

        # Read full DEM (it's manageable at ~40k x 32k for float32)
        # Actually read a subset around the trajectory
        margin = 500  # meters
        from rasterio.windows import from_bounds
        window = from_bounds(
            max(utm_e.min() - margin, bounds.left),
            max(utm_n.min() - margin, bounds.bottom),
            min(utm_e.max() + margin, bounds.right),
            min(utm_n.max() + margin, bounds.top),
            transform=src.transform,
        )
        dem_data = src.read(1, window=window)
        win_transform = src.window_transform(window)

    # Simple hillshade
    from numpy import gradient
    dy, dx = gradient(dem_data)
    hillshade = np.clip((-dx + dy + 2) / 4, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    extent = [
        win_transform[2],
        win_transform[2] + dem_data.shape[1] * win_transform[0],
        win_transform[5] + dem_data.shape[0] * win_transform[4],
        win_transform[5],
    ]
    ax.imshow(hillshade, extent=extent, cmap="gray", origin="upper")
    ax.plot(utm_e, utm_n, "r-", linewidth=0.5, alpha=0.8, label="KITTI-360 trajectory")
    ax.plot(utm_e[0], utm_n[0], "go", markersize=8, label="Start")
    ax.plot(utm_e[-1], utm_n[-1], "bs", markersize=8, label="End")
    ax.set_xlabel("UTM Easting (m)")
    ax.set_ylabel("UTM Northing (m)")
    ax.set_title("KITTI-360 Trajectory on DEM (UTM 32N)")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"  Saved trajectory plot to {output_path}")


if __name__ == "__main__":
    main()
