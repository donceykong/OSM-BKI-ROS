#!/usr/bin/env python3
"""View the CU North Campus robot1 trajectory and accumulated lidar in PyVista.

Pose file format (mcd): scan #,timestamp,x,y,z,qx,qy,qz,w
Lidar bins: KITTI-style float32 (x, y, z, intensity), file name = %010d.bin
where the file index equals the pose row index (use_pose_index_as_scan_id: true).
See config/methods/cu_north_campus.yaml.
"""

import argparse
import os
import numpy as np
import pyvista as pv

DEFAULT_POSES = (
    "/media/donceykong/doncey_ssd_02/datasets/CU_MULTI/"
    "north_campus/robot1/poses/poses_interpolated.csv"
)
DEFAULT_LIDAR_DIR = (
    "/media/donceykong/doncey_ssd_02/datasets/CU_MULTI/"
    "north_campus/robot1/lidar_bin/data"
)

PATH_COLOR = [150 / 255, 200 / 255, 50 / 255]
POINT_COLOR_LOW = [0.5*80 / 255, 0.5*110 / 255, 0.5*30 / 255]
POINT_COLOR_HIGH = [0.5*230 / 255, 0.5*255 / 255, 0.5*160 / 255]

def load_poses(path):
    data = np.loadtxt(path, delimiter=",", skiprows=1)
    scan_ids = data[:, 0].astype(np.int64)
    xyz = data[:, 2:5]
    quat_xyzw = data[:, 5:9]
    return scan_ids, xyz, quat_xyzw


def quat_to_rot(q_xyzw):
    """Batched quaternion (xyzw) -> 3x3 rotation. Returns (N, 3, 3)."""
    qx, qy, qz, qw = q_xyzw[:, 0], q_xyzw[:, 1], q_xyzw[:, 2], q_xyzw[:, 3]
    R = np.empty((len(q_xyzw), 3, 3))
    R[:, 0, 0] = 1 - 2 * (qy * qy + qz * qz)
    R[:, 0, 1] = 2 * (qx * qy - qz * qw)
    R[:, 0, 2] = 2 * (qx * qz + qy * qw)
    R[:, 1, 0] = 2 * (qx * qy + qz * qw)
    R[:, 1, 1] = 1 - 2 * (qx * qx + qz * qz)
    R[:, 1, 2] = 2 * (qy * qz - qx * qw)
    R[:, 2, 0] = 2 * (qx * qz - qy * qw)
    R[:, 2, 1] = 2 * (qy * qz + qx * qw)
    R[:, 2, 2] = 1 - 2 * (qx * qx + qy * qy)
    return R


def voxel_downsample(pts, voxel_size):
    if voxel_size <= 0 or len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


def load_bin(path, max_range):
    raw = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    pts = raw[:, :3]
    intensity = raw[:, 3]
    if max_range > 0:
        r2 = (pts * pts).sum(axis=1)
        mask = r2 < max_range * max_range
        pts = pts[mask]
        intensity = intensity[mask]
    return pts, intensity


def select_keyframes(xyz_local, keyframe_dist):
    """Return indices of poses spaced at least keyframe_dist apart (>=0)."""
    if keyframe_dist <= 0:
        return np.arange(len(xyz_local))
    keep = [0]
    last = xyz_local[0]
    for i in range(1, len(xyz_local)):
        if np.linalg.norm(xyz_local[i] - last) >= keyframe_dist:
            keep.append(i)
            last = xyz_local[i]
    return np.array(keep, dtype=np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poses", default=DEFAULT_POSES, help="Path to poses_interpolated.csv")
    ap.add_argument("--lidar-dir", default=DEFAULT_LIDAR_DIR,
                    help="Directory containing %%010d.bin lidar scans")
    ap.add_argument("--stride", type=int, default=1, help="Subsample every N poses (path)")
    ap.add_argument("--keyframe-dist", type=float, default=2.0,
                    help="Min euclidean distance (m) between consecutive scans used in accumulation (0 = use every scan)")
    ap.add_argument("--voxel-size", type=float, default=0.3,
                    help="Per-scan voxel downsample size (m) before accumulation (0 = off)")
    ap.add_argument("--final-voxel-size", type=float, default=0.0,
                    help="Voxel downsample over the accumulated cloud (0 = off)")
    ap.add_argument("--max-range", type=float, default=80.0,
                    help="Drop points beyond this range (m) per scan (0 = off)")
    ap.add_argument("--max-scans", type=int, default=0,
                    help="Cap number of accumulated scans (0 = no cap)")
    ap.add_argument("--no-cloud", action="store_true", help="Skip lidar accumulation (path only)")
    ap.add_argument("--color-by", choices=["z", "intensity"], default="z",
                    help="Per-scan gradient source: body-frame Z height or LiDAR intensity")
    ap.add_argument("--intensity-gamma", type=float, default=0.5,
                    help="Gamma applied to normalized intensity (lower = brighten dim returns)")
    ap.add_argument("--cmap", default="",
                    help="matplotlib colormap (e.g. viridis, magma, gray, turbo). "
                         "If empty, interpolate between POINT_COLOR_LOW/HIGH")
    ap.add_argument("--line-width", type=float, default=15.0, help="Trajectory line width (px)")
    ap.add_argument("--point-size", type=float, default=2.0, help="Point cloud point size (px)")
    ap.add_argument("--marker-radius", type=float, default=2.0,
                    help="Start/end sphere radius (m)")
    args = ap.parse_args()

    if not os.path.isfile(args.poses):
        raise SystemExit(f"Pose file not found: {args.poses}")

    scan_ids, xyz, quat = load_poses(args.poses)
    if args.stride > 1:
        scan_ids = scan_ids[::args.stride]
        xyz = xyz[::args.stride]
        quat = quat[::args.stride]

    origin = xyz[0].copy()
    xyz_local = xyz - origin
    n = len(xyz_local)
    print(f"Loaded {n} poses from {args.poses}")
    print(f"  origin (first pose): {origin}")

    # Build path polyline
    cells = np.empty(n + 1, dtype=np.int64)
    cells[0] = n
    cells[1:] = np.arange(n)
    poly = pv.PolyData(xyz_local)
    poly.lines = cells

    plotter = pv.Plotter(window_size=(1400, 900))
    plotter.set_background("white")
    plotter.add_mesh(poly, color=PATH_COLOR, line_width=args.line_width)
    plotter.add_mesh(pv.Sphere(radius=args.marker_radius, center=xyz_local[0]),
                     color="green")
    plotter.add_mesh(pv.Sphere(radius=args.marker_radius, center=xyz_local[-1]),
                     color="red")

    # Accumulate lidar
    if not args.no_cloud:
        if not os.path.isdir(args.lidar_dir):
            raise SystemExit(f"Lidar dir not found: {args.lidar_dir}")

        # Use the CSV "scan #" column directly as the bin filename index
        keep_local = select_keyframes(xyz_local, args.keyframe_dist)
        if args.max_scans > 0:
            keep_local = keep_local[:args.max_scans]
        kept_scan_ids = scan_ids[keep_local]

        R_all = quat_to_rot(quat[keep_local])
        t_all = xyz_local[keep_local]

        low = np.array(POINT_COLOR_LOW, dtype=np.float32)
        high = np.array(POINT_COLOR_HIGH, dtype=np.float32)
        cmap_obj = None
        if args.cmap:
            import matplotlib.cm as mcm
            cmap_obj = mcm.get_cmap(args.cmap)

        pt_chunks = []
        rgb_chunks = []
        n_loaded = 0
        n_missing = 0
        n_points_total = 0
        for k, (sid, R, t) in enumerate(zip(kept_scan_ids, R_all, t_all)):
            bin_path = os.path.join(args.lidar_dir, f"{int(sid):010d}.bin")
            if not os.path.isfile(bin_path):
                n_missing += 1
                continue
            pts, intensity = load_bin(bin_path, args.max_range)
            if len(pts) == 0:
                continue
            # Voxel downsample together so colors stay aligned with surviving points
            if args.voxel_size > 0:
                keys = np.floor(pts / args.voxel_size).astype(np.int64)
                _, ds_idx = np.unique(keys, axis=0, return_index=True)
                pts = pts[ds_idx]
                intensity = intensity[ds_idx]
            if args.color_by == "intensity":
                src = intensity
                lo, hi = float(np.percentile(src, 1)), float(np.percentile(src, 99))
                t_norm = np.clip((src - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
                t_norm = np.power(t_norm, args.intensity_gamma)
            else:
                src = pts[:, 2]
                lo, hi = float(np.percentile(src, 1)), float(np.percentile(src, 99))
                t_norm = np.clip((src - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            if cmap_obj is not None:
                rgb = cmap_obj(t_norm)[:, :3].astype(np.float32)
            else:
                tn = t_norm[:, None]
                rgb = (low * (1.0 - tn) + high * tn).astype(np.float32)
            world = (pts @ R.T + t).astype(np.float32)
            pt_chunks.append(world)
            rgb_chunks.append(rgb)
            n_loaded += 1
            n_points_total += len(world)
            if k % 100 == 0:
                print(f"  scan {k+1}/{len(keep_local)}  loaded={n_loaded}  "
                      f"missing={n_missing}  points={n_points_total:,}")

        print(f"Accumulated {n_loaded} scans (missing {n_missing}), {n_points_total:,} points")

        if pt_chunks:
            cloud = np.concatenate(pt_chunks, axis=0)
            rgb = np.concatenate(rgb_chunks, axis=0)
            if args.final_voxel_size > 0:
                # Keep the first occurrence per voxel and its precomputed color
                keys = np.floor(cloud / args.final_voxel_size).astype(np.int64)
                _, idx = np.unique(keys, axis=0, return_index=True)
                cloud = cloud[idx]
                rgb = rgb[idx]
                print(f"After final voxel downsample ({args.final_voxel_size} m): {len(cloud):,} points")
            cloud_pd = pv.PolyData(cloud)
            cloud_pd["rgb"] = rgb
            plotter.add_mesh(cloud_pd, scalars="rgb", rgb=True,
                             point_size=args.point_size,
                             render_points_as_spheres=False)

    plotter.add_axes()
    plotter.add_text(
        f"CU North Campus robot1 — {n} poses (stride={args.stride}, keyframe_dist={args.keyframe_dist} m)",
        font_size=10, color="black",
    )
    plotter.view_xy()
    plotter.show()


if __name__ == "__main__":
    main()
