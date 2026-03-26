#!/usr/bin/env python3
"""Precompute DEM and DSM elevation grids in KITTI-360 local frame.

Generates simple binary grid files that C++ can load without GDAL.
For each grid cell at (local_x, local_y), stores the DEM/DSM elevation
offset by kitti_first_z, so that in C++:

    height_above_ground = point_z - grid_value

Binary format (little-endian):
    Header (32 bytes):
        char[4]  magic     = "DEMG"
        int32    version   = 1
        float32  origin_x  = local X of cell (0,0)
        float32  origin_y  = local Y of cell (0,0)
        float32  cell_size = meters per cell
        int32    cols      = grid width
        int32    rows      = grid height
        float32  nodata    = nodata sentinel (NaN)
    Data:
        float32[rows * cols]  row-major, row 0 = min Y

Usage (inside container):
    python3 /ros2_ws/src/osm_bki/height_kernel/precompute_dem_grid.py
"""

import os
import struct

import numpy as np
import rasterio

# --- Config ---
DATASET_DIR = "/media/sgarimella34/hercules-collect1/datasets/kitti360"
SEQUENCE = "2013_05_28_drive_0000_sync"
SEQUENCE_DIR = os.path.join(DATASET_DIR, SEQUENCE)
POSE_FILE = os.path.join(SEQUENCE_DIR, "velodyne_poses.txt")
TRANSFORM_FILE = os.path.join(SEQUENCE_DIR, "kitti360_to_utm.npz")

DEM_PATH = "/media/sgarimella34/hercules-collect1/kitti360_DGM025/merged_dem.tif"
DSM_PATH = "/media/sgarimella34/hercules-collect1/kitti360_DOM1/merged_dsm.tif"

CELL_SIZE = 0.5  # meters (matches BKI resolution)
MARGIN = 100.0   # meters beyond trajectory bounding box

OUTPUT_DEM = os.path.join(SEQUENCE_DIR, "dem_local_grid.bin")
OUTPUT_DSM = os.path.join(SEQUENCE_DIR, "dsm_local_grid.bin")

MAGIC = b"DEMG"
VERSION = 1


def load_poses_local(pose_file):
    """Load poses and shift to local frame (first pose at origin)."""
    raw = []
    with open(pose_file) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) < 13:
                continue
            floats = [float(v) for v in vals[1:]]
            if len(floats) >= 16:
                mat = np.array(floats[:16]).reshape(4, 4)
            elif len(floats) == 12:
                mat = np.eye(4)
                mat[:3, :] = np.array(floats[:12]).reshape(3, 4)
            else:
                continue
            raw.append(mat[:3, 3])
    raw = np.array(raw)
    first_t = raw[0].copy()
    local = raw - first_t
    return local, first_t


def write_grid(path, origin_x, origin_y, cell_size, data):
    """Write a binary grid file."""
    rows, cols = data.shape
    nodata = float("nan")
    header = struct.pack(
        "<4sifffiif",
        MAGIC, VERSION,
        float(origin_x), float(origin_y), float(cell_size),
        int(cols), int(rows), nodata,
    )
    with open(path, "wb") as f:
        f.write(header)
        # Replace NaN with nodata marker (NaN itself for float32)
        f.write(data.astype(np.float32).tobytes())
    print(f"  Written {path} ({rows}x{cols}, {os.path.getsize(path) / 1e6:.1f} MB)")


def precompute_grid(local_poses, first_t, T_umeyama, raster_path, output_path, cell_size, margin):
    """Precompute elevation grid in local frame from a UTM raster."""
    # Bounding box in local frame
    x_min = local_poses[:, 0].min() - margin
    x_max = local_poses[:, 0].max() + margin
    y_min = local_poses[:, 1].min() - margin
    y_max = local_poses[:, 1].max() + margin

    cols = int(np.ceil((x_max - x_min) / cell_size))
    rows = int(np.ceil((y_max - y_min) / cell_size))
    print(f"  Grid: origin=({x_min:.1f}, {y_min:.1f}), size={cols}x{rows}, "
          f"cell={cell_size}m, coverage={cols*cell_size:.0f}x{rows*cell_size:.0f}m")

    # Generate local (x, y) for each grid cell
    xs = x_min + (np.arange(cols) + 0.5) * cell_size  # cell centers
    ys = y_min + (np.arange(rows) + 0.5) * cell_size
    gx, gy = np.meshgrid(xs, ys)  # (rows, cols)
    flat_xy = np.column_stack([gx.ravel(), gy.ravel()])

    # Transform local → UTM via Umeyama
    ones = np.ones((len(flat_xy), 1))
    xy_hom = np.hstack([flat_xy, ones])
    utm_xy = (T_umeyama @ xy_hom.T).T  # (N, 2)
    utm_e = utm_xy[:, 0]
    utm_n = utm_xy[:, 1]

    # Query raster
    with rasterio.open(raster_path) as src:
        inv_t = ~src.transform
        col_f, row_f = inv_t * (utm_e, utm_n)
        col_f = np.asarray(col_f, dtype=np.float64)
        row_f = np.asarray(row_f, dtype=np.float64)

        raster_data = src.read(1)
        h, w = raster_data.shape
        nodata = src.nodata

        # Bilinear interpolation via scipy
        from scipy.ndimage import map_coordinates
        valid = (col_f >= 0.5) & (col_f < w - 0.5) & (row_f >= 0.5) & (row_f < h - 0.5)
        elevations = np.full(len(flat_xy), np.nan, dtype=np.float32)
        if valid.any():
            coords = np.array([row_f[valid], col_f[valid]])
            vals = map_coordinates(raster_data, coords, order=1, mode="nearest")
            if nodata is not None:
                vals[vals == nodata] = np.nan
            elevations[valid] = vals

    # Offset by first_z so C++ can compute: h = local_z - grid_value
    # grid_value = dem_elevation - first_z
    first_z = first_t[2]
    elevations -= first_z

    grid = elevations.reshape(rows, cols)
    valid_count = np.count_nonzero(~np.isnan(grid))
    print(f"  Valid cells: {valid_count}/{rows*cols} ({100*valid_count/(rows*cols):.1f}%)")

    write_grid(output_path, x_min, y_min, cell_size, grid)
    return x_min, y_min


def main():
    print("Precomputing DEM/DSM grids in KITTI-360 local frame")

    # Load poses
    local_poses, first_t = load_poses_local(POSE_FILE)
    print(f"  {len(local_poses)} poses, first_t=[{first_t[0]:.2f}, {first_t[1]:.2f}, {first_t[2]:.2f}]")

    # Load Umeyama transform
    data = np.load(TRANSFORM_FILE)
    T_umeyama = data["T_umeyama"]
    print(f"  Umeyama transform loaded from {TRANSFORM_FILE}")

    # Precompute DEM grid
    print(f"\nDEM ({DEM_PATH}):")
    precompute_grid(local_poses, first_t, T_umeyama, DEM_PATH, OUTPUT_DEM, CELL_SIZE, MARGIN)

    # Precompute DSM grid
    print(f"\nDSM ({DSM_PATH}):")
    precompute_grid(local_poses, first_t, T_umeyama, DSM_PATH, OUTPUT_DSM, CELL_SIZE, MARGIN)

    print("\nDone. Grid files ready for C++ loading.")


if __name__ == "__main__":
    main()
