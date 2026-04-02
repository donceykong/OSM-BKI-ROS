#!/usr/bin/env python3
"""Precompute DEM and DSM elevation grids in MCD local frame.

MCD poses (pose_inW.csv) are in a local ENU frame. The transform to EPSG:3006
is a simple translation: EPSG3006 = origin_3006 + local_pose.

The origin is derived from the OSM origin lat/lon in the MCD config.

Generates binary grid files identical in format to the KITTI-360 version.

Usage (inside container):
    python3 /ros2_ws/src/osm_bki/height_kernel/precompute_dem_grid_mcd.py
"""

import os
import struct

import numpy as np
import pyproj
import rasterio
from scipy.ndimage import map_coordinates

# --- Config ---
DATASET_DIR = "/media/sgarimella34/hercules-collect1/datasets/mcd"
SEQUENCE = os.environ.get("MCD_SEQUENCE", "kth_day_09")
SEQUENCE_DIR = os.path.join(DATASET_DIR, SEQUENCE)
POSE_FILE = os.path.join(SEQUENCE_DIR, "pose_inW.csv")

DEM_PATH = "/media/sgarimella34/hercules-collect1/kth_dem_dsm/kth_dem.tif"
DSM_PATH = "/media/sgarimella34/hercules-collect1/kth_dem_dsm/kth_dsm.tif"

# MCD OSM origin (KTH campus) — defines local frame origin in geographic coords
ORIGIN_LAT = 59.347671416
ORIGIN_LON = 18.072069652

CELL_SIZE = 0.5   # meters (matches BKI resolution)
MARGIN = 100.0    # meters beyond trajectory bounding box

OUTPUT_DEM = os.path.join(SEQUENCE_DIR, "dem_local_grid.bin")
OUTPUT_DSM = os.path.join(SEQUENCE_DIR, "dsm_local_grid.bin")

MAGIC = b"DEMG"
VERSION = 1


def load_mcd_poses(pose_file):
    """Load MCD poses from CSV. Returns raw translations (in local W frame)."""
    raw = []
    with open(pose_file) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 5:
                continue
            x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            raw.append([x, y, z])
    raw = np.array(raw)
    return raw


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
        f.write(data.astype(np.float32).tobytes())
    print(f"  Written {path} ({rows}x{cols}, {os.path.getsize(path) / 1e6:.1f} MB)")


def precompute_grid(local_poses, first_t, origin_e, origin_n, raster_path, output_path, cell_size, margin):
    """Precompute elevation grid in C++-local frame from an EPSG:3006 raster."""
    # C++ shifts all poses by first_t, so local_shifted = raw - first_t
    local_shifted = local_poses - first_t

    # Bounding box in C++ local frame
    x_min = local_shifted[:, 0].min() - margin
    x_max = local_shifted[:, 0].max() + margin
    y_min = local_shifted[:, 1].min() - margin
    y_max = local_shifted[:, 1].max() + margin

    cols = int(np.ceil((x_max - x_min) / cell_size))
    rows = int(np.ceil((y_max - y_min) / cell_size))
    print(f"  Grid: origin=({x_min:.1f}, {y_min:.1f}), size={cols}x{rows}, "
          f"cell={cell_size}m, coverage={cols * cell_size:.0f}x{rows * cell_size:.0f}m")

    # Generate local (x, y) for each grid cell (in C++ local frame)
    xs = x_min + (np.arange(cols) + 0.5) * cell_size
    ys = y_min + (np.arange(rows) + 0.5) * cell_size
    gx, gy = np.meshgrid(xs, ys)
    flat_xy = np.column_stack([gx.ravel(), gy.ravel()])

    # Transform C++ local → raw local → EPSG:3006
    # C++ local = raw - first_t, so raw = C++ local + first_t
    # EPSG:3006 = origin + raw
    epsg_e = origin_e + flat_xy[:, 0] + first_t[0]
    epsg_n = origin_n + flat_xy[:, 1] + first_t[1]

    # Query raster
    with rasterio.open(raster_path) as src:
        inv_t = ~src.transform
        col_f, row_f = inv_t * (epsg_e, epsg_n)
        col_f = np.asarray(col_f, dtype=np.float64)
        row_f = np.asarray(row_f, dtype=np.float64)

        raster_data = src.read(1).astype(np.float64)
        h, w = raster_data.shape
        nodata = src.nodata

        # Replace nodata with NaN BEFORE interpolation to prevent blending
        if nodata is not None:
            raster_data[raster_data == nodata] = np.nan

        valid = (col_f >= 0.5) & (col_f < w - 0.5) & (row_f >= 0.5) & (row_f < h - 0.5)
        elevations = np.full(len(flat_xy), np.nan, dtype=np.float32)
        if valid.any():
            coords = np.array([row_f[valid], col_f[valid]])
            vals = map_coordinates(raster_data, coords, order=1, mode="constant", cval=np.nan)
            elevations[valid] = vals

    # Offset by first_z so C++ can compute: h = local_z - grid_value
    first_z = first_t[2]
    elevations -= first_z

    grid = elevations.reshape(rows, cols)
    valid_count = np.count_nonzero(~np.isnan(grid))
    print(f"  Valid cells: {valid_count}/{rows * cols} ({100 * valid_count / (rows * cols):.1f}%)")

    write_grid(output_path, x_min, y_min, cell_size, grid)


def main():
    print(f"Precomputing DEM/DSM grids for MCD sequence: {SEQUENCE}")

    # Compute origin in EPSG:3006
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3006", always_xy=True)
    origin_e, origin_n = transformer.transform(ORIGIN_LON, ORIGIN_LAT)
    print(f"  Origin (EPSG:3006): E={origin_e:.2f}, N={origin_n:.2f}")

    # Load raw poses
    raw_poses = load_mcd_poses(POSE_FILE)
    first_t = raw_poses[0].copy()
    print(f"  {len(raw_poses)} poses, first_t=[{first_t[0]:.2f}, {first_t[1]:.2f}, {first_t[2]:.2f}]")

    # DEM
    print(f"\nDEM ({DEM_PATH}):")
    precompute_grid(raw_poses, first_t, origin_e, origin_n, DEM_PATH, OUTPUT_DEM, CELL_SIZE, MARGIN)

    # DSM
    print(f"\nDSM ({DSM_PATH}):")
    precompute_grid(raw_poses, first_t, origin_e, origin_n, DSM_PATH, OUTPUT_DSM, CELL_SIZE, MARGIN)

    print("\nDone. Grid files ready for C++ loading.")


if __name__ == "__main__":
    main()
