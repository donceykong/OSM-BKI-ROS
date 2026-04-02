#!/usr/bin/env python3
"""Convert a classified .laz point cloud to DEM (bare-earth) and DSM (surface) GeoTIFFs.

Uses LAS classification codes:
  - Ground (class 2) → DEM
  - All first returns / highest points → DSM

Output CRS matches the input .laz file (e.g., EPSG:3006 for Swedish data).

Usage:
    python3 convert_laz_to_dem_dsm.py
"""

import os

import laspy
import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import generic_filter

# --- Config ---
LAZ_PATH = "/media/sgarimella34/hercules-collect1/kth_geopatial_data/kth_geopatial_data/65_6/21C031_658_67_2525.laz"
OUTPUT_DIR = "/media/sgarimella34/hercules-collect1/kth_dem_dsm"
CELL_SIZE = 0.5  # meters per pixel (matches BKI resolution)
NODATA = -9999.0


def rasterize(x, y, z, cell_size, agg_func, bounds=None):
    """Rasterize scattered points into a regular grid using the given aggregation function."""
    if bounds is None:
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
    else:
        x_min, y_min, x_max, y_max = bounds

    cols = int(np.ceil((x_max - x_min) / cell_size))
    rows = int(np.ceil((y_max - y_min) / cell_size))
    print(f"  Grid: {cols}x{rows} ({cols * cell_size:.0f}x{rows * cell_size:.0f} m)")

    grid = np.full((rows, cols), np.nan, dtype=np.float32)

    # Bin points
    col_idx = np.clip(((x - x_min) / cell_size).astype(int), 0, cols - 1)
    row_idx = np.clip(((y_max - y) / cell_size).astype(int), 0, rows - 1)  # y_max at row 0 (north-up)

    # Aggregate per cell
    from collections import defaultdict
    bins = defaultdict(list)
    for i in range(len(z)):
        bins[(row_idx[i], col_idx[i])].append(z[i])

    for (r, c), vals in bins.items():
        grid[r, c] = agg_func(vals)

    # Fill small holes with median filter
    valid_before = np.count_nonzero(~np.isnan(grid))
    nan_mask = np.isnan(grid)
    if nan_mask.any():
        filled = generic_filter(grid, lambda v: np.nanmedian(v) if np.any(~np.isnan(v)) else np.nan,
                                size=3, mode='constant', cval=np.nan)
        grid[nan_mask] = filled[nan_mask]
    valid_after = np.count_nonzero(~np.isnan(grid))
    print(f"  Valid cells: {valid_before} → {valid_after}/{grid.size} ({100 * valid_after / grid.size:.1f}%)")

    # Create transform (north-up: origin at top-left)
    transform = from_origin(x_min, y_max, cell_size, cell_size)
    return grid, transform, (rows, cols)


def write_geotiff(path, grid, transform, crs_wkt, nodata=NODATA):
    """Write a single-band GeoTIFF."""
    grid_out = grid.copy()
    grid_out[np.isnan(grid_out)] = nodata

    with rasterio.open(
        path, 'w', driver='GTiff',
        height=grid.shape[0], width=grid.shape[1],
        count=1, dtype='float32',
        crs=crs_wkt, transform=transform, nodata=nodata,
        compress='deflate'
    ) as dst:
        dst.write(grid_out, 1)
    print(f"  Written {path} ({os.path.getsize(path) / 1e6:.1f} MB)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading {LAZ_PATH}...")
    las = laspy.read(LAZ_PATH)
    x, y, z = las.x, las.y, las.z
    classification = las.classification
    print(f"  {len(x):,} points, X=[{x.min():.1f}, {x.max():.1f}], Y=[{y.min():.1f}, {y.max():.1f}], Z=[{z.min():.1f}, {z.max():.1f}]")

    # Print classification distribution
    unique, counts = np.unique(classification, return_counts=True)
    print("  Classifications:")
    CLASS_NAMES = {0: 'unclassified', 1: 'unassigned', 2: 'ground', 3: 'low-veg',
                   4: 'med-veg', 5: 'high-veg', 6: 'building', 7: 'noise',
                   8: 'reserved', 9: 'water', 17: 'bridge'}
    for cls, cnt in zip(unique, counts):
        name = CLASS_NAMES.get(cls, f'class-{cls}')
        print(f"    {cls:3d} ({name:>15s}): {cnt:>10,} ({100 * cnt / len(x):.1f}%)")

    # CRS from the tile JSON (EPSG:3006)
    crs_wkt = "EPSG:3006"

    # Compute shared bounds
    bounds = (x.min(), y.min(), x.max(), y.max())

    # DEM: ground points only (class 2), take minimum elevation per cell
    print("\nDEM (ground points, min elevation):")
    ground_mask = classification == 2
    print(f"  Ground points: {ground_mask.sum():,}")
    if ground_mask.sum() == 0:
        print("  ERROR: No ground points found! Cannot create DEM.")
        return
    dem_grid, dem_transform, dem_shape = rasterize(
        x[ground_mask], y[ground_mask], z[ground_mask],
        CELL_SIZE, lambda v: np.min(v), bounds=bounds
    )
    write_geotiff(os.path.join(OUTPUT_DIR, "kth_dem.tif"), dem_grid, dem_transform, crs_wkt)

    # DSM: all points, take maximum elevation per cell
    print("\nDSM (all points, max elevation):")
    dsm_grid, dsm_transform, dsm_shape = rasterize(
        x, y, z, CELL_SIZE, lambda v: np.max(v), bounds=bounds
    )
    write_geotiff(os.path.join(OUTPUT_DIR, "kth_dsm.tif"), dsm_grid, dsm_transform, crs_wkt)

    print("\nDone.")


if __name__ == "__main__":
    main()
