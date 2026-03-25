#!/usr/bin/env python3
"""Step 1: Merge DEM (DGM025) and DSM (DOM1) GeoTIFF tiles into single mosaics.

Usage (inside container):
    python3 /ros2_ws/src/osm_bki/height_kernel/merge_tiles.py

Outputs:
    /media/sgarimella34/hercules-collect1/kitti360_DGM025/merged_dem.tif
    /media/sgarimella34/hercules-collect1/kitti360_DOM1/merged_dsm.tif
"""

import glob
import os
import sys

import rasterio
from rasterio.merge import merge


DEM_DIR = "/media/sgarimella34/hercules-collect1/kitti360_DGM025"
DSM_DIR = "/media/sgarimella34/hercules-collect1/kitti360_DOM1"

DEM_OUT = os.path.join(DEM_DIR, "merged_dem.tif")
DSM_OUT = os.path.join(DSM_DIR, "merged_dsm.tif")


def find_tifs(root_dir):
    """Find all .tif files recursively under root_dir."""
    tifs = sorted(glob.glob(os.path.join(root_dir, "**", "*.tif"), recursive=True))
    return tifs


def merge_tiles(tif_paths, output_path, label=""):
    """Merge a list of GeoTIFF files into a single mosaic."""
    if os.path.exists(output_path):
        print(f"[{label}] Output already exists: {output_path} — skipping merge.")
        with rasterio.open(output_path) as src:
            print_raster_info(src, label)
        return output_path

    print(f"[{label}] Found {len(tif_paths)} tiles. Merging...")

    datasets = [rasterio.open(p) for p in tif_paths]
    mosaic, out_transform = merge(datasets)

    out_meta = datasets[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform,
        "compress": "deflate",
    })

    for ds in datasets:
        ds.close()

    print(f"[{label}] Writing mosaic to {output_path} ...")
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    with rasterio.open(output_path) as src:
        print_raster_info(src, label)

    return output_path


def print_raster_info(src, label=""):
    """Print summary info about an open rasterio dataset."""
    bounds = src.bounds
    print(f"\n[{label}] Raster info:")
    print(f"  CRS:        {src.crs}")
    print(f"  Resolution: {src.res}")
    print(f"  Shape:      {src.height} x {src.width}")
    print(f"  Bounds:     west={bounds.left:.2f}  east={bounds.right:.2f}")
    print(f"              south={bounds.bottom:.2f}  north={bounds.top:.2f}")
    print(f"  Dtype:      {src.dtypes[0]}")
    data = src.read(1)
    valid = data[data != src.nodata] if src.nodata is not None else data
    print(f"  Elevation:  min={valid.min():.2f}  max={valid.max():.2f}  mean={valid.mean():.2f}")


def main():
    # --- DEM ---
    dem_tifs = find_tifs(DEM_DIR)
    if not dem_tifs:
        print(f"ERROR: No .tif files found under {DEM_DIR}")
        sys.exit(1)
    merge_tiles(dem_tifs, DEM_OUT, label="DEM")

    # --- DSM ---
    dsm_tifs = find_tifs(DSM_DIR)
    if not dsm_tifs:
        print(f"ERROR: No .tif files found under {DSM_DIR}")
        sys.exit(1)
    merge_tiles(dsm_tifs, DSM_OUT, label="DSM")

    print("\nDone. Merged mosaics:")
    print(f"  DEM: {DEM_OUT}")
    print(f"  DSM: {DSM_OUT}")


if __name__ == "__main__":
    main()
