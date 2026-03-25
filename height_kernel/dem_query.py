#!/usr/bin/env python3
"""Step 3: DEM/DSM query utility.

Provides a class that:
1. Loads the Umeyama transform (KITTI-360 local → UTM 32N)
2. Opens the merged DEM and DSM rasters
3. For any 3D point in KITTI-360 local frame, returns:
   - z_dem: bare-earth elevation
   - z_dsm: surface elevation (with buildings/trees)
   - h_above_ground: z_point - z_dem (height above ground)

Usage (inside container):
    python3 /ros2_ws/src/osm_bki/height_kernel/dem_query.py
"""

import os

import numpy as np
import rasterio
from scipy.ndimage import map_coordinates


class DEMQuery:
    """Query DEM/DSM elevations for points in KITTI-360 local frame."""

    def __init__(
        self,
        transform_file,
        dem_path,
        dsm_path=None,
    ):
        # Load the Umeyama transform
        data = np.load(transform_file)
        self.T = data["T_umeyama"]  # (2, 3): [R | t]
        self.kitti_first_z = float(data["kitti_first_z"])

        # Open rasters (keep open for repeated queries)
        self.dem_src = rasterio.open(dem_path)
        self.dem_data = self.dem_src.read(1)
        self.dem_transform = self.dem_src.transform
        self.dem_nodata = self.dem_src.nodata

        self.dsm_src = None
        self.dsm_data = None
        if dsm_path and os.path.exists(dsm_path):
            self.dsm_src = rasterio.open(dsm_path)
            self.dsm_data = self.dsm_src.read(1)
            self.dsm_transform = self.dsm_src.transform
            self.dsm_nodata = self.dsm_src.nodata

    def close(self):
        self.dem_src.close()
        if self.dsm_src:
            self.dsm_src.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def kitti_to_utm(self, xyz):
        """Transform points from KITTI-360 local frame to UTM 32N.

        Args:
            xyz: (N, 3) array in KITTI-360 local frame (first pose at origin)

        Returns:
            (N, 3) array in UTM 32N (easting, northing, original_z)
        """
        xy = xyz[:, :2]
        ones = np.ones((len(xy), 1))
        xy_hom = np.hstack([xy, ones])  # (N, 3)
        utm_xy = (self.T @ xy_hom.T).T  # (N, 2)
        return np.column_stack([utm_xy, xyz[:, 2]])

    def _query_raster(self, utm_e, utm_n, raster_data, raster_transform, nodata):
        """Query a raster at UTM coordinates using bilinear interpolation.

        Returns elevation array. NaN where out of bounds or nodata.
        """
        # Convert UTM coords to fractional pixel coordinates
        inv_transform = ~raster_transform
        col, row = inv_transform * (utm_e, utm_n)

        # Handle scalar inputs
        col = np.atleast_1d(np.asarray(col, dtype=np.float64))
        row = np.atleast_1d(np.asarray(row, dtype=np.float64))

        h, w = raster_data.shape
        result = np.full(len(col), np.nan)

        # Mask valid pixels (with 1-pixel border for bilinear)
        valid = (col >= 0.5) & (col < w - 0.5) & (row >= 0.5) & (row < h - 0.5)

        if valid.any():
            # scipy map_coordinates expects (row, col) order
            coords = np.array([row[valid], col[valid]])
            vals = map_coordinates(raster_data, coords, order=1, mode="nearest")

            # Check for nodata
            if nodata is not None:
                vals[vals == nodata] = np.nan

            result[valid] = vals

        return result

    def query(self, xyz):
        """Query DEM and DSM for points in KITTI-360 local frame.

        Args:
            xyz: (N, 3) array in KITTI-360 local frame

        Returns:
            dict with:
                z_dem: (N,) bare-earth elevation
                z_dsm: (N,) surface elevation (NaN if no DSM)
                h_above_ground: (N,) height above ground = z_utm - z_dem
                valid: (N,) boolean mask where DEM data was available
        """
        xyz = np.atleast_2d(xyz)
        utm = self.kitti_to_utm(xyz)
        utm_e = utm[:, 0]
        utm_n = utm[:, 1]
        # The KITTI z-coordinate is in the Mercator frame (roughly elevation in meters)
        # The absolute elevation = kitti_z (raw, not shifted) = local_z + kitti_first_z
        z_abs = xyz[:, 2] + self.kitti_first_z

        z_dem = self._query_raster(
            utm_e, utm_n, self.dem_data, self.dem_transform, self.dem_nodata
        )

        z_dsm = np.full(len(xyz), np.nan)
        if self.dsm_data is not None:
            z_dsm = self._query_raster(
                utm_e, utm_n, self.dsm_data, self.dsm_transform, self.dsm_nodata
            )

        h_above_ground = z_abs - z_dem
        valid = ~np.isnan(z_dem)

        return {
            "z_dem": z_dem,
            "z_dsm": z_dsm,
            "h_above_ground": h_above_ground,
            "z_abs": z_abs,
            "valid": valid,
        }


def main():
    """Quick test: query DEM at a few trajectory poses."""
    SEQUENCE_DIR = "/media/sgarimella34/hercules-collect1/datasets/kitti360/2013_05_28_drive_0000_sync"
    TRANSFORM_FILE = os.path.join(SEQUENCE_DIR, "kitti360_to_utm.npz")
    DEM_PATH = "/media/sgarimella34/hercules-collect1/kitti360_DGM025/merged_dem.tif"
    DSM_PATH = "/media/sgarimella34/hercules-collect1/kitti360_DOM1/merged_dsm.tif"
    POSE_FILE = "/media/sgarimella34/hercules-collect1/datasets/kitti360/2013_05_28_drive_0000_sync/velodyne_poses.txt"

    # Load a few poses
    poses = []
    with open(POSE_FILE) as f:
        for i, line in enumerate(f):
            if i >= 100:
                break
            vals = line.strip().split()
            floats = [float(v) for v in vals[1:]]
            if len(floats) >= 16:
                mat = np.array(floats[:16]).reshape(4, 4)
            else:
                mat = np.eye(4)
                mat[:3, :] = np.array(floats[:12]).reshape(3, 4)
            poses.append(mat[:3, 3])
    raw_poses = np.array(poses)

    # Shift to local frame (subtract first pose translation)
    first_t = raw_poses[0].copy()
    local_poses = raw_poses - first_t

    with DEMQuery(TRANSFORM_FILE, DEM_PATH, DSM_PATH) as dq:
        result = dq.query(local_poses)

        print("DEM Query Test (first 10 poses):")
        print(f"{'Pose':>5} {'z_abs':>8} {'z_DEM':>8} {'z_DSM':>8} {'h_above':>8} {'valid':>6}")
        print("-" * 52)
        for i in range(min(10, len(local_poses))):
            print(
                f"{i:5d} "
                f"{result['z_abs'][i]:8.2f} "
                f"{result['z_dem'][i]:8.2f} "
                f"{result['z_dsm'][i]:8.2f} "
                f"{result['h_above_ground'][i]:8.2f} "
                f"{'Y' if result['valid'][i] else 'N':>6}"
            )

        valid_mask = result["valid"]
        h = result["h_above_ground"][valid_mask]
        print(f"\nHeight above ground stats (first 100 poses):")
        print(f"  Valid: {valid_mask.sum()}/{len(valid_mask)}")
        print(f"  Mean:  {h.mean():.2f}m")
        print(f"  Std:   {h.std():.2f}m")
        print(f"  Min:   {h.min():.2f}m")
        print(f"  Max:   {h.max():.2f}m")


if __name__ == "__main__":
    main()
