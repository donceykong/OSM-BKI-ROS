#!/usr/bin/env python3
"""ROS2 node that publishes DEM and DSM surfaces as PointCloud2 topics in the KITTI-360 frame.

Publishes:
  /dem_surface  - bare-earth DEM colored green
  /dsm_surface  - DSM (buildings/trees) colored blue

Both are in the KITTI-360 local frame (map frame) so they align with the
semantic map and OSM overlays in RViz.

Usage (inside container):
    source /opt/ros/humble/setup.bash
    source /ros2_ws/install/setup.bash
    python3 /ros2_ws/src/osm_bki/height_kernel/dem_visualizer_node.py
"""

import os
import struct
import sys
import time

import numpy as np
import rasterio
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header


def numpy_to_pointcloud2(points, colors, frame_id="map", stamp=None):
    """Convert Nx3 points + Nx3 uint8 colors to a PointCloud2 message."""
    N = len(points)
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]

    # Pack RGB into float32 (RViz convention)
    rgb_packed = np.zeros(N, dtype=np.float32)
    r = colors[:, 0].astype(np.uint32)
    g = colors[:, 1].astype(np.uint32)
    b = colors[:, 2].astype(np.uint32)
    rgb_int = (r << 16) | (g << 8) | b
    rgb_packed = rgb_int.view(np.float32)

    # Build data buffer
    data = np.zeros(N, dtype=[
        ("x", np.float32), ("y", np.float32), ("z", np.float32), ("rgb", np.float32)
    ])
    data["x"] = points[:, 0].astype(np.float32)
    data["y"] = points[:, 1].astype(np.float32)
    data["z"] = points[:, 2].astype(np.float32)
    data["rgb"] = rgb_packed

    msg = PointCloud2()
    msg.header = Header()
    msg.header.frame_id = frame_id
    if stamp:
        msg.header.stamp = stamp
    msg.height = 1
    msg.width = N
    msg.fields = fields
    msg.is_bigendian = False
    msg.point_step = 16
    msg.row_step = 16 * N
    msg.data = data.tobytes()
    msg.is_dense = True
    return msg


class DEMVisualizerNode(Node):
    def __init__(self):
        super().__init__("dem_visualizer_node")

        # Load config
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "config.yaml"
        )
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.dem_path = cfg["dem_path"]
        self.dsm_path = cfg["dsm_path"]
        transform_file = cfg["transform_file"]
        self.dem_res = cfg.get("dem_resolution", 2.0)
        self.dsm_res = cfg.get("dsm_resolution", 2.0)
        self.vis_radius = cfg.get("vis_radius", 150.0)

        # Load transform
        data = np.load(transform_file)
        self.T = data["T_umeyama"]  # (2, 3) KITTI local → UTM
        self.kitti_first_z = float(data["kitti_first_z"])

        # Compute inverse transform: UTM → KITTI local
        R = self.T[:2, :2]
        t = self.T[:2, 2]
        R_inv = R.T
        t_inv = -R_inv @ t
        self.T_inv = np.zeros((2, 3))
        self.T_inv[:2, :2] = R_inv
        self.T_inv[:2, 2] = t_inv

        # Load trajectory to determine area of interest
        pose_dir = os.path.dirname(transform_file)
        pose_file = os.path.join(pose_dir, "velodyne_poses.txt")
        self.trajectory_local = self._load_trajectory(pose_file)

        # Publishers
        dem_topic = cfg.get("dem_topic", "/dem_surface")
        dsm_topic = cfg.get("dsm_topic", "/dsm_surface")
        self.dem_pub = self.create_publisher(PointCloud2, dem_topic, 1)
        self.dsm_pub = self.create_publisher(PointCloud2, dsm_topic, 1)

        self.get_logger().info(
            f"DEM Visualizer: publishing DEM on '{dem_topic}', DSM on '{dsm_topic}'"
        )

        # Build and publish once (static surfaces)
        self._publish_surfaces()

        # Re-publish periodically so late-joining RViz subscribers get it
        self.timer = self.create_timer(5.0, self._publish_surfaces)

    def _load_trajectory(self, pose_file):
        """Load trajectory and return local-frame XYZ (shifted to first-pose origin)."""
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
        return raw - first_t

    def _utm_to_kitti_local(self, utm_e, utm_n, z_elevation):
        """Transform UTM (easting, northing, elevation) → KITTI local frame."""
        xy_hom = np.column_stack([utm_e, utm_n, np.ones(len(utm_e))])
        local_xy = (self.T_inv @ xy_hom.T).T  # (N, 2)
        # z in local frame: elevation - kitti_first_z
        local_z = z_elevation - self.kitti_first_z
        return np.column_stack([local_xy, local_z])

    def _extract_surface(self, raster_path, resolution):
        """Extract a subsampled surface around the trajectory."""
        # Get trajectory bounding box in UTM
        traj_xy = self.trajectory_local[:, :2]
        ones = np.ones((len(traj_xy), 1))
        traj_hom = np.hstack([traj_xy, ones])
        traj_utm = (self.T @ traj_hom.T).T  # (N, 2)

        margin = self.vis_radius
        utm_min_e = traj_utm[:, 0].min() - margin
        utm_max_e = traj_utm[:, 0].max() + margin
        utm_min_n = traj_utm[:, 1].min() - margin
        utm_max_n = traj_utm[:, 1].max() + margin

        with rasterio.open(raster_path) as src:
            bounds = src.bounds
            # Clamp to raster bounds
            utm_min_e = max(utm_min_e, bounds.left)
            utm_max_e = min(utm_max_e, bounds.right)
            utm_min_n = max(utm_min_n, bounds.bottom)
            utm_max_n = min(utm_max_n, bounds.top)

            # Generate grid at desired resolution
            eastings = np.arange(utm_min_e, utm_max_e, resolution)
            northings = np.arange(utm_min_n, utm_max_n, resolution)
            ee, nn = np.meshgrid(eastings, northings)
            ee_flat = ee.ravel()
            nn_flat = nn.ravel()

            # Query raster
            from rasterio.transform import rowcol
            rows, cols = rowcol(src.transform, ee_flat, nn_flat)
            rows = np.array(rows)
            cols = np.array(cols)

            # Filter in-bounds
            h, w = src.height, src.width
            valid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            data = src.read(1)
            elevations = np.full(len(ee_flat), np.nan)
            elevations[valid] = data[rows[valid], cols[valid]]

            # Remove nodata
            if src.nodata is not None:
                elevations[elevations == src.nodata] = np.nan
            good = ~np.isnan(elevations)

        return ee_flat[good], nn_flat[good], elevations[good]

    def _publish_surfaces(self):
        """Build and publish DEM and DSM point clouds."""
        now = self.get_clock().now().to_msg()

        # DEM
        if os.path.exists(self.dem_path):
            e, n, z = self._extract_surface(self.dem_path, self.dem_res)
            local_pts = self._utm_to_kitti_local(e, n, z)
            # Green color for DEM
            colors = np.tile(np.array([34, 139, 34], dtype=np.uint8), (len(local_pts), 1))
            msg = numpy_to_pointcloud2(local_pts, colors, frame_id="map", stamp=now)
            self.dem_pub.publish(msg)
            self.get_logger().info(
                f"Published DEM: {len(local_pts)} points", throttle_duration_sec=30.0
            )

        # DSM
        if os.path.exists(self.dsm_path):
            e, n, z = self._extract_surface(self.dsm_path, self.dsm_res)
            local_pts = self._utm_to_kitti_local(e, n, z)
            # Blue color for DSM
            colors = np.tile(np.array([65, 105, 225], dtype=np.uint8), (len(local_pts), 1))
            msg = numpy_to_pointcloud2(local_pts, colors, frame_id="map", stamp=now)
            self.dsm_pub.publish(msg)
            self.get_logger().info(
                f"Published DSM: {len(local_pts)} points", throttle_duration_sec=30.0
            )


def main():
    rclpy.init()
    node = DEMVisualizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
