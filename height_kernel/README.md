# Height Kernel Pipeline

## Prerequisites

All commands run inside the Docker container. From the host:

```bash
cd ~/OSM-BKI-ROS

# Allow X11 for RViz
xhost +local:docker

# Build and start container (rebuilds if code changed)
docker compose down && docker compose up -d --build
```

Get a shell inside:
```bash
docker exec -it osm-bki bash
```

Inside the container, source ROS2:
```bash
source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
```

## Step 1: Merge DEM/DSM Tiles

Only needed once (outputs persist on the mounted drive).

```bash
python3 /ros2_ws/src/osm_bki/height_kernel/merge_tiles.py
```

Outputs:
- `/media/sgarimella34/hercules-collect1/kitti360_DGM025/merged_dem.tif`
- `/media/sgarimella34/hercules-collect1/kitti360_DOM1/merged_dsm.tif`

## Step 2: Align Trajectory to UTM

Only needed once per sequence (output persists on the mounted drive).

```bash
python3 /ros2_ws/src/osm_bki/height_kernel/align_trajectory.py
```

Outputs (in the sequence directory):
- `kitti360_to_utm.npz` — Umeyama transform (KITTI-360 local → UTM 32N)
- `trajectory_on_dem.png` — verification plot

## Step 3: Test DEM Query

Optional sanity check — queries DEM/DSM elevation at the first 100 poses.

```bash
python3 /ros2_ws/src/osm_bki/height_kernel/dem_query.py
```

## Step 4: Test Height Kernel

Optional — runs the height kernel on the first LiDAR scan and prints weight statistics.

```bash
cd /ros2_ws/src/osm_bki/height_kernel && python3 height_kernel.py
```

## Running KITTI-360 with DEM/DSM Visualization

Needs two terminals inside the container.

**Terminal 1** — KITTI-360 pipeline + RViz:
```bash
docker exec -it osm-bki bash
source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
ros2 launch osm_bki kitti360_launch.py
```

**Terminal 2** — DEM/DSM surface publisher:
```bash
docker exec -it osm-bki bash
source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
python3 /ros2_ws/src/osm_bki/height_kernel/dem_visualizer_node.py
```

In RViz, add two PointCloud2 displays:
- `/dem_surface` — bare-earth terrain (green)
- `/dsm_surface` — surface with buildings/trees (blue)

Both publish in the `map` frame. Subsampling resolution is configurable in `height_kernel/config.yaml` (`dem_resolution` / `dsm_resolution`, default 2.0m).

## What's Implemented vs What's Left

| Component | Status |
|-----------|--------|
| DEM/DSM tile merging | Done |
| KITTI-360 → UTM alignment (Umeyama) | Done |
| DEM/DSM query utility (bilinear interp) | Done |
| Height kernel k_ht (Python) | Done |
| Per-class height priors config | Done |
| DEM/DSM RViz visualizer (PointCloud2) | Done |
| **C++ BKI integration (k_sp * k_sem * k_ht)** | **Not yet** |

The height kernel math is fully implemented in `height_kernel.py` but it is **not yet wired into the C++ BKI update rule**. The existing C++ pipeline computes `k_sp * k_sem * y_i` per point during octree insertion. To use the height kernel, each point needs to be multiplied by its `k_ht` weight during that insertion step.

## File Overview

```
height_kernel/
├── merge_tiles.py          # Step 1: merge GeoTIFF tiles into mosaics
├── align_trajectory.py     # Step 2: Umeyama alignment KITTI-360 → UTM 32N
├── dem_query.py            # Step 3: DEM/DSM elevation query class
├── height_kernel.py        # Step 4: k_ht kernel implementation
├── config.yaml             # Height priors, paths, RViz settings
├── dem_visualizer_node.py  # ROS2 node: publish DEM/DSM as PointCloud2
└── README.md               # This file
```
