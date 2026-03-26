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

## One-Time Setup (Steps 1-3)

These only need to run once. Outputs persist on the mounted drive.

### Step 1: Merge DEM/DSM Tiles

```bash
python3 /ros2_ws/src/osm_bki/height_kernel/merge_tiles.py
```

### Step 2: Align Trajectory to UTM

```bash
python3 /ros2_ws/src/osm_bki/height_kernel/align_trajectory.py
```

### Step 3: Precompute Binary Grids for C++

Converts DEM/DSM from GeoTIFF (UTM 32N) into simple binary grids in KITTI-360 local frame so the C++ pipeline can query height-above-ground without GDAL.

```bash
python3 /ros2_ws/src/osm_bki/height_kernel/precompute_dem_grid.py
```

Outputs (in the sequence directory):
- `dem_local_grid.bin` — bare-earth DEM in local frame (17 MB)
- `dsm_local_grid.bin` — surface DSM in local frame (17 MB)

## Running KITTI-360 with Height Kernel

Single terminal — the height kernel is now integrated into the C++ BKI pipeline:

```bash
docker exec -it osm-bki bash
source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
ros2 launch osm_bki kitti360_launch.py
```

The height kernel is controlled by `config/methods/kitti360.yaml`:
- `height_kernel_enabled: true` — master switch
- `height_kernel_lambda: 0.5` — strength (0 = off, 1 = full)
- `height_kernel_mu` / `height_kernel_tau` — per-class Gaussian priors
- `dem_occupancy_strength: 0.5` — free-space evidence above DSM / below DEM
- `dem_occupancy_margin: 1.0` — tolerance in meters

### Optional: DEM/DSM Surface Visualization

In a second terminal:
```bash
docker exec -it osm-bki bash
source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash
python3 /ros2_ws/src/osm_bki/height_kernel/dem_visualizer_node.py
```

In RViz, add PointCloud2 displays for `/dem_surface` and `/dsm_surface`.

## What the Height Kernel Does

### Semantic Height Filtering (k_ht)
For each voxel at height `h` above ground (from DEM), each class k gets a per-class modulation:

```
ybars[k] *= (1 - lambda) + lambda * exp(-(h - mu_k)^2 / (2 * tau_k^2))
```

This means:
- Road predicted at ground level (h=0): phi_road(0) = 1.0, full weight
- Road predicted at 10m: phi_road(10) ~ 0.0, heavily downweighted
- Building predicted at 10m: phi_building(10) ~ 0.97, mostly preserved

The effect: implausible class predictions at wrong heights get suppressed in favor of height-consistent classes.

### DEM/DSM Occupancy Prior
Separately, voxels above the DSM surface or below the DEM surface get extra free-space evidence (`ybars[0] += dem_occupancy_strength`), pushing them toward the unlabeled/free class. This prevents phantom objects in empty air above rooftops or underground.

### Where It Plugs In
The height kernel runs after the existing OSM confusion matrix prior and before the Dirichlet update, in all three `insert_pointcloud` variants in `bkioctomap.cpp`:

```
predict -> ybars[j] -> apply_osm_prior_to_ybars -> apply_height_kernel_to_ybars -> node.update(ybars[j])
```

## Implementation Status

| Component | Status |
|-----------|--------|
| DEM/DSM tile merging (Python) | Done |
| KITTI-360 -> UTM alignment / Umeyama (Python) | Done |
| DEM/DSM query utility (Python) | Done |
| Height kernel k_ht (Python prototype) | Done |
| Precompute binary grids for C++ (Python) | Done |
| C++ binary grid loader + bilinear interp | Done |
| C++ height kernel (per-class phi_k modulation) | Done |
| C++ DEM/DSM occupancy prior | Done |
| Config/launch wiring | Done |
| DEM/DSM RViz visualizer (Python ROS2 node) | Done |

## File Overview

```
height_kernel/
  merge_tiles.py            # Merge GeoTIFF tiles into mosaics
  align_trajectory.py       # Umeyama alignment KITTI-360 -> UTM 32N
  dem_query.py              # DEM/DSM elevation query class (Python)
  height_kernel.py          # k_ht kernel implementation (Python prototype)
  precompute_dem_grid.py    # Convert DEM/DSM to binary grids for C++
  config.yaml               # Python-side config (RViz visualization)
  dem_visualizer_node.py    # ROS2 node: publish DEM/DSM as PointCloud2
  README.md                 # This file

include/osm_bki/common/
  dem_height_query.h        # C++ binary grid loader + bilinear interpolation

src/mapping/
  bkioctomap.cpp            # Modified: apply_height_kernel_to_ybars()

config/methods/
  kitti360.yaml             # Height kernel params (mu, tau, lambda, etc.)
```
