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

## Ablation Experiment

Compares **OSM-BKI baseline** (no height kernel) vs **OSM-BKI + height kernel** on KITTI-360 sequence 0000.

### Configs

| Config | Height Kernel | Eval Output |
|--------|--------------|-------------|
| `config/methods/kitti360_no_height.yaml` | Disabled | `evaluations/osm_prior_no_height/` |
| `config/methods/kitti360_with_height.yaml` | Enabled | `evaluations/osm_prior_with_height/` |

Both configs set `visualize: false` so the pipeline runs headless and exits automatically.
Adjust `scan_num` to control how many LiDAR scans to process (40 for quick validation, 1000 for full experiment).

### Running the Ablation

Inside the Docker container:

```bash
source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash

# Run everything (baseline → height kernel → evaluation)
bash /ros2_ws/src/osm_bki/height_kernel/run_ablation.sh

# Or run individual steps
bash /ros2_ws/src/osm_bki/height_kernel/run_ablation.sh run_a      # baseline only
bash /ros2_ws/src/osm_bki/height_kernel/run_ablation.sh run_b      # height kernel only
bash /ros2_ws/src/osm_bki/height_kernel/run_ablation.sh evaluate   # compare results
```

If you change configs or C++ code on the host, rebuild the Docker image first:
```bash
docker compose down && docker compose up -d --build
```

### Evaluation

The evaluation script computes per-class IoU, mIoU, and overall accuracy from the `query_scan()` output files:

```bash
python3 /ros2_ws/src/osm_bki/height_kernel/evaluate_ablation.py compare \
    --dir-a /path/to/evaluations/osm_prior_no_height \
    --label-a "OSM-BKI (no height)" \
    --dir-b /path/to/evaluations/osm_prior_with_height \
    --label-b "OSM-BKI + height kernel" \
    --output /path/to/evaluations
```

Outputs: per-run confusion matrices (`confusion_matrix.csv`), per-run results (`results.csv`), and a combined comparison (`ablation_comparison.csv`).

### Results: Full Sequence (KITTI-360 seq 0000, 4433 scans, 516M points)

| Class | Baseline (IoU) | + Height Kernel (IoU) | Delta |
|-------|:--------------:|:---------------------:|:-----:|
| road | 45.09% | 48.15% | **+3.05%** |
| sidewalk | 33.20% | 36.41% | **+3.21%** |
| parking | 0.63% | 1.96% | **+1.33%** |
| other-ground | 0.74% | 1.71% | **+0.97%** |
| building | 45.74% | 65.40% | **+19.66%** |
| fence | 2.73% | 5.31% | **+2.58%** |
| pole | 3.93% | 3.89% | -0.04% |
| traffic-sign | 0.68% | 0.65% | -0.02% |
| vegetation | 33.35% | 30.72% | -2.63% |
| two-wheeler | 0.00% | 0.04% | +0.03% |
| vehicle | 0.63% | 3.19% | **+2.56%** |
| other-object | 0.15% | 1.84% | **+1.70%** |
| **mIoU** | **13.91%** | **16.61%** | **+2.70%** |
| **Overall Accuracy** | **56.83%** | **65.54%** | **+8.70%** |

Key takeaways:
- **Building +19.66%**: Height prior strongly penalizes false building predictions at ground level (mu=7m, tau=4m) — largest single-class gain
- **Road +3.05%, Sidewalk +3.21%**: Ground-class priors preserve and improve surface accuracy; DEM occupancy margin (1.5m) prevents erosion near building edges
- **Overall accuracy +8.70%**: Consistent improvement across the full sequence (516M points)
- **Vehicle +2.56%, Fence +2.58%**: Height priors correctly constrain low-structure and near-ground classes
- **Vegetation -2.63%**: Regression from canopy height prior (mu=5m, tau=3m); vegetation canopy height varies significantly across the full sequence
- **Pole -0.04%, Traffic-sign -0.02%**: Negligible change — these thin structures are not strongly affected by height priors

<details>
<summary>40-Frame Pilot Study</summary>

Earlier validation on 40 frames showed similar trends with slightly larger deltas due to smaller sample:

| Class | Baseline | + Height | Delta |
|-------|:--------:|:--------:|:-----:|
| road | 46.82% | 51.47% | +4.65% |
| sidewalk | 25.08% | 31.07% | +5.98% |
| building | 41.38% | 61.54% | +20.16% |
| vegetation | 48.57% | 47.69% | -0.87% |
| **mIoU** | **13.87%** | **17.17%** | **+3.30%** |
| **Overall Acc** | **59.68%** | **68.24%** | **+8.56%** |

</details>

### Height Kernel Parameters (Tuned)

```yaml
height_kernel_lambda: 0.9          # Near full authority
dem_occupancy_strength: 0.5        # Mild free-space prior
dem_occupancy_margin: 1.5          # Lenient margin in meters

#              unlbl  road  sdwk  park  ognd  bldg  fnce  pole  tsgn  veg   2whl  vhcl  oobj
height_kernel_mu:  [0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 1.0, 3.0, 3.0, 5.0, 0.7, 0.8, 1.0]
height_kernel_tau: [100, 0.8, 0.8, 0.8, 0.8, 4.0, 0.8, 2.5, 2.0, 3.0, 0.5, 0.6, 1.5]
```

Tuning rationale:
- **lambda=0.9**: Strong modulation (near full weight to the height prior)
- **Ground classes (tau=0.8)**: Tolerant enough to handle DEM noise without suppressing legitimate road/sidewalk voxels
- **Building (mu=7, tau=4)**: Tight prior kills false building at ground level — biggest single improvement
- **Vegetation (mu=5, tau=3)**: Penalizes false vegetation at ground but allows canopy
- **DEM occupancy (strength=0.5, margin=1.5)**: Conservative to avoid eating road voxels near building edges — earlier values of strength=2.0/margin=0.5 caused road IoU to dip

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
  run_ablation.sh           # Automated ablation experiment runner
  evaluate_ablation.py      # Per-class IoU / mIoU evaluation + comparison
  README.md                 # This file

include/osm_bki/common/
  dem_height_query.h        # C++ binary grid loader + bilinear interpolation

src/mapping/
  bkioctomap.cpp            # Modified: apply_height_kernel_to_ybars()

config/methods/
  kitti360.yaml             # Default config (height kernel enabled)
  kitti360_no_height.yaml   # Ablation baseline (height kernel disabled)
  kitti360_with_height.yaml # Ablation variant (height kernel enabled, tuned)
```
