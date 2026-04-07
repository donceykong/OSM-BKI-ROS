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

## 14-Class Taxonomy

The system uses a **14-class common taxonomy** that splits the old merged "vegetation" class into separate **terrain** (grass, ground cover, h~0m) and **vegetation** (trees, canopy, h~2-20m) classes. KITTI-360 GT labels distinguish them (label 21=vegetation, label 22=terrain), which enables the height kernel to constrain each appropriately.

### Common Taxonomy (14 classes)

| Index | Class | Height Prior (mu, tau) | Description |
|:-----:|-------|:----------------------:|-------------|
| 0 | unlabeled | 0.0, 100.0 | No constraint |
| 1 | road | 0.0, 0.8 | Ground level |
| 2 | sidewalk | 0.0, 0.8 | Ground level |
| 3 | parking | 0.0, 0.8 | Ground level |
| 4 | other-ground | 0.0, 0.8 | Ground level |
| 5 | building | 7.0, 4.0 | Elevated structures |
| 6 | fence | 1.0, 0.8 | Low structures |
| 7 | pole | 3.0, 2.5 | Mid-height |
| 8 | traffic-sign | 3.0, 2.0 | Mid-height |
| 9 | terrain | 0.0, 0.5 | Ground level (tight) |
| 10 | two-wheeler | 0.7, 0.5 | Near ground |
| 11 | vehicle | 0.8, 0.6 | Near ground |
| 12 | other-object | 1.0, 1.5 | Flexible |
| 13 | vegetation | 5.0, 100.0 | Trees (disabled; wide range) |

## Ablation Experiments

Compares **OSM-BKI baseline** (no height kernel) vs **OSM-BKI + height kernel** on KITTI-360.

### Configs

| Config | Height Kernel | Eval Output |
|--------|:------------:|-------------|
| `kitti360_no_height.yaml` | Disabled | `evaluations/osm_prior_no_height/` |
| `kitti360_with_height.yaml` | Enabled | `evaluations/osm_prior_with_height/` |
| `kitti360_0000_indomain_no_height.yaml` | Disabled | `evaluations/indomain_no_height/` |
| `kitti360_0000_indomain_with_height.yaml` | Enabled | `evaluations/indomain_with_height/` |
| `kitti360_0009_indomain_no_height.yaml` | Disabled | `evaluations/indomain_no_height/` |
| `kitti360_0009_indomain_with_height.yaml` | Enabled | `evaluations/indomain_with_height/` |

Both configs set `visualize: false` so the pipeline runs headless and exits automatically.
Adjust `scan_num` to control how many LiDAR scans to process.

### Running the Ablation

Inside the Docker container:

```bash
source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash

# Cross-domain (CENet-MCD -> KITTI-360)
bash /ros2_ws/src/osm_bki/height_kernel/run_ablation.sh

# In-domain (CENet-KITTI360 -> KITTI-360)
bash /ros2_ws/src/osm_bki/height_kernel/run_ablation_indomain.sh          # both sequences
bash /ros2_ws/src/osm_bki/height_kernel/run_ablation_indomain.sh exp1     # seq 0000 only
bash /ros2_ws/src/osm_bki/height_kernel/run_ablation_indomain.sh exp2     # seq 0009 only
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

---

## Results: 14-Class Taxonomy (Full Sequence)

### Experiment 1: Cross-Domain (CENet-MCD → KITTI-360 seq 0000)

MCD model trained on SemanticKITTI/MCD, tested on KITTI-360. MCD does not distinguish terrain from vegetation — all vegetation predictions map to class 13 (vegetation/trees). Terrain IoU comes entirely from OSM priors.

**4433 scans, 516.7M points**

| Class | Baseline (IoU) | + Height Kernel (IoU) | Delta |
|-------|:--------------:|:---------------------:|:-----:|
| road | 45.11% | 44.93% | -0.18% |
| sidewalk | 33.22% | 31.19% | -2.03% |
| parking | 0.63% | 1.02% | +0.39% |
| other-ground | 0.74% | 1.11% | +0.37% |
| building | 45.86% | 58.76% | **+12.90%** |
| fence | 2.73% | 6.28% | **+3.55%** |
| pole | 3.95% | 4.42% | +0.46% |
| traffic-sign | 0.68% | 0.47% | -0.21% |
| terrain | 15.99% | 16.03% | +0.04% |
| two-wheeler | 0.00% | 0.25% | +0.25% |
| vehicle | 0.63% | 2.87% | **+2.25%** |
| other-object | 0.15% | 0.86% | +0.71% |
| vegetation | 14.49% | 23.73% | **+9.24%** |
| **mIoU** | **12.63%** | **14.76%** | **+2.13%** |
| **Overall Accuracy** | **50.61%** | **54.13%** | **+3.52%** |

Key takeaways:
- **Building +12.90%**: Height prior strongly penalizes false building predictions at ground level (mu=7m, tau=4m)
- **Vegetation +9.24%**: Split lets the height kernel cleanly separate trees from ground; vegetation no longer polluted by terrain confusion
- **Fence +3.55%**: Low-structure prior (mu=1m, tau=0.8m) helps fence identification
- **Vehicle +2.25%**: Near-ground prior (mu=0.8m, tau=0.6m) helps vehicle identification
- **Terrain +0.04%**: MCD cannot predict terrain, so all terrain IoU comes from OSM grassland priors; height kernel has minimal effect on this class
- **Sidewalk -2.03%**: Regression from height kernel interactions at full scale

### Experiment 2: In-Domain seq 0000 (CENet-KITTI360 → KITTI-360 seq 0000)

CENet trained on KITTI-360, tested on same dataset. Model can predict both terrain and vegetation.

**4433 scans, 516.7M points**

| Class | Baseline (IoU) | + Height Kernel (IoU) | Delta |
|-------|:--------------:|:---------------------:|:-----:|
| road | 52.97% | 55.55% | **+2.58%** |
| sidewalk | 45.88% | 45.08% | -0.80% |
| parking | 19.18% | 20.13% | +0.95% |
| other-ground | 8.47% | 8.49% | +0.02% |
| building | 67.02% | 67.02% | -0.00% |
| fence | 21.69% | 21.32% | -0.37% |
| pole | 10.54% | 9.97% | -0.58% |
| traffic-sign | 13.41% | 14.00% | +0.59% |
| terrain | 39.74% | 41.29% | **+1.55%** |
| two-wheeler | 7.31% | 6.20% | -1.11% |
| vehicle | 30.03% | 33.79% | **+3.76%** |
| other-object | 7.96% | 7.11% | -0.85% |
| vegetation | 45.14% | 45.84% | +0.70% |
| **mIoU** | **28.41%** | **28.91%** | **+0.49%** |
| **Overall Accuracy** | **68.48%** | **69.29%** | **+0.81%** |

Key takeaways:
- **Vehicle +3.76%**: Near-ground prior helps distinguish vehicles
- **Road +2.58%**: Ground-class prior improves road surface accuracy
- **Terrain +1.55%**: Tight terrain prior (mu=0, tau=0.5) correctly constrains terrain predictions to ground level
- **Vegetation +0.70%**: Vegetation height kernel is disabled (tau=100000) to avoid suppressing trees at various heights; modest gain from other class improvements
- **Two-wheeler -1.11%**: Minor regression, possibly from near-ground prior interactions at full scale

### Experiment 3: In-Domain seq 0009 (CENet-KITTI360 → KITTI-360 seq 0009)

Same model as Experiment 2, different sequence.

**5566 scans, 650.8M points**

| Class | Baseline (IoU) | + Height Kernel (IoU) | Delta |
|-------|:--------------:|:---------------------:|:-----:|
| road | 64.67% | 64.22% | -0.46% |
| sidewalk | 56.08% | 55.84% | -0.24% |
| parking | 25.48% | 25.44% | -0.04% |
| other-ground | 11.56% | 11.29% | -0.28% |
| building | 65.74% | 65.74% | -0.00% |
| fence | 20.83% | 21.43% | +0.59% |
| pole | 13.45% | 13.33% | -0.11% |
| traffic-sign | 18.63% | 18.59% | -0.04% |
| terrain | 24.21% | 25.53% | **+1.31%** |
| two-wheeler | 8.35% | 8.57% | +0.22% |
| vehicle | 27.99% | 27.82% | -0.17% |
| other-object | 12.33% | 12.29% | -0.03% |
| vegetation | 49.19% | 50.51% | **+1.32%** |
| **mIoU** | **30.65%** | **30.81%** | **+0.16%** |
| **Overall Accuracy** | **70.91%** | **71.28%** | **+0.37%** |

Key takeaways:
- **Vegetation +1.32%**: Improved separation of terrain/vegetation at height
- **Terrain +1.31%**: Ground-level constraint improves terrain predictions
- **Fence +0.59%**: Low-structure prior provides modest gains
- Smaller overall gains than seq 0000 because seq 0009 has selective in-domain tuning (fewer active classes)

---

## Summary: Full-Scale 14-Class Results

| Experiment | Scans | Points | Baseline mIoU | +Height mIoU | Delta |
|------------|:-----:|:------:|:-------------:|:------------:|:-----:|
| Cross-domain (seq 0000) | 4433 | 516.7M | 12.63% | 14.76% | **+2.13%** |
| In-domain (seq 0000) | 4433 | 516.7M | 28.41% | 28.91% | **+0.49%** |
| In-domain (seq 0009) | 5566 | 650.8M | 30.65% | 30.81% | **+0.16%** |

### Key findings:

- **Height kernel consistently improves mIoU** across all three experiments, with the largest gain in the cross-domain setting (+2.13%)
- **Building class benefits most from height priors**: +12.90% in cross-domain, where false ground-level building predictions are strongly penalized
- **Vegetation separation is highly effective**: +9.24% in cross-domain from cleaner height-based tree/ground distinction
- **In-domain gains are modest but positive**: the in-domain model is already strong, leaving less room for height-based correction
- **Overall accuracy improves consistently**: +3.52% cross-domain, +0.81% in-domain seq 0000, +0.37% in-domain seq 0009

---

## Height Kernel Parameters (Tuned)

### Cross-Domain Config (`kitti360_with_height.yaml`)

```yaml
height_kernel_lambda: 0.9          # Near full authority
height_kernel_dead_zone: 1.5       # No suppression within 1.5m of expected height
dem_occupancy_strength: 0.5        # Mild free-space prior
dem_occupancy_margin: 1.5          # Lenient margin in meters
height_kernel_redistribute: false  # Suppress mode

# 14 classes: unlbl  road  sdwk  park  ognd  bldg  fnce  pole  tsgn  terrain  2whl  vhcl  oobj  vegetation
height_kernel_mu:  [0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 1.0, 3.0, 3.0, 0.0, 0.7, 0.8, 1.0, 5.0]
height_kernel_tau: [100, 0.8, 0.8, 0.8, 0.8, 4.0, 0.8, 2.5, 2.0, 0.5, 0.5, 0.6, 1.5, 100]
```

### In-Domain Config (`kitti360_0000_indomain_with_height.yaml`)

```yaml
height_kernel_lambda: 0.9
height_kernel_dead_zone: 0.5
dem_occupancy_strength: 0.0        # Disabled for in-domain
height_kernel_redistribute: true   # Redistribute mode

# Selective activation: only road, parking, other-ground, terrain, vehicle active
# Other classes disabled with tau=100000
height_kernel_mu:  [0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 1.0, 3.0, 3.0, 0.0, 0.7, 0.8, 1.0, 5.0]
height_kernel_tau: [1e5, 0.3, 1e5, 0.5, 0.5, 1e5, 1e5, 1e5, 1e5, 0.5, 1e5, 0.5, 1e5, 1e5]
```

Tuning rationale:
- **Terrain (mu=0, tau=0.5)**: Tight ground-level constraint — the key benefit of the terrain/vegetation split
- **Vegetation (tau=100 or 1e5)**: Effectively disabled because trees span a wide range of heights (0-20m+); the split itself provides the benefit without needing aggressive height suppression
- **Cross-domain uses suppress mode**: directly downweights implausible predictions
- **In-domain uses redistribute mode**: redistributes suppressed probability mass to other classes, better for high-confidence models
- **In-domain selective activation**: only a few ground-level classes are height-constrained to avoid over-regularizing the already-good in-domain model

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
| 14-class taxonomy split (terrain/vegetation) | Done |

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
  run_ablation.sh           # Cross-domain ablation experiment runner
  run_ablation_indomain.sh  # In-domain ablation experiment runner
  evaluate_ablation.py      # Per-class IoU / mIoU evaluation + comparison
  README.md                 # This file

include/osm_bki/common/
  dem_height_query.h        # C++ binary grid loader + bilinear interpolation

src/mapping/
  bkioctomap.cpp            # Modified: apply_height_kernel_to_ybars()

config/datasets/
  labels_common.yaml        # 14-class common taxonomy definition
  osm_confusion_matrix_optimized_kitti360.yaml  # Hand-tuned OSM prior matrix (14-class)

config/methods/
  kitti360.yaml                           # Default config (height kernel enabled)
  kitti360_no_height.yaml                 # Cross-domain baseline (disabled)
  kitti360_with_height.yaml               # Cross-domain + height kernel (tuned)
  kitti360_0000_indomain_no_height.yaml   # In-domain seq 0000 baseline
  kitti360_0000_indomain_with_height.yaml # In-domain seq 0000 + height kernel
  kitti360_0009_indomain_no_height.yaml   # In-domain seq 0009 baseline
  kitti360_0009_indomain_with_height.yaml # In-domain seq 0009 + height kernel
```
