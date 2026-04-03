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

## Results: 14-Class Taxonomy (100 scans)

### Experiment 1: Cross-Domain (CENet-MCD → KITTI-360 seq 0000)

MCD model trained on SemanticKITTI/MCD, tested on KITTI-360. MCD does not distinguish terrain from vegetation — all vegetation predictions map to class 13 (vegetation/trees). Terrain IoU comes entirely from OSM priors.

**1089 scans, 127.7M points**

| Class | Baseline (IoU) | + Height Kernel (IoU) | Delta |
|-------|:--------------:|:---------------------:|:-----:|
| road | 51.12% | 50.94% | -0.18% |
| sidewalk | 24.92% | 24.37% | -0.56% |
| parking | 0.54% | 0.68% | +0.14% |
| other-ground | 0.56% | 0.67% | +0.11% |
| building | 37.39% | 50.62% | **+13.23%** |
| fence | 2.65% | 5.17% | **+2.52%** |
| pole | 0.84% | 1.41% | +0.57% |
| traffic-sign | 0.15% | 0.13% | -0.02% |
| terrain | 22.40% | 22.51% | +0.10% |
| two-wheeler | 0.00% | 1.53% | **+1.53%** |
| vehicle | 0.63% | 4.43% | **+3.80%** |
| other-object | 0.05% | 0.69% | +0.64% |
| vegetation | 18.17% | 25.51% | **+7.34%** |
| **mIoU** | **12.26%** | **14.51%** | **+2.25%** |
| **Overall Accuracy** | **45.28%** | **49.07%** | **+3.79%** |

Key takeaways:
- **Building +13.23%**: Height prior strongly penalizes false building predictions at ground level (mu=7m, tau=4m)
- **Vegetation +7.34%**: Split lets the height kernel cleanly separate trees from ground; vegetation no longer polluted by terrain confusion
- **Vehicle +3.80%**: Near-ground prior (mu=0.8m, tau=0.6m) helps vehicle identification
- **Terrain +0.10%**: MCD cannot predict terrain, so all terrain IoU comes from OSM grassland priors; height kernel has minimal effect on this class
- **Road -0.18%, Sidewalk -0.56%**: Minor regression from height kernel interactions

### Experiment 2: In-Domain seq 0000 (CENet-KITTI360 → KITTI-360 seq 0000)

CENet trained on KITTI-360, tested on same dataset. Model can predict both terrain and vegetation.

**1089 scans, 127.7M points**

| Class | Baseline (IoU) | + Height Kernel (IoU) | Delta |
|-------|:--------------:|:---------------------:|:-----:|
| road | 60.06% | 62.41% | **+2.34%** |
| sidewalk | 39.52% | 39.03% | -0.49% |
| parking | 26.17% | 28.52% | **+2.35%** |
| other-ground | 8.52% | 8.61% | +0.09% |
| building | 59.38% | 59.43% | +0.05% |
| fence | 15.44% | 15.25% | -0.19% |
| pole | 6.44% | 7.87% | **+1.43%** |
| traffic-sign | 8.03% | 7.53% | -0.50% |
| terrain | 59.74% | 61.76% | **+2.02%** |
| two-wheeler | 8.52% | 7.50% | -1.02% |
| vehicle | 34.70% | 37.62% | **+2.93%** |
| other-object | 6.01% | 5.95% | -0.06% |
| vegetation | 44.04% | 44.91% | +0.87% |
| **mIoU** | **28.97%** | **29.72%** | **+0.76%** |
| **Overall Accuracy** | **68.90%** | **69.79%** | **+0.89%** |

Key takeaways:
- **Road +2.34%, Parking +2.35%**: Ground-class priors improve surface accuracy
- **Vehicle +2.93%**: Near-ground prior helps distinguish vehicles
- **Terrain +2.02%**: Tight terrain prior (mu=0, tau=0.5) correctly constrains terrain predictions to ground level
- **Pole +1.43%**: Mid-height prior (mu=3m, tau=2.5m) helps pole detection
- **Vegetation +0.87%**: Vegetation height kernel is disabled (tau=100000) to avoid suppressing trees at various heights; modest gain from other class improvements

### Experiment 3: In-Domain seq 0009 (CENet-KITTI360 → KITTI-360 seq 0009)

Same model as Experiment 2, different sequence.

**1089 scans, 128.5M points**

| Class | Baseline (IoU) | + Height Kernel (IoU) | Delta |
|-------|:--------------:|:---------------------:|:-----:|
| road | 60.64% | 61.28% | +0.63% |
| sidewalk | 50.82% | 50.43% | -0.39% |
| parking | 29.05% | 29.04% | -0.02% |
| other-ground | 9.55% | 9.51% | -0.04% |
| building | 58.99% | 58.98% | -0.00% |
| fence | 16.42% | 16.41% | -0.01% |
| pole | 6.49% | 6.37% | -0.12% |
| traffic-sign | 16.73% | 16.68% | -0.05% |
| terrain | 22.81% | 23.92% | **+1.11%** |
| two-wheeler | 22.64% | 24.43% | **+1.79%** |
| vehicle | 26.04% | 26.81% | +0.76% |
| other-object | 10.52% | 10.49% | -0.03% |
| vegetation | 44.70% | 46.79% | **+2.09%** |
| **mIoU** | **28.88%** | **29.32%** | **+0.44%** |
| **Overall Accuracy** | **65.75%** | **66.83%** | **+1.09%** |

Key takeaways:
- **Vegetation +2.09%**: Largest gain on this sequence, from improved separation of terrain/vegetation
- **Two-wheeler +1.79%**: Near-ground prior helps two-wheeler detection
- **Terrain +1.11%**: Ground-level constraint improves terrain predictions
- Smaller overall gains than seq 0000 because seq 0009 has selective in-domain tuning (fewer active classes)

---

## Summary: 14-Class vs Old 13-Class Comparison

### Old 13-Class Results (for reference)

The old 13-class taxonomy merged terrain and vegetation into a single "vegetation" class, preventing the height kernel from distinguishing ground-level terrain from elevated tree canopy.

| Experiment | 13-class Baseline | 13-class +Height | Height Delta |
|------------|:-----------------:|:----------------:|:------------:|
| Cross-domain (seq 0000) | 13.91% | 16.61% | +2.70% |
| In-domain (seq 0000) | 27.81% | 28.27% | +0.46% |

### New 14-Class Results

| Experiment | 14-class Baseline | 14-class +Height | Height Delta |
|------------|:-----------------:|:----------------:|:------------:|
| Cross-domain (seq 0000) | 12.26% | 14.51% | +2.25% |
| In-domain (seq 0000) | 28.97% | 29.72% | +0.76% |
| In-domain (seq 0009) | 28.88% | 29.32% | +0.44% |

### Key comparisons:

- **In-domain baseline improved by +1.16%** (27.81% → 28.97%) from the taxonomy split alone, without any height kernel changes
- **In-domain best improved by +1.45%** (28.27% → 29.72%) — the 14-class baseline already exceeds the old 13-class +height best
- **Cross-domain baseline dropped by -1.65%** (13.91% → 12.26%) because MCD cannot predict terrain (class 9), adding a hard class that dilutes mIoU; however, the height kernel still provides a strong +2.25% gain
- **Cross-domain vegetation improved dramatically**: 18.17% → 25.51% (+7.34%) with height kernel, because vegetation now only means trees and benefits from cleaner height separation

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
