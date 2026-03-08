# OSM-BKI: OpenStreetMap-Enhanced Bayesian Kernel Inference for Semantic Mapping

This project extends [Lu Gan's Semantic BKI (S-BKI)](https://github.com/ganlumomo/BKISemanticMapping) with **OpenStreetMap (OSM) geographic priors** to improve 3D semantic mapping accuracy. The core BKI mapping engine is adapted for ROS2 and augmented with OSM-based prior fusion, uncertainty-aware inference, and multi-dataset support.

## Building on S-BKI

The original S-BKI work by Lu Gan et al. introduced Bayesian Spatial Kernel Smoothing for scalable dense semantic mapping. It uses Bayesian Kernel Inference to propagate semantic observations through an octree-based 3D map, producing continuous semantic predictions with calibrated uncertainty.

**Original paper:**

> L. Gan, R. Zhang, J. W. Grizzle, R. M. Eustice, and M. Ghaffari, "Bayesian Spatial Kernel Smoothing for Scalable Dense Semantic Mapping," *IEEE Robotics and Automation Letters (RA-L)*, 2020.

**Original repository:** [github.com/ganlumomo/BKISemanticMapping](https://github.com/ganlumomo/BKISemanticMapping)

## What OSM-BKI Adds

This project extends S-BKI in several key ways:

### OSM Prior Integration

- **Geographic priors from OpenStreetMap**: Buildings, roads, sidewalks, cycleways, parking areas, grasslands, trees, forests, fences, walls, stairs, water bodies, and poles are loaded from `.osm` files and projected into the map frame.
- **OSM confusion matrix fusion**: A learned confusion matrix maps semantic class predictions to OSM categories, biasing BKI inference toward OSM-consistent classes. The matrix can be optimized from ground-truth co-occurrence data.
- **Point-level and voxel-level prior application**: OSM priors can be applied directly to per-point softmax distributions (before BKI insertion) or at the voxel level (during prediction).
- **Height-based filtering**: OSM priors are modulated by a height confusion matrix, so ground-level priors (roads, sidewalks) don't influence elevated structures and vice versa.
- **Configurable decay**: Prior influence drops off smoothly with distance from OSM geometry boundaries.

### Multi-Dataset Support

- **KITTI-360**: Velodyne LiDAR with 4x4 pose matrices, Mercator-projected OSM alignment.
- **MCD (Multi-Campus Dataset)**: Body-frame poses with quaternion orientation and body-to-lidar calibration.
- **CU North Campus**: Custom campus dataset with aligned LiDAR and inferred labels.
- **Common taxonomy**: All datasets are mapped to a shared 13-class semantic label space for consistent evaluation.

### Multiclass Confidence Scores

- Ingests per-point softmax distributions (e.g., from CENet) rather than hard labels, preserving prediction uncertainty through the BKI pipeline.
- Supports uncertainty-based filtering: discard or down-weight highly uncertain observations using per-class precision thresholds or top-N% discounting.

### ROS2 and Visualization

- Full ROS2 (Humble+) port of the original ROS1 codebase.
- Real-time RViz visualization of semantic maps, OSM geometry overlays, variance maps, and semantic uncertainty maps.
- Published PointCloud2 scans with label-based RGB coloring in the map frame.
- OSM geometry visualization as MarkerArray (buildings, roads, vegetation, etc.).

### Evaluation Pipeline

- Per-scan query against ground-truth labels with IoU and accuracy metrics.
- Results written under the sequence directory for organized multi-run comparisons.

## Dependencies

- ROS2 (Humble or later)
- PCL (Point Cloud Library)
- Eigen3
- OpenCV
- yaml-cpp
- libosmium (`sudo apt install libosmium2-dev`)

## Building

```bash
mkdir -p ros2_ws/src #(wherever you choose)
cd ros2_ws/src
git clone 
cd ../
colcon build --packages-select osm_bki
source install/setup.bash
```

## Usage

### KITTI-360

```bash
ros2 launch osm_bki kitti360_launch.py
```

### MCD (Multi-Campus Dataset)

```bash
ros2 launch osm_bki mcd_with_osm_launch.py
```

### CU North Campus

```bash
ros2 launch osm_bki cu_north_campus_launch.py
```

### Configuration

Dataset-specific parameters are in `config/methods/`:

- `kitti360.yaml` — KITTI-360 sequence, scan count, resolution, OSM file, prior strength
- `mcd.yaml` — MCD dataset configuration
- `cu_north_campus.yaml` — North Campus configuration

OSM confusion matrices are in `config/datasets/` and can be re-optimized:

```bash
python3 optimize_osm_cm_scaled_kitti360.py --output config/datasets/osm_confusion_matrix_optimized_kitti360.yaml
```

## Acknowledgments

This work builds directly on Lu Gan's [BKISemanticMapping](https://github.com/ganlumomo/BKISemanticMapping). The core octree structure, Bayesian kernel inference engine, and semantic octree node representation originate from that project. We gratefully acknowledge the original authors for making their code publicly available.