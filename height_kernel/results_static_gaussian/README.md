# Static Gaussian Height Kernel Results (No DEM)

Branch: ssg-static-gaussians
Date: April 15, 2026
Approach: Scan-relative height estimation. Ground level estimated as `origin_z - sensor_mounting_height`. No DEM raster lookup required.

## Config

| Parameter | Value |
|---|---|
| sensor_mounting_height | 1.73 (KITTI-360) |
| dem_occupancy_strength | 0.0 (disabled) |

### Experiment 1 (cross-domain, `kitti360_with_height.yaml`)
- `height_kernel_lambda`: 0.9
- `height_kernel_dead_zone`: 1.5
- `height_kernel_redistribute`: false (suppress mode)
- `height_kernel_mu`: `[0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 1.0, 3.0, 3.0, 0.0, 0.7, 0.8, 1.0, 5.0]`
- `height_kernel_tau`: `[100.0, 0.8, 0.8, 0.8, 0.8, 4.0, 0.8, 2.5, 2.0, 0.5, 0.5, 0.6, 1.5, 100.0]`

### Experiment 2 (in-domain seq 0000, `kitti360_0000_indomain_with_height.yaml`)
- `height_kernel_lambda`: 0.9
- `height_kernel_dead_zone`: 0.5
- `height_kernel_redistribute`: true
- `height_kernel_mu`: `[0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 1.0, 3.0, 3.0, 0.0, 0.7, 0.8, 1.0, 5.0]`
- `height_kernel_tau`: `[100000.0, 0.3, 100000.0, 0.5, 0.5, 100000.0, 100000.0, 100000.0, 100000.0, 0.5, 100000.0, 0.5, 100000.0, 100000.0]`

### Experiment 3 (in-domain seq 0009, `kitti360_0009_indomain_with_height.yaml`)
- `height_kernel_lambda`: 0.8
- `height_kernel_dead_zone`: 0.75
- `height_kernel_redistribute`: true
- `height_kernel_mu`: `[0.0, 0.0, 0.0, 0.0, 0.0, 7.0, 1.0, 3.0, 3.0, 0.0, 0.7, 0.8, 1.0, 5.0]`
- `height_kernel_tau`: `[100000.0, 0.5, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 100000.0, 0.5, 0.4, 100000.0, 100000.0, 100000.0]`

---

## Experiment 1: Cross-Domain CENet-MCD -> KITTI-360 seq 0000 (14-class)

Scans: 4433, Points: 516,744,189

| Class | Baseline | Static Gaussian | Delta |
|---|---:|---:|---:|
| road | 45.11% | 44.84% | -0.27% |
| sidewalk | 33.22% | 31.51% | -1.71% |
| parking | 0.63% | 1.02% | +0.39% |
| other-ground | 0.74% | 1.11% | +0.37% |
| building | 45.86% | 56.44% | +10.58% |
| fence | 2.73% | 6.29% | +3.56% |
| pole | 3.95% | 5.07% | +1.12% |
| traffic-sign | 0.68% | 1.30% | +0.62% |
| terrain | 15.99% | 16.07% | +0.08% |
| two-wheeler | 0.00% | 0.30% | +0.29% |
| vehicle | 0.63% | 2.87% | +2.24% |
| other-object | 0.15% | 1.00% | +0.85% |
| vegetation | 14.49% | 22.76% | +8.27% |
| **mIoU** | **12.63%** | **14.66%** | **+2.03%** |
| **Overall Acc** | **50.61%** | **53.55%** | **+2.94%** |

---

## Experiment 2: In-Domain CENet-KITTI360 -> KITTI-360 seq 0000 (14-class)

Scans: 4433, Points: 516,744,189

| Class | Baseline | Static Gaussian | Delta |
|---|---:|---:|---:|
| road | 52.97% | 56.66% | +3.69% |
| sidewalk | 45.88% | 45.17% | -0.71% |
| parking | 19.18% | 19.86% | +0.68% |
| other-ground | 8.47% | 8.58% | +0.10% |
| building | 67.02% | 67.02% | +0.00% |
| fence | 21.69% | 21.46% | -0.23% |
| pole | 10.54% | 9.71% | -0.83% |
| traffic-sign | 13.41% | 14.37% | +0.96% |
| terrain | 39.74% | 40.34% | +0.60% |
| two-wheeler | 7.31% | 7.08% | -0.23% |
| vehicle | 30.03% | 34.78% | +4.75% |
| other-object | 7.96% | 6.99% | -0.97% |
| vegetation | 45.14% | 46.45% | +1.31% |
| **mIoU** | **28.41%** | **29.11%** | **+0.70%** |
| **Overall Acc** | **68.48%** | **69.59%** | **+1.11%** |

---

## Experiment 3: In-Domain CENet-KITTI360 -> KITTI-360 seq 0009 (14-class)

Scans: 5566, Points: 650,828,899

| Class | Baseline | Static Gaussian | Delta |
|---|---:|---:|---:|
| road | 64.67% | 65.69% | +1.02% |
| sidewalk | 56.08% | 55.76% | -0.32% |
| parking | 25.48% | 25.47% | -0.01% |
| other-ground | 11.56% | 11.40% | -0.16% |
| building | 65.74% | 65.74% | -0.00% |
| fence | 20.83% | 20.91% | +0.08% |
| pole | 13.45% | 13.48% | +0.03% |
| traffic-sign | 18.63% | 18.73% | +0.10% |
| terrain | 24.21% | 24.94% | +0.73% |
| two-wheeler | 8.35% | 8.39% | +0.05% |
| vehicle | 27.99% | 29.05% | +1.06% |
| other-object | 12.33% | 12.26% | -0.06% |
| vegetation | 49.19% | 50.02% | +0.83% |
| **mIoU** | **30.65%** | **30.91%** | **+0.26%** |
| **Overall Acc** | **70.91%** | **71.37%** | **+0.46%** |
