# Height Kernel Improvement Ceiling Analysis — Need Your Help

I'm working on OSM-BKI, a Bayesian Kernel Inference (BKI) 3D semantic mapping system that fuses LiDAR semantic segmentation with OpenStreetMap priors. I've added a DEM-based height consistency kernel that modulates per-class predictions based on height above ground. It works great in one setting but barely helps in another, and I need to understand why and how to fix it.

## The System Pipeline

```
Raw LiDAR scan → CENet (neural net, frozen) → per-point semantic labels
  → BKI voxel octree insertion
  → OSM confusion matrix prior (spatial: road/building/tree from OpenStreetMap)
  → Height kernel (elevation: suppress classes at wrong height above DEM)
  → Dirichlet posterior update → final voxel labels
```

The height kernel formula per voxel:
```
h = voxel_z - DEM_elevation(voxel_x, voxel_y)    # height above ground
For each class k:
  phi_k = exp(-((|h - mu_k| - dead_zone)^2) / (2 * tau_k^2))   if |h - mu_k| > dead_zone, else 1.0
  ybars[k] *= (1 - lambda) + lambda * phi_k
```

13-class common taxonomy: unlabeled, road, sidewalk, parking, other-ground, building, fence, pole, traffic-sign, vegetation, two-wheeler, vehicle, other-object.

## Two Settings: Cross-Domain vs In-Domain

### Cross-Domain: CENet trained on MCD (Stockholm), tested on KITTI-360 (Karlsruhe)
- The neural net has never seen Karlsruhe → predictions are noisy
- **Baseline mIoU: 13.91%** (with OSM prior, no height kernel)
- **With height kernel: 16.61%** → **+2.70% mIoU, +8.70% overall accuracy**
- Building went from 45.74% → 65.40% (+19.66%)

### In-Domain: CENet trained on KITTI-360, tested on KITTI-360
- The neural net has seen this domain → predictions are much better
- **Baseline mIoU: 27.81%** (with OSM prior, no height kernel)
- **Best with height kernel: 28.27%** → only **+0.46% mIoU, +0.44% OA**
- I need 10-20% improvement, not 0.46%

## Confusion Matrix Analysis (This Is the Key Data)

### Cross-Domain Confusion (where height kernel helps hugely):
```
road:          65.0% correct. Confused with: vegetation 20.2%, sidewalk 14.1%
sidewalk:      46.7% correct. Confused with: vegetation 35.4%, building 11.3%
building:      88.0% correct. Confused with: vegetation 6.7%, sidewalk 4.5%
fence:          3.2% correct. Confused with: building 52.7%, vegetation 20.9%, sidewalk 20.8%
pole:           1.0% correct. Confused with: vegetation 63.5%, building 23.1%
traffic-sign:   0.2% correct. Confused with: vegetation 74.5%, building 9.8%
vegetation:    67.0% correct. Confused with: building 27.3%, sidewalk 5.1%
two-wheeler:    0.0% correct. Confused with: building 56.7%, road 23.1%
vehicle:        0.6% correct. Confused with: building 38.8%, road 32.2%, vegetation 20.0%
```

**Pattern**: Ground-level objects (fence, two-wheeler, vehicle) get massively confused with **building**. The height kernel suppresses false building at h≈0 (building mu=7m), recovering these classes. This is a HEIGHT-CORRELATED error.

### In-Domain Confusion (where height kernel barely helps):
```
road:          77.7% correct. Confused with: vegetation 17.0%, sidewalk 3.6%
sidewalk:      63.2% correct. Confused with: vegetation 23.5%, road 6.5%
building:      75.6% correct. Confused with: sidewalk 9.8%, vegetation 8.7%
fence:         36.9% correct. Confused with: sidewalk 30.9%, vegetation 21.6%
pole:          10.0% correct. Confused with: vegetation 65.7%, sidewalk 8.9%
traffic-sign:  15.3% correct. Confused with: vegetation 64.3%, road 9.6%
vegetation:    85.4% correct. Confused with: sidewalk 5.8%, fence 2.8%
two-wheeler:   13.9% correct. Confused with: road 22.8%, building 20.6%, vegetation 15.9%
vehicle:       45.8% correct. Confused with: road 33.3%, vegetation 16.5%
```

**Pattern**: The dominant confusion is everything → **vegetation** (pole 65.7%, traffic-sign 64.3%, other-ground 41.9%). Vegetation exists at all heights (ground shrubs to 20m canopy), so the height kernel CANNOT distinguish false vegetation from real vegetation. This is NOT a height-correlated error.

## What I've Already Tried for In-Domain

1. **Aggressive height kernel** (lambda=0.9, tight taus): -5.03% mIoU. Destroys vegetation (-20%), traffic-sign (-11%).
2. **Gentle height kernel** (lambda=0.15, wide taus): -0.32% mIoU. Nearly neutral.
3. **Selective classes** (only building+vegetation active): Still hurts — correct predictions at unusual heights get suppressed.
4. **Dead zone** (don't suppress near expected height): Helps avoid damage but limits gains.
5. **Redistribute mode** (reweight rather than suppress): Best result at +0.46% mIoU — but still tiny.
6. **Only ground classes active** (road tau=0.3, parking tau=0.5): +0.46% mIoU, +0.44% OA.

## Available Tools I Haven't Fully Exploited

1. **OSM confusion matrix prior** — already enabled at strength=2.0, using `osm_confusion_matrix_optimized_kitti360.yaml`. But this was likely optimized for cross-domain labels. Could re-optimize for in-domain.
   - Script: `optimize_osm_cm.py` — builds confusion matrix from GT co-occurrence with OSM geometry
   - OSM categories: building, road, grassland, tree, parking, fence → mapped to semantic classes

2. **OSM height filtering** — `osm_height_filtering: true` — separate from the DEM height kernel, uses height bins for OSM priors.

3. **Uncertainty filtering** — `use_uncertainty_filter: false` — could gate the height kernel on model uncertainty.

4. **The height kernel itself** — currently only suppresses (multiply by ≤1). Could be modified to also BOOST correct predictions or redistribute evidence.

## My Questions

1. **Is the height kernel fundamentally limited here?** The dominant in-domain error (everything→vegetation) is not height-correlated. Can a height-based approach ever give 10-20% improvement when the errors aren't height-dependent?

2. **What's the theoretical maximum improvement from a height kernel on this confusion pattern?** If I could perfectly suppress every height-incompatible prediction, how much would mIoU improve?

3. **What other approaches should I try?** Given the confusion matrices, what's the most promising path to 10-20% improvement? Re-optimizing the OSM confusion matrix? Uncertainty weighting? Something else entirely?

4. **Should I change the height kernel architecture?** Currently it's purely suppressive. Would a different formulation (e.g., boosting height-compatible classes, or using height as a feature in BKI kernel itself rather than a post-hoc modulation) be more effective?

5. **Is the comparison fair?** The cross-domain baseline is so low (13.91%) that any improvement looks large in absolute terms. The in-domain baseline is 27.81% — similar absolute improvement (+2.7%) would only be a fraction of what the cross-domain shows. Should I be comparing relative improvement instead?

## Context on the Broader System

This is for a research paper. The paper's thesis is that geospatial priors (OSM + DEM) improve 3D semantic mapping. The OSM prior already provides significant improvement over vanilla S-BKI. The height kernel is an additional contribution. If the height kernel can only add +0.5% on in-domain, that's still a valid result — but I want to understand if I'm leaving significant gains on the table or if this is close to the theoretical ceiling for a height-based approach on this confusion pattern.
