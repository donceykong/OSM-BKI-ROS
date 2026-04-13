#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <pcl/filters/voxel_grid.h>

#include "bkioctomap.h"
#include "bki.h"

using std::vector;

// #define DEBUG true;

#ifdef DEBUG

#include <iostream>

#define Debug_Msg(msg) {\
std::cout << "Debug: " << msg << std::endl; }
#endif

namespace osm_bki {

    SemanticBKIOctoMap::SemanticBKIOctoMap() : SemanticBKIOctoMap(0.1f, // resolution
                                        4, // block_depth
                                        3,  // num_class
                                        1.0, // sf2
                                        1.0, // ell
                                        1.0f, // prior
                                        1.0f, // var_thresh
                                        0.3f, // free_thresh
                                        0.7f // occupied_thresh
                                    ) { }

    SemanticBKIOctoMap::SemanticBKIOctoMap(float resolution,
                        unsigned short block_depth,
                        int num_class,
                        float sf2,
                        float ell,
                        float prior,
                        float var_thresh,
                        float free_thresh,
                        float occupied_thresh)
            : resolution(resolution), block_depth(block_depth),
              block_size((float) pow(2, block_depth - 1) * resolution),
              osm_decay_meters_(2.0f) {
        Block::resolution = resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
        Block::index_map = init_index_map(Block::key_loc_map, block_depth);
        
        // Bug fixed
        Block::cell_num = static_cast<unsigned short>(round(Block::size / Block::resolution));
        std::cout << "block::resolution: " << Block::resolution << std::endl;
        std::cout << "block::size: " << Block::size << std::endl;
        std::cout << "block::cell_num: " << Block::cell_num << std::endl;
        
        SemanticOcTree::max_depth = block_depth;

        SemanticOcTreeNode::num_class = num_class;
        SemanticOcTreeNode::sf2 = sf2;
        SemanticOcTreeNode::ell = ell;
        SemanticOcTreeNode::prior = prior;
        SemanticOcTreeNode::var_thresh = var_thresh;
        SemanticOcTreeNode::free_thresh = free_thresh;
        SemanticOcTreeNode::occupied_thresh = occupied_thresh;
    }

    SemanticBKIOctoMap::~SemanticBKIOctoMap() {
        for (auto it = block_arr.begin(); it != block_arr.end(); ++it) {
            if (it->second != nullptr) {
                delete it->second;
            }
        }
    }

    void SemanticBKIOctoMap::set_resolution(float resolution) {
        this->resolution = resolution;
        Block::resolution = resolution;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void SemanticBKIOctoMap::set_block_depth(unsigned short max_depth) {
        this->block_depth = max_depth;
        SemanticOcTree::max_depth = max_depth;
        this->block_size = (float) pow(2, block_depth - 1) * resolution;
        Block::size = this->block_size;
        Block::key_loc_map = init_key_loc_map(resolution, block_depth);
    }

    void SemanticBKIOctoMap::insert_pointcloud_csm(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range) {
        insert_pointcloud_csm(cloud, origin, ds_resolution, free_res, max_range,
                              std::vector<float>{}, nullptr);
    }

    void SemanticBKIOctoMap::insert_pointcloud_csm(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range,
                                      const std::vector<float> &point_weights) {
        insert_pointcloud_csm(cloud, origin, ds_resolution, free_res, max_range, point_weights, nullptr);
    }

    void SemanticBKIOctoMap::insert_pointcloud_csm(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range,
                                      const std::vector<float> &point_weights,
                                      const std::vector<std::vector<float>> *multiclass_probs) {

#ifdef DEBUG
        Debug_Msg("Insert pointcloud (CSM): " << "cloud size: " << cloud.size() << " origin: " << origin);
#endif

        bool use_soft = (multiclass_probs != nullptr && !multiclass_probs->empty() &&
                         multiclass_probs->size() == cloud.size());
        bool has_weights = (!point_weights.empty() && point_weights.size() == cloud.size());

        ////////// Preparation //////////////////////////
        GPPointCloud xy;
        if (use_soft)
            get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy, point_weights, multiclass_probs);
        else if (has_weights)
            get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy, point_weights);
        else
            get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy);

        if (xy.size() == 0)
            return;

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);
        osm_height_min_z_ = lim_min.z();
        osm_height_max_z_ = lim_max.z();

        // Pre-filter OSM geometry to scan vicinity (used by init_osm_prior_for_block).
        float cx = (lim_min.x() + lim_max.x()) * 0.5f;
        float cy = (lim_min.y() + lim_max.y()) * 0.5f;
        float half_dx = (lim_max.x() - lim_min.x()) * 0.5f;
        float half_dy = (lim_max.y() - lim_min.y()) * 0.5f;
        float max_xy = std::sqrt(half_dx * half_dx + half_dy * half_dy);
        filter_osm_for_scan(cx, cy, max_xy);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float p[] = {it->first.x(), it->first.y(), it->first.z()};
            rtree.Insert(p, p, const_cast<GPPointType *>(&*it));
        }

        ////////// Training /////////////////////////////
        // Reference-faithful CSM: no per-class OSM kernel boost, no kernel weights.
        // Only Dirichlet prior (init_osm_prior_for_block) carries OSM signal.
        int nc = SemanticOcTreeNode::num_class;
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, SemanticBKI3f *> bgk_arr;
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            { test_blocks.push_back(key); };

            GPPointCloud block_xy;
            get_gp_points_in_bbox(key, block_xy);
            if (block_xy.size() < 1)
                continue;

            vector<float> block_x, block_y;
            std::vector<std::vector<float>> block_y_soft;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());
                block_y.push_back(it->second);
                if (use_soft) {
                    if (it->soft_probs && it->soft_probs->size() == static_cast<size_t>(nc)) {
                        block_y_soft.push_back(*it->soft_probs);
                    } else {
                        // Free-space points: zero soft probs
                        block_y_soft.emplace_back(static_cast<size_t>(nc), 0.f);
                    }
                }
            }

            SemanticBKI3f *bgk = new SemanticBKI3f(nc, SemanticOcTreeNode::sf2, SemanticOcTreeNode::ell);
            if (use_soft && block_y_soft.size() == block_xy.size())
                bgk->train_soft(block_x, block_y_soft);
            else
                bgk->train(block_x, block_y);
#ifdef OPENMP
#pragma omp critical
#endif
            { bgk_arr.emplace(key, bgk); };
        }

        ////////// Prediction ///////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
            Block *block = nullptr;
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bool is_new_block = (block_arr.find(key) == block_arr.end());
                if (is_new_block)
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
                block = block_arr[key];
                if (is_new_block)
                    init_osm_prior_for_block(block);
            };
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }

            // Reference CSM: predict only from this block's own GP (no extended block loop).
            auto bgk = bgk_arr.find(key);
            if (bgk == bgk_arr.end())
              continue;

            vector<vector<float>> ybars;
            bgk->second->predict_csm(xs, ybars);

            int j = 0;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                SemanticOcTreeNode &node = leaf_it.get_node();
                node.update(ybars[j]);
            }
        }

        ////////// Cleaning /////////////////////////////
        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;

        clear_osm_scan_filter();
        rtree.RemoveAll();
    }

    void SemanticBKIOctoMap::set_osm_buildings(const std::vector<Geometry2D> &buildings) {
        osm_buildings_ = buildings;
    }

    void SemanticBKIOctoMap::set_osm_roads(const std::vector<Geometry2D> &roads) {
        osm_roads_ = roads;
    }

    void SemanticBKIOctoMap::set_osm_sidewalks(const std::vector<Geometry2D> &sidewalks) {
        osm_sidewalks_ = sidewalks;
    }

    void SemanticBKIOctoMap::set_osm_cycleways(const std::vector<Geometry2D> &cycleways) {
        osm_cycleways_ = cycleways;
    }

    void SemanticBKIOctoMap::set_osm_grasslands(const std::vector<Geometry2D> &grasslands) {
        osm_grasslands_ = grasslands;
    }

    void SemanticBKIOctoMap::set_osm_trees(const std::vector<Geometry2D> &trees) {
        osm_trees_ = trees;
    }

    void SemanticBKIOctoMap::set_osm_forests(const std::vector<Geometry2D> &forests) {
        osm_forests_ = forests;
    }

    void SemanticBKIOctoMap::set_osm_tree_points(const std::vector<std::pair<float, float>> &tree_points) {
        osm_tree_points_ = tree_points;
    }

    void SemanticBKIOctoMap::set_osm_tree_point_radius(float radius_m) {
        osm_tree_point_radius_ = std::max(0.1f, radius_m);
    }

    void SemanticBKIOctoMap::set_osm_parking(const std::vector<Geometry2D> &parking) {
        osm_parking_ = parking;
    }

    void SemanticBKIOctoMap::set_osm_fences(const std::vector<Geometry2D> &fences) {
        osm_fences_ = fences;
    }

    void SemanticBKIOctoMap::set_osm_road_width(float width_m) {
        osm_road_width_ = std::max(0.1f, width_m);
    }

    void SemanticBKIOctoMap::set_osm_sidewalk_width(float width_m) {
        osm_sidewalk_width_ = std::max(0.1f, width_m);
    }

    void SemanticBKIOctoMap::set_osm_cycleway_width(float width_m) {
        osm_cycleway_width_ = std::max(0.1f, width_m);
    }

    void SemanticBKIOctoMap::set_osm_fence_width(float width_m) {
        osm_fence_width_ = std::max(0.1f, width_m);
    }

    void SemanticBKIOctoMap::set_osm_decay_meters(float decay_m) {
        osm_decay_meters_ = decay_m;
    }

    void SemanticBKIOctoMap::set_osm_prior_strength(float strength) {
        osm_prior_strength_ = strength;
    }

    void SemanticBKIOctoMap::set_osm_dirichlet_prior_strength(float strength) {
        osm_dirichlet_prior_strength_ = strength;
    }

    void SemanticBKIOctoMap::init_osm_prior_for_block(Block *block) {
        if (!osm_cm_loaded_ || osm_dirichlet_prior_strength_ <= 0.0f || block == nullptr) return;

        int nc = SemanticOcTreeNode::num_class;
        float base_prior = SemanticOcTreeNode::prior;

        for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
            SemanticOcTreeNode &node = leaf_it.get_node();
            if (node.classified) continue;  // already has sensor data, don't overwrite

            point3f loc = block->get_loc(leaf_it);

            // Query OSM at this voxel's location
            float osm_vec[N_OSM_PRIOR_COLS];
            compute_osm_prior_vec(loc.x(), loc.y(), osm_vec);

            // OSM confidence
            float c_x = 0.f;
            for (int c = 0; c < N_OSM_PRIOR_COLS - 1; ++c) {
                if (osm_vec[c] > c_x) c_x = osm_vec[c];
            }
            if (c_x <= 0.f) continue;  // no OSM coverage, keep uniform prior

            // Compute Mm[r] = M * osm_vec
            std::vector<float> Mm(osm_cm_rows_, 0.f);
            for (int r = 0; r < osm_cm_rows_; ++r) {
                for (int c = 0; c < N_OSM_PRIOR_COLS; ++c)
                    Mm[r] += osm_cm_[r][c] * osm_vec[c];
            }

            // Variant A: apply per-class height filter to common-class rows before
            // computing max(Mm). Uses the same per-scan relative z bounds as the kernel path.
            if (use_osm_height_filter_ && osm_height_cm_loaded_ && osm_height_num_bins_ > 0) {
                float z_range = osm_height_max_z_ - osm_height_min_z_ + 1e-6f;
                int bin = static_cast<int>((loc.z() - osm_height_min_z_) / z_range * static_cast<float>(osm_height_num_bins_));
                bin = std::max(0, std::min(bin, osm_height_num_bins_ - 1));
                const auto &hrow = osm_height_cm_[static_cast<size_t>(bin)];
                int ncols = std::min(osm_cm_rows_, static_cast<int>(hrow.size()));
                for (int r = 0; r < ncols; ++r)
                    Mm[r] *= hrow[r];
            }

            float max_Mm = 0.f;
            for (int r = 0; r < osm_cm_rows_; ++r)
                if (Mm[r] > max_Mm) max_Mm = Mm[r];
            if (max_Mm <= 0.f) continue;

            // Build spatially varying prior: base_prior + boost for OSM-supported classes
            std::vector<float> osm_prior(nc, base_prior);
            for (int r = 0; r < osm_cm_rows_; ++r) {
                float support = Mm[r] / max_Mm;
                if (support <= 0.f) continue;
                float boost = osm_dirichlet_prior_strength_ * c_x * support;
                const auto &labels = osm_cm_row_to_labels_[r];
                for (int lbl : labels) {
                    if (lbl >= 0 && lbl < nc)
                        osm_prior[lbl] = base_prior + boost;
                }
            }

            node.set_osm_prior(osm_prior);
        }
    }

    void SemanticBKIOctoMap::set_osm_height_filter_enabled(bool enabled) {
        use_osm_height_filter_ = enabled;
    }

    void SemanticBKIOctoMap::set_osm_height_confusion_matrix(const std::vector<std::vector<float>> &matrix) {
        // Variant A: rows = height bins, cols = common-class rows (same index space as osm_cm_ rows).
        osm_height_num_bins_ = static_cast<int>(matrix.size());
        osm_height_cm_.clear();
        osm_height_cm_.resize(static_cast<size_t>(osm_height_num_bins_));
        for (int r = 0; r < osm_height_num_bins_; ++r) {
            osm_height_cm_[static_cast<size_t>(r)] = matrix[r];
        }
        osm_height_cm_loaded_ = (osm_height_num_bins_ > 0);
    }

    void SemanticBKIOctoMap::get_osm_priors_for_visualization(float x, float y, float &building, float &road,
                                                              float &grassland, float &tree, float &parking,
                                                              float &fence) const {
        building = compute_osm_building_prior(x, y);
        road = compute_osm_road_prior(x, y);
        grassland = compute_osm_grassland_prior(x, y);
        tree = compute_osm_tree_prior(x, y);
        parking = compute_osm_parking_prior(x, y);
        fence = compute_osm_fence_prior(x, y);
    }

    void SemanticBKIOctoMap::set_osm_confusion_matrix(
            const std::vector<std::vector<float>> &matrix,
            const std::vector<std::vector<int>> &row_to_labels) {
        osm_cm_rows_ = std::min(static_cast<int>(matrix.size()), 13);
        std::memset(osm_cm_, 0, sizeof(osm_cm_));
        for (int r = 0; r < osm_cm_rows_; ++r) {
            int ncols = std::min(static_cast<int>(matrix[r].size()), N_OSM_PRIOR_COLS);
            for (int c = 0; c < ncols; ++c)
                osm_cm_[r][c] = matrix[r][c];
        }
        osm_cm_row_to_labels_ = row_to_labels;
        osm_cm_row_to_labels_.resize(osm_cm_rows_);
        osm_cm_loaded_ = true;
    }

    void SemanticBKIOctoMap::compute_osm_prior_vec(float x, float y,
                                                    float osm_vec[N_OSM_PRIOR_COLS]) const {
        osm_vec[0] = compute_osm_road_prior(x, y);
        osm_vec[1] = compute_osm_sidewalk_prior(x, y);
        osm_vec[2] = compute_osm_cycleway_prior(x, y);
        osm_vec[3] = compute_osm_parking_prior(x, y);
        osm_vec[4] = compute_osm_grassland_prior(x, y);
        osm_vec[5] = compute_osm_tree_prior(x, y);
        osm_vec[6] = compute_osm_forest_prior(x, y);
        osm_vec[7] = compute_osm_building_prior(x, y);
        osm_vec[8] = compute_osm_fence_prior(x, y);
        // "none" = 1 when no OSM geometry covers this point, 0 when fully covered
        float max_geom = 0.f;
        for (int c = 0; c < 9; ++c)
            if (osm_vec[c] > max_geom) max_geom = osm_vec[c];
        osm_vec[9] = 1.0f - max_geom;
    }

    void SemanticBKIOctoMap::compute_osm_semantic_kernel(
            float x, float y, float z, std::vector<float> &k_vec) const {
        int nc = static_cast<int>(k_vec.size());
        // Default: all classes get weight 1.0 (no modulation)
        std::fill(k_vec.begin(), k_vec.end(), 1.0f);

        if (!osm_cm_loaded_ || osm_prior_strength_ <= 0.0f) return;

        // 1. Compute OSM prior vector m(x,y) at the training point location
        float osm_vec[N_OSM_PRIOR_COLS];
        compute_osm_prior_vec(x, y, osm_vec);

        // 2. OSM confidence c(x) = max of OSM prior (excluding "none" column)
        float c_x = 0.f;
        for (int c = 0; c < N_OSM_PRIOR_COLS - 1; ++c) {
            if (osm_vec[c] > c_x) c_x = osm_vec[c];
        }
        if (c_x <= 0.f) return;  // No OSM coverage: all weights stay 1.0

        // 3. Variant A: project OSM -> common, then apply per-class height filter.
        std::vector<float> Mm(osm_cm_rows_, 0.f);
        for (int r = 0; r < osm_cm_rows_; ++r)
            for (int c = 0; c < N_OSM_PRIOR_COLS; ++c)
                Mm[r] += osm_cm_[r][c] * osm_vec[c];

        if (use_osm_height_filter_ && osm_height_cm_loaded_ && osm_height_num_bins_ > 0) {
            float z_range = osm_height_max_z_ - osm_height_min_z_ + 1e-6f;
            int bin = static_cast<int>((z - osm_height_min_z_) / z_range * static_cast<float>(osm_height_num_bins_));
            bin = std::max(0, std::min(bin, osm_height_num_bins_ - 1));
            const auto &hrow = osm_height_cm_[static_cast<size_t>(bin)];
            int ncols = std::min(osm_cm_rows_, static_cast<int>(hrow.size()));
            for (int r = 0; r < ncols; ++r)
                Mm[r] *= hrow[r];
        }

        float max_Mm = 0.f;
        for (int r = 0; r < osm_cm_rows_; ++r)
            if (Mm[r] > max_Mm) max_Mm = Mm[r];
        if (max_Mm <= 0.f) return;

        // 5. Per-class boost: classes supported by OSM get weight > 1.0
        //    Classes not supported stay at 1.0 (no suppression)
        //    k_vec[lbl] = 1.0 + strength * c(x) * Mm[r] / max(Mm)
        for (int r = 0; r < osm_cm_rows_; ++r) {
            float support = Mm[r] / max_Mm;  // normalized to [0, 1]
            if (support <= 0.f) continue;     // no boost for unsupported classes
            float boost = osm_prior_strength_ * c_x * support;
            const auto &labels = osm_cm_row_to_labels_[r];
            for (int lbl : labels) {
                if (lbl >= 0 && lbl < nc)
                    k_vec[lbl] = 1.0f + boost;
            }
        }
    }

    void SemanticBKIOctoMap::filter_osm_for_scan(float center_x, float center_y, float max_xy_dist) {
        float radius = max_xy_dist * osm_scan_radius_extension_ + osm_decay_meters_;

        auto filter_geom = [&](const std::vector<Geometry2D> &src, std::vector<Geometry2D> &dst) {
            dst.clear();
            for (const auto &g : src) {
                if (geometry_overlaps_circle(g, center_x, center_y, radius))
                    dst.push_back(g);
            }
        };
        auto filter_pts = [&](const std::vector<std::pair<float,float>> &src,
                              std::vector<std::pair<float,float>> &dst) {
            dst.clear();
            for (const auto &p : src) {
                if (point_overlaps_circle(p.first, p.second, center_x, center_y, radius))
                    dst.push_back(p);
            }
        };

        filter_geom(osm_buildings_,  active_buildings_);
        filter_geom(osm_roads_,      active_roads_);
        filter_geom(osm_sidewalks_,  active_sidewalks_);
        filter_geom(osm_cycleways_,  active_cycleways_);
        filter_geom(osm_grasslands_, active_grasslands_);
        filter_geom(osm_trees_,      active_trees_);
        filter_geom(osm_forests_,    active_forests_);
        filter_geom(osm_parking_,    active_parking_);
        filter_geom(osm_fences_,     active_fences_);
        filter_pts(osm_tree_points_, active_tree_points_);

        osm_scan_filtered_ = true;
    }

    void SemanticBKIOctoMap::clear_osm_scan_filter() {
        osm_scan_filtered_ = false;
    }

    float SemanticBKIOctoMap::compute_osm_building_prior(float x, float y) const {
        const auto &buildings = osm_scan_filtered_ ? active_buildings_ : osm_buildings_;
        if (buildings.empty()) return 0.f;
        float min_positive_d = std::numeric_limits<float>::max();
        for (const auto &poly : buildings) {
            float signed_d = distance_to_polygon_boundary(x, y, poly);
            if (signed_d <= 0.f) return 1.f;  // inside a building
            if (signed_d < min_positive_d) min_positive_d = signed_d;
        }
        return osm_prior_from_signed_distance(min_positive_d, osm_decay_meters_);
    }

    float SemanticBKIOctoMap::compute_osm_road_prior(float x, float y) const {
        const auto &roads = osm_scan_filtered_ ? active_roads_ : osm_roads_;
        if (roads.empty()) return 0.f;
        float min_signed_d = std::numeric_limits<float>::max();
        for (const auto &road : roads) {
            float signed_d = distance_to_polyline_band_signed(x, y, road, osm_road_width_);
            if (signed_d <= 0.f) return 1.f;
            if (signed_d < min_signed_d) min_signed_d = signed_d;
        }
        return osm_prior_from_signed_distance(min_signed_d, osm_decay_meters_);
    }

    float SemanticBKIOctoMap::compute_osm_grassland_prior(float x, float y) const {
        const auto &grasslands = osm_scan_filtered_ ? active_grasslands_ : osm_grasslands_;
        if (grasslands.empty()) return 0.f;
        float min_positive_d = std::numeric_limits<float>::max();
        for (const auto &poly : grasslands) {
            float signed_d = distance_to_polygon_boundary(x, y, poly);
            if (signed_d <= 0.f) return 1.f;  // inside grassland
            if (signed_d < min_positive_d) min_positive_d = signed_d;
        }
        return osm_prior_from_signed_distance(min_positive_d, osm_decay_meters_);
    }

    float SemanticBKIOctoMap::compute_osm_forest_prior(float x, float y) const {
        const auto &forests = osm_scan_filtered_ ? active_forests_ : osm_forests_;
        if (forests.empty()) return 0.f;
        float min_positive_d = std::numeric_limits<float>::max();
        for (const auto &poly : forests) {
            float signed_d = distance_to_polygon_boundary(x, y, poly);
            if (signed_d <= 0.f) return 1.f;  // inside forest
            if (signed_d < min_positive_d) min_positive_d = signed_d;
        }
        return osm_prior_from_signed_distance(min_positive_d, osm_decay_meters_);
    }

    float SemanticBKIOctoMap::compute_osm_tree_prior(float x, float y) const {
        const auto &trees = osm_scan_filtered_ ? active_trees_ : osm_trees_;
        const auto &tree_pts = osm_scan_filtered_ ? active_tree_points_ : osm_tree_points_;
        float max_prior = 0.f;
        // Check tree polygons (forests/woods)
        if (!trees.empty()) {
            float min_positive_d = std::numeric_limits<float>::max();
            for (const auto &poly : trees) {
                float signed_d = distance_to_polygon_boundary(x, y, poly);
                if (signed_d <= 0.f) {
                    max_prior = 1.f;  // inside forest, max prior
                    break;
                }
                if (signed_d < min_positive_d) min_positive_d = signed_d;
            }
            if (max_prior < 1.f) {
                float poly_prior = osm_prior_from_signed_distance(min_positive_d, osm_decay_meters_);
                if (poly_prior > max_prior) max_prior = poly_prior;
            }
        }
        // Check tree points (single trees): treat as circles with radius; prior = 1 inside, decay outside (same as polygons)
        if (!tree_pts.empty()) {
            float min_signed_d = std::numeric_limits<float>::max();
            for (const auto& pt : tree_pts) {
                float signed_d = distance_to_circle_signed(x, y, pt.first, pt.second, osm_tree_point_radius_);
                if (signed_d <= 0.f) {
                    max_prior = 1.f;  // inside a tree circle
                    break;
                }
                if (signed_d < min_signed_d) min_signed_d = signed_d;
            }
            if (max_prior < 1.f && min_signed_d < std::numeric_limits<float>::max()) {
                float circle_prior = osm_prior_from_signed_distance(min_signed_d, osm_decay_meters_);
                if (circle_prior > max_prior) max_prior = circle_prior;
            }
        }
        return max_prior;
    }

    float SemanticBKIOctoMap::compute_osm_parking_prior(float x, float y) const {
        const auto &parking = osm_scan_filtered_ ? active_parking_ : osm_parking_;
        if (parking.empty()) return 0.f;
        float max_prior = 0.f;
        float min_positive_d = std::numeric_limits<float>::max();
        for (const auto &poly : parking) {
            if (poly.coords.size() < 3) {
                float d = distance_to_polyline(x, y, poly);
                float prior = osm_prior_from_distance(d, osm_decay_meters_);
                if (prior > max_prior) max_prior = prior;
                continue;
            }
            float signed_d = distance_to_polygon_boundary(x, y, poly);
            if (signed_d <= 0.f) return 1.f;  // inside parking
            if (signed_d < min_positive_d) min_positive_d = signed_d;
        }
        float poly_prior = osm_prior_from_signed_distance(min_positive_d, osm_decay_meters_);
        return std::max(max_prior, poly_prior);
    }

    float SemanticBKIOctoMap::compute_osm_fence_prior(float x, float y) const {
        const auto &fences = osm_scan_filtered_ ? active_fences_ : osm_fences_;
        if (fences.empty()) return 0.f;
        float min_signed_d = std::numeric_limits<float>::max();
        for (const auto &fence : fences) {
            float signed_d = distance_to_polyline_band_signed(x, y, fence, osm_fence_width_);
            if (signed_d <= 0.f) return 1.f;
            if (signed_d < min_signed_d) min_signed_d = signed_d;
        }
        return osm_prior_from_signed_distance(min_signed_d, osm_decay_meters_);
    }

    float SemanticBKIOctoMap::compute_osm_sidewalk_prior(float x, float y) const {
        const auto &sidewalks = osm_scan_filtered_ ? active_sidewalks_ : osm_sidewalks_;
        if (sidewalks.empty()) return 0.f;
        float min_signed_d = std::numeric_limits<float>::max();
        for (const auto &sw : sidewalks) {
            float signed_d = distance_to_polyline_band_signed(x, y, sw, osm_sidewalk_width_);
            if (signed_d <= 0.f) return 1.f;
            if (signed_d < min_signed_d) min_signed_d = signed_d;
        }
        return osm_prior_from_signed_distance(min_signed_d, osm_decay_meters_);
    }

    float SemanticBKIOctoMap::compute_osm_cycleway_prior(float x, float y) const {
        const auto &cycleways = osm_scan_filtered_ ? active_cycleways_ : osm_cycleways_;
        if (cycleways.empty()) return 0.f;
        float min_signed_d = std::numeric_limits<float>::max();
        for (const auto &cw : cycleways) {
            float signed_d = distance_to_polyline_band_signed(x, y, cw, osm_cycleway_width_);
            if (signed_d <= 0.f) return 1.f;
            if (signed_d < min_signed_d) min_signed_d = signed_d;
        }
        return osm_prior_from_signed_distance(min_signed_d, osm_decay_meters_);
    }

    void SemanticBKIOctoMap::insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_res, float max_range) {

#ifdef DEBUG
        Debug_Msg("Insert pointcloud: " << "cloud size: " << cloud.size() << " origin: " << origin);
#endif

        ////////// Preparation //////////////////////////
        /////////////////////////////////////////////////
        GPPointCloud xy;
        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy);
#ifdef DEBUG
        Debug_Msg("Training data size: " << xy.size());
#endif
        // If pointcloud after max_range filtering is empty
        //  no need to do anything
        if (xy.size() == 0) {
            return;
        }

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);
        osm_height_min_z_ = lim_min.z();
        osm_height_max_z_ = lim_max.z();

        // Pre-filter OSM geometry to scan vicinity
        float cx = (lim_min.x() + lim_max.x()) * 0.5f;
        float cy = (lim_min.y() + lim_max.y()) * 0.5f;
        float half_dx = (lim_max.x() - lim_min.x()) * 0.5f;
        float half_dy = (lim_max.y() - lim_min.y()) * 0.5f;
        float max_xy = std::sqrt(half_dx * half_dx + half_dy * half_dy);
        filter_osm_for_scan(cx, cy, max_xy);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float p[] = {it->first.x(), it->first.y(), it->first.z()};
            rtree.Insert(p, p, const_cast<GPPointType *>(&*it));
        }
        /////////////////////////////////////////////////

        ////////// Training /////////////////////////////
        /////////////////////////////////////////////////
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, SemanticBKI3f *> bgk_arr;
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            {
                test_blocks.push_back(key);
            };

            GPPointCloud block_xy;
            get_gp_points_in_bbox(key, block_xy);
            if (block_xy.size() < 1)
                continue;

            int nc_basic = SemanticOcTreeNode::num_class;
            vector<float> block_x, block_y, block_w;
            std::vector<std::vector<float>> block_w_class;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());
                block_y.push_back(it->second);
                block_w.push_back(1.0f);

                // Per-class OSM semantic kernel (boost-only, no suppression)
                std::vector<float> k_vec(nc_basic, 1.0f);
                int lbl = static_cast<int>(it->second);
                if (lbl > 0 && lbl < nc_basic)
                    compute_osm_semantic_kernel(it->first.x(), it->first.y(), it->first.z(), k_vec);
                block_w_class.push_back(std::move(k_vec));
            }

            SemanticBKI3f *bgk = new SemanticBKI3f(SemanticOcTreeNode::num_class, SemanticOcTreeNode::sf2, SemanticOcTreeNode::ell);
            bgk->train(block_x, block_y, block_w, block_w_class);
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bgk_arr.emplace(key, bgk);
            };
        }
#ifdef DEBUG
        Debug_Msg("Training done");
        Debug_Msg("Prediction: block number: " << test_blocks.size());
#endif
        /////////////////////////////////////////////////

        ////////// Prediction ///////////////////////////
        /////////////////////////////////////////////////
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
            Block *block = nullptr;
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bool is_new_block = (block_arr.find(key) == block_arr.end());
                if (is_new_block)
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
                block = block_arr[key];
                if (is_new_block)
                    init_osm_prior_for_block(block);
            };
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }

            ExtendedBlock eblock = block->get_extended_block();
            for (auto block_it = eblock.cbegin(); block_it != eblock.cend(); ++block_it) {
                auto bgk = bgk_arr.find(*block_it);
                if (bgk == bgk_arr.end())
                    continue;

                vector<vector<float>> ybars;
                bgk->second->predict(xs, ybars);

                int j = 0;
                for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                    SemanticOcTreeNode &node = leaf_it.get_node();
                    // OSM prior is now in the kernel weights — no post-prediction bias needed
                    node.update(ybars[j]);
                }
            }
        }
#ifdef DEBUG
        Debug_Msg("Prediction done");
#endif

        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////
        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;

        clear_osm_scan_filter();
        rtree.RemoveAll();
    }

    void SemanticBKIOctoMap::get_bbox(point3f &lim_min, point3f &lim_max) const {
        lim_min = point3f(0, 0, 0);
        lim_max = point3f(0, 0, 0);

        GPPointCloud centers;
        for (auto it = block_arr.cbegin(); it != block_arr.cend(); ++it) {
            centers.emplace_back(it->second->get_center(), 1);
        }
        if (centers.size() > 0) {
            bbox(centers, lim_min, lim_max);
            lim_min -= point3f(block_size, block_size, block_size) * 0.5;
            lim_max += point3f(block_size, block_size, block_size) * 0.5;
        }
    }

    void SemanticBKIOctoMap::get_training_data(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                                      float free_resolution, float max_range, GPPointCloud &xy) const {
        PCLPointCloud sampled_hits;
        downsample(cloud, sampled_hits, ds_resolution);

        PCLPointCloud frees;
        frees.height = 1;
        frees.width = 0;
        xy.clear();
        for (auto it = sampled_hits.begin(); it != sampled_hits.end(); ++it) {
            point3f p(it->x, it->y, it->z);
            if (max_range > 0) {
                double l = (p - origin).norm();
                if (l > max_range)
                    continue;
            }
            
            xy.emplace_back(p, it->label);

            PointCloud frees_n;
            beam_sample(p, origin, frees_n, free_resolution);

            PCLPointType p_origin = PCLPointType();
            p_origin.x = origin.x();
            p_origin.y = origin.y();
            p_origin.z = origin.z();
            p_origin.label = 0;
            frees.push_back(p_origin);
            
            for (auto p = frees_n.begin(); p != frees_n.end(); ++p) {
                PCLPointType p_free = PCLPointType();
                p_free.x = p->x();
                p_free.y = p->y();
                p_free.z = p->z();
                p_free.label = 0;
                frees.push_back(p_free);
                frees.width++;
            }
        }

        PCLPointCloud sampled_frees;    
        downsample(frees, sampled_frees, ds_resolution);

        for (auto it = sampled_frees.begin(); it != sampled_frees.end(); ++it) {
            xy.emplace_back(point3f(it->x, it->y, it->z), 0.0f);
        }
    }

    void SemanticBKIOctoMap::get_training_data(const PCLPointCloud &cloud, const point3f &origin,
                                      float ds_resolution, float free_resolution, float max_range,
                                      GPPointCloud &xy, const std::vector<float> &point_weights) const {
        // Manual voxel downsampling that preserves per-point weights.
        // For each voxel: centroid position, label from highest-weight point, mean weight.
        xy.clear();

        struct VoxelData {
            double sx, sy, sz;
            uint32_t label;
            float best_w;
            double w_sum;
            int count;
        };

        float inv_res = (ds_resolution > 0) ? (1.0f / ds_resolution) : 0.0f;
        auto voxel_key = [inv_res](float x, float y, float z) -> int64_t {
            int64_t ix = static_cast<int64_t>(std::floor(x * inv_res));
            int64_t iy = static_cast<int64_t>(std::floor(y * inv_res));
            int64_t iz = static_cast<int64_t>(std::floor(z * inv_res));
            return (ix * 73856093) ^ (iy * 19349669) ^ (iz * 83492791);
        };

        std::unordered_map<int64_t, VoxelData> voxels;

        PCLPointCloud frees;
        frees.height = 1;
        frees.width = 0;

        for (size_t i = 0; i < cloud.size(); ++i) {
            point3f p(cloud[i].x, cloud[i].y, cloud[i].z);
            if (max_range > 0 && (p - origin).norm() > max_range)
                continue;

            float w = (i < point_weights.size()) ? point_weights[i] : 1.0f;

            if (ds_resolution > 0) {
                int64_t key = voxel_key(cloud[i].x, cloud[i].y, cloud[i].z);
                auto it = voxels.find(key);
                if (it == voxels.end()) {
                    voxels[key] = {cloud[i].x, cloud[i].y, cloud[i].z,
                                   cloud[i].label, w, w, 1};
                } else {
                    auto &v = it->second;
                    v.sx += cloud[i].x; v.sy += cloud[i].y; v.sz += cloud[i].z;
                    if (w > v.best_w) { v.label = cloud[i].label; v.best_w = w; }
                    v.w_sum += w;
                    v.count++;
                }
            } else {
                xy.emplace_back(p, cloud[i].label, w);
            }

            PointCloud frees_n;
            beam_sample(p, origin, frees_n, free_resolution);
            PCLPointType p_origin;
            p_origin.x = origin.x(); p_origin.y = origin.y(); p_origin.z = origin.z();
            p_origin.label = 0;
            frees.push_back(p_origin);
            for (auto fp = frees_n.begin(); fp != frees_n.end(); ++fp) {
                PCLPointType pf;
                pf.x = fp->x(); pf.y = fp->y(); pf.z = fp->z(); pf.label = 0;
                frees.push_back(pf);
                frees.width++;
            }
        }

        if (ds_resolution > 0) {
            for (auto &kv : voxels) {
                auto &v = kv.second;
                float cx = static_cast<float>(v.sx / v.count);
                float cy = static_cast<float>(v.sy / v.count);
                float cz = static_cast<float>(v.sz / v.count);
                float avg_w = static_cast<float>(v.w_sum / v.count);
                xy.emplace_back(point3f(cx, cy, cz), v.label, avg_w);
            }
        }

        // Free space points always get weight 1.0
        PCLPointCloud sampled_frees;
        downsample(frees, sampled_frees, ds_resolution);
        for (auto it = sampled_frees.begin(); it != sampled_frees.end(); ++it) {
            xy.emplace_back(point3f(it->x, it->y, it->z), 0.0f, 1.0f);
        }
    }

    void SemanticBKIOctoMap::get_training_data(const PCLPointCloud &cloud, const point3f &origin,
                                      float ds_resolution, float free_resolution, float max_range,
                                      GPPointCloud &xy, const std::vector<float> &point_weights,
                                      const std::vector<std::vector<float>> *multiclass_probs) const {
        xy.clear();
        bool use_soft = (multiclass_probs != nullptr && !multiclass_probs->empty() &&
                         multiclass_probs->size() == cloud.size());
        int nc = use_soft ? static_cast<int>((*multiclass_probs)[0].size()) : 0;

        struct VoxelDataSoft {
            double sx, sy, sz;
            uint32_t label;
            float best_w;
            double w_sum;
            int count;
            std::vector<double> prob_sum;
        };

        float inv_res = (ds_resolution > 0) ? (1.0f / ds_resolution) : 0.0f;
        auto voxel_key = [inv_res](float x, float y, float z) -> int64_t {
            int64_t ix = static_cast<int64_t>(std::floor(x * inv_res));
            int64_t iy = static_cast<int64_t>(std::floor(y * inv_res));
            int64_t iz = static_cast<int64_t>(std::floor(z * inv_res));
            return (ix * 73856093) ^ (iy * 19349669) ^ (iz * 83492791);
        };

        std::unordered_map<int64_t, VoxelDataSoft> voxels;
        PCLPointCloud frees;
        frees.height = 1;
        frees.width = 0;

        for (size_t i = 0; i < cloud.size(); ++i) {
            point3f p(cloud[i].x, cloud[i].y, cloud[i].z);
            if (max_range > 0 && (p - origin).norm() > max_range)
                continue;

            float w = (i < point_weights.size()) ? point_weights[i] : 1.0f;

            if (ds_resolution > 0) {
                int64_t key = voxel_key(cloud[i].x, cloud[i].y, cloud[i].z);
                auto it = voxels.find(key);
                if (it == voxels.end()) {
                    VoxelDataSoft v;
                    v.sx = cloud[i].x; v.sy = cloud[i].y; v.sz = cloud[i].z;
                    v.label = cloud[i].label; v.best_w = w; v.w_sum = w; v.count = 1;
                    if (use_soft && nc > 0) {
                        v.prob_sum.resize(static_cast<size_t>(nc), 0.0);
                        for (int c = 0; c < nc; ++c)
                            v.prob_sum[static_cast<size_t>(c)] = (*multiclass_probs)[i][static_cast<size_t>(c)];
                    }
                    voxels[key] = std::move(v);
                } else {
                    auto &v = it->second;
                    v.sx += cloud[i].x; v.sy += cloud[i].y; v.sz += cloud[i].z;
                    if (w > v.best_w) { v.label = cloud[i].label; v.best_w = w; }
                    v.w_sum += w;
                    v.count++;
                    if (use_soft && nc > 0 && v.prob_sum.size() == static_cast<size_t>(nc)) {
                        for (int c = 0; c < nc; ++c)
                            v.prob_sum[static_cast<size_t>(c)] += (*multiclass_probs)[i][static_cast<size_t>(c)];
                    }
                }
            } else {
                std::shared_ptr<std::vector<float>> soft;
                if (use_soft && i < multiclass_probs->size())
                    soft = std::make_shared<std::vector<float>>((*multiclass_probs)[i]);
                xy.emplace_back(p, cloud[i].label, w, std::move(soft));
            }

            PointCloud frees_n;
            beam_sample(p, origin, frees_n, free_resolution);
            PCLPointType p_origin;
            p_origin.x = origin.x(); p_origin.y = origin.y(); p_origin.z = origin.z();
            p_origin.label = 0;
            frees.push_back(p_origin);
            for (auto fp = frees_n.begin(); fp != frees_n.end(); ++fp) {
                PCLPointType pf;
                pf.x = fp->x(); pf.y = fp->y(); pf.z = fp->z(); pf.label = 0;
                frees.push_back(pf);
                frees.width++;
            }
        }

        if (ds_resolution > 0) {
            for (auto &kv : voxels) {
                auto &v = kv.second;
                float cx = static_cast<float>(v.sx / v.count);
                float cy = static_cast<float>(v.sy / v.count);
                float cz = static_cast<float>(v.sz / v.count);
                float avg_w = static_cast<float>(v.w_sum / v.count);
                std::shared_ptr<std::vector<float>> soft;
                if (use_soft && v.prob_sum.size() == static_cast<size_t>(nc)) {
                    soft = std::make_shared<std::vector<float>>(static_cast<size_t>(nc));
                    for (int c = 0; c < nc; ++c)
                        (*soft)[static_cast<size_t>(c)] = static_cast<float>(v.prob_sum[static_cast<size_t>(c)] / v.count);
                }
                xy.emplace_back(point3f(cx, cy, cz), v.label, avg_w, std::move(soft));
            }
        }

        PCLPointCloud sampled_frees;
        downsample(frees, sampled_frees, ds_resolution);
        for (auto it = sampled_frees.begin(); it != sampled_frees.end(); ++it) {
            xy.emplace_back(point3f(it->x, it->y, it->z), 0.0f, 1.0f, nullptr);
        }
    }

    void SemanticBKIOctoMap::insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin,
                                      float ds_resolution, float free_res, float max_range,
                                      const std::vector<float> &point_weights) {
        insert_pointcloud(cloud, origin, ds_resolution, free_res, max_range, point_weights, nullptr);
    }

    void SemanticBKIOctoMap::insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin,
                                      float ds_resolution, float free_res, float max_range,
                                      const std::vector<float> &point_weights,
                                      const std::vector<std::vector<float>> *multiclass_probs) {
        bool use_soft = (multiclass_probs != nullptr && !multiclass_probs->empty() &&
                         multiclass_probs->size() == cloud.size());
        GPPointCloud xy;
        if (use_soft)
            get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy, point_weights, multiclass_probs);
        else
            get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy, point_weights);

        if (xy.size() == 0) return;

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);
        osm_height_min_z_ = lim_min.z();
        osm_height_max_z_ = lim_max.z();

        // Pre-filter OSM geometry to scan vicinity
        float cx = (lim_min.x() + lim_max.x()) * 0.5f;
        float cy = (lim_min.y() + lim_max.y()) * 0.5f;
        float half_dx = (lim_max.x() - lim_min.x()) * 0.5f;
        float half_dy = (lim_max.y() - lim_min.y()) * 0.5f;
        float max_xy = std::sqrt(half_dx * half_dx + half_dy * half_dy);
        filter_osm_for_scan(cx, cy, max_xy);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float p[] = {it->first.x(), it->first.y(), it->first.z()};
            rtree.Insert(p, p, const_cast<GPPointType *>(&*it));
        }

        int nc = SemanticOcTreeNode::num_class;
        vector<BlockHashKey> test_blocks;
        std::unordered_map<BlockHashKey, SemanticBKI3f *> bgk_arr;
#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < blocks.size(); ++i) {
            BlockHashKey key = blocks[i];
            ExtendedBlock eblock = get_extended_block(key);
            if (has_gp_points_in_bbox(eblock))
#ifdef OPENMP
#pragma omp critical
#endif
            { test_blocks.push_back(key); };

            GPPointCloud block_xy;
            get_gp_points_in_bbox(key, block_xy);
            if (block_xy.size() < 1) continue;

            vector<float> block_x, block_y, block_w;
            std::vector<std::vector<float>> block_y_soft, block_w_class;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());
                block_y.push_back(it->second);
                // Per-class OSM semantic kernel (boost-only, no suppression)
                block_w.push_back(it->weight);
                std::vector<float> k_vec(nc, 1.0f);
                if (use_soft) {
                    if (it->soft_probs && it->soft_probs->size() == static_cast<size_t>(nc)) {
                        block_y_soft.push_back(*it->soft_probs);
                        compute_osm_semantic_kernel(it->first.x(), it->first.y(), it->first.z(), k_vec);
                    } else {
                        // Free-space points: zero soft probs, no OSM modulation
                        std::vector<float> zero_vec(static_cast<size_t>(nc), 0.f);
                        block_y_soft.push_back(std::move(zero_vec));
                    }
                } else {
                    int lbl = static_cast<int>(it->second);
                    if (lbl > 0 && lbl < nc)
                        compute_osm_semantic_kernel(it->first.x(), it->first.y(), it->first.z(), k_vec);
                }
                block_w_class.push_back(std::move(k_vec));
            }

            SemanticBKI3f *bgk = new SemanticBKI3f(nc, SemanticOcTreeNode::sf2, SemanticOcTreeNode::ell);
            if (use_soft && block_y_soft.size() == block_xy.size())
                bgk->train_soft(block_x, block_y_soft, block_w, block_w_class);
            else
                bgk->train(block_x, block_y, block_w, block_w_class);
#ifdef OPENMP
#pragma omp critical
#endif
            { bgk_arr.emplace(key, bgk); };
        }

#ifdef OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
        for (int i = 0; i < test_blocks.size(); ++i) {
            BlockHashKey key = test_blocks[i];
            Block *block = nullptr;
#ifdef OPENMP
#pragma omp critical
#endif
            {
                bool is_new_block = (block_arr.find(key) == block_arr.end());
                if (is_new_block)
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
                block = block_arr[key];
                if (is_new_block)
                    init_osm_prior_for_block(block);
            };
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }

            ExtendedBlock eblock = block->get_extended_block();
            for (auto block_it = eblock.cbegin(); block_it != eblock.cend(); ++block_it) {
                auto bgk = bgk_arr.find(*block_it);
                if (bgk == bgk_arr.end()) continue;
                vector<vector<float>> ybars;
                if (use_soft)
                    bgk->second->predict_csm(xs, ybars);
                else
                    bgk->second->predict(xs, ybars);
                int j = 0;
                for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                    SemanticOcTreeNode &node = leaf_it.get_node();
                    // OSM prior is now in the kernel weights — no post-prediction bias needed
                    node.update(ybars[j]);
                }
            }
        }

        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;
        clear_osm_scan_filter();
        rtree.RemoveAll();
    }

    void SemanticBKIOctoMap::downsample(const PCLPointCloud &in, PCLPointCloud &out, float ds_resolution) const {
        if (ds_resolution < 0) {
            out = in;
            return;
        }

        PCLPointCloud::Ptr pcl_in(new PCLPointCloud(in));

        pcl::VoxelGrid<PCLPointType> sor;
        sor.setInputCloud(pcl_in);
        sor.setLeafSize(ds_resolution, ds_resolution, ds_resolution);
        sor.filter(out);
    }

    void SemanticBKIOctoMap::beam_sample(const point3f &hit, const point3f &origin, PointCloud &frees,
                                float free_resolution) const {
        frees.clear();

        float x0 = origin.x();
        float y0 = origin.y();
        float z0 = origin.z();

        float x = hit.x();
        float y = hit.y();
        float z = hit.z();

        float l = (float) sqrt((x - x0) * (x - x0) + (y - y0) * (y - y0) + (z - z0) * (z - z0));

        float nx = (x - x0) / l;
        float ny = (y - y0) / l;
        float nz = (z - z0) / l;

        float d = free_resolution;
        while (d < l) {
            frees.emplace_back(x0 + nx * d, y0 + ny * d, z0 + nz * d);
            d += free_resolution;
        }
        if (l > free_resolution)
            frees.emplace_back(x0 + nx * (l - free_resolution), y0 + ny * (l - free_resolution), z0 + nz * (l - free_resolution));
    }

    /*
     * Compute bounding box of pointcloud
     * Precondition: cloud non-empty
     */
    void SemanticBKIOctoMap::bbox(const GPPointCloud &cloud, point3f &lim_min, point3f &lim_max) const {
        assert(cloud.size() > 0);
        vector<float> x, y, z;
        for (auto it = cloud.cbegin(); it != cloud.cend(); ++it) {
            x.push_back(it->first.x());
            y.push_back(it->first.y());
            z.push_back(it->first.z());
        }

        auto xlim = std::minmax_element(x.cbegin(), x.cend());
        auto ylim = std::minmax_element(y.cbegin(), y.cend());
        auto zlim = std::minmax_element(z.cbegin(), z.cend());

        lim_min.x() = *xlim.first;
        lim_min.y() = *ylim.first;
        lim_min.z() = *zlim.first;

        lim_max.x() = *xlim.second;
        lim_max.y() = *ylim.second;
        lim_max.z() = *zlim.second;
    }

    void SemanticBKIOctoMap::get_blocks_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                       vector<BlockHashKey> &blocks) const {
        for (float x = lim_min.x() - block_size; x <= lim_max.x() + 2 * block_size; x += block_size) {
            for (float y = lim_min.y() - block_size; y <= lim_max.y() + 2 * block_size; y += block_size) {
                for (float z = lim_min.z() - block_size; z <= lim_max.z() + 2 * block_size; z += block_size) {
                    blocks.push_back(block_to_hash_key(x, y, z));
                }
            }
        }
    }

    int SemanticBKIOctoMap::get_gp_points_in_bbox(const BlockHashKey &key,
                                         GPPointCloud &out) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return get_gp_points_in_bbox(lim_min, lim_max, out);
    }

    int SemanticBKIOctoMap::has_gp_points_in_bbox(const BlockHashKey &key) {
        point3f half_size(block_size / 2.0f, block_size / 2.0f, block_size / 2.0);
        point3f lim_min = hash_key_to_block(key) - half_size;
        point3f lim_max = hash_key_to_block(key) + half_size;
        return has_gp_points_in_bbox(lim_min, lim_max);
    }

    int SemanticBKIOctoMap::get_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                         GPPointCloud &out) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, SemanticBKIOctoMap::search_callback, static_cast<void *>(&out));
    }

    int SemanticBKIOctoMap::has_gp_points_in_bbox(const point3f &lim_min,
                                         const point3f &lim_max) {
        float a_min[] = {lim_min.x(), lim_min.y(), lim_min.z()};
        float a_max[] = {lim_max.x(), lim_max.y(), lim_max.z()};
        return rtree.Search(a_min, a_max, SemanticBKIOctoMap::count_callback, NULL);
    }

    bool SemanticBKIOctoMap::count_callback(GPPointType *p, void *arg) {
        return false;
    }

    bool SemanticBKIOctoMap::search_callback(GPPointType *p, void *arg) {
        GPPointCloud *out = static_cast<GPPointCloud *>(arg);
        out->push_back(*p);
        return true;
    }


    int SemanticBKIOctoMap::has_gp_points_in_bbox(const ExtendedBlock &block) {
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            if (has_gp_points_in_bbox(*it) > 0)
                return 1;
        }
        return 0;
    }

    int SemanticBKIOctoMap::get_gp_points_in_bbox(const ExtendedBlock &block,
                                         GPPointCloud &out) {
        int n = 0;
        for (auto it = block.cbegin(); it != block.cend(); ++it) {
            n += get_gp_points_in_bbox(*it, out);
        }
        return n;
    }

    Block *SemanticBKIOctoMap::search(BlockHashKey key) const {
        auto block = block_arr.find(key);
        if (block == block_arr.end()) {
            return nullptr;
        } else {
            return block->second;
        }
    }

    SemanticOcTreeNode SemanticBKIOctoMap::search(point3f p) const {
        Block *block = search(block_to_hash_key(p));
        if (block == nullptr) {
          return SemanticOcTreeNode();
        } else {
          return SemanticOcTreeNode(block->search(p));
        }
    }

    SemanticOcTreeNode SemanticBKIOctoMap::search(float x, float y, float z) const {
        return search(point3f(x, y, z));
    }
}
