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

namespace semantic_bki {

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

        if (use_osm_height_filter_ && osm_prior_strength_ > 0.0f)
            compute_osm_height_stats_from_cloud(cloud);

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

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

            vector<float> block_x, block_y;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());
                block_y.push_back(it->second);
            
            
            //std::cout << search(it->first.x(), it->first.y(), it->first.z()) << std::endl;
            }

            SemanticBKI3f *bgk = new SemanticBKI3f(SemanticOcTreeNode::num_class, SemanticOcTreeNode::sf2, SemanticOcTreeNode::ell);
            bgk->train(block_x, block_y);
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
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (block_arr.find(key) == block_arr.end())
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            Block *block = block_arr[key];
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }
            //std::cout << "xs size: "<<xs.size() << std::endl;

	          // For counting sensor model
            auto bgk = bgk_arr.find(key);
            if (bgk == bgk_arr.end())
              continue;

            vector<vector<float>> ybars;
            bgk->second->predict_csm(xs, ybars);

            int j = 0;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                SemanticOcTreeNode &node = leaf_it.get_node();
                point3f loc = block->get_loc(leaf_it);

                float ybar_sum = 0.f;
                for (auto v : ybars[j]) ybar_sum += std::abs(v);
                apply_osm_prior_to_ybars(ybars[j], loc.x(), loc.y(), loc.z(), std::min(ybar_sum, 1.0f));

                node.update(ybars[j]);
                node.set_osm_building(compute_osm_building_prior(loc.x(), loc.y()));
                node.set_osm_road(compute_osm_road_prior(loc.x(), loc.y()));
                node.set_osm_grassland(compute_osm_grassland_prior(loc.x(), loc.y()));
                node.set_osm_tree(compute_osm_tree_prior(loc.x(), loc.y()));
                node.set_osm_parking(compute_osm_parking_prior(loc.x(), loc.y()));
                node.set_osm_fence(compute_osm_fence_prior(loc.x(), loc.y()));
                node.set_osm_stairs(compute_osm_stairs_prior(loc.x(), loc.y()));
            }

        }
#ifdef DEBUG
        Debug_Msg("Prediction done");
#endif

        ////////// Cleaning /////////////////////////////
        /////////////////////////////////////////////////
        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;

        rtree.RemoveAll();
    }

    void SemanticBKIOctoMap::set_osm_buildings(const std::vector<Geometry2D> &buildings) {
        osm_buildings_ = buildings;
    }

    void SemanticBKIOctoMap::set_osm_roads(const std::vector<Geometry2D> &roads) {
        osm_roads_ = roads;
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

    void SemanticBKIOctoMap::set_osm_stairs(const std::vector<Geometry2D> &stairs) {
        osm_stairs_ = stairs;
    }

    void SemanticBKIOctoMap::set_osm_stairs_width(float width_m) {
        osm_stairs_width_ = std::max(0.1f, width_m);
    }

    void SemanticBKIOctoMap::set_osm_decay_meters(float decay_m) {
        osm_decay_meters_ = decay_m;
    }

    void SemanticBKIOctoMap::set_osm_prior_strength(float strength) {
        osm_prior_strength_ = strength;
    }

    void SemanticBKIOctoMap::set_osm_height_filter_enabled(bool enabled) {
        use_osm_height_filter_ = enabled;
    }

    void SemanticBKIOctoMap::set_osm_height_std_multiplier(float k) {
        osm_height_std_multiplier_ = std::max(0.5f, k);
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

    void SemanticBKIOctoMap::compute_osm_height_stats_from_cloud(const PCLPointCloud &cloud) {
        static constexpr int N_CAT = 8;  // roads, parking, grasslands, trees, forest, buildings, fences, stairs
        std::vector<std::vector<float>> z_per_cat(N_CAT);
        const float threshold = 0.3f;

        for (size_t i = 0; i < cloud.size(); ++i) {
            float x = cloud[i].x;
            float y = cloud[i].y;
            float z = cloud[i].z;
            float osm_vec[N_OSM_PRIOR_COLS];
            compute_osm_prior_vec(x, y, osm_vec);
            for (int c = 0; c < N_CAT; ++c) {
                if (osm_vec[c] >= threshold)
                    z_per_cat[c].push_back(z);
            }
        }

        for (int c = 0; c < N_CAT; ++c) {
            const size_t n = z_per_cat[c].size();
            if (n < 5u) {
                osm_height_valid_[c] = false;
                continue;
            }
            float sum = std::accumulate(z_per_cat[c].begin(), z_per_cat[c].end(), 0.f);
            osm_height_mean_[c] = sum / static_cast<float>(n);
            float var = 0.f;
            for (float z : z_per_cat[c])
                var += (z - osm_height_mean_[c]) * (z - osm_height_mean_[c]);
            osm_height_std_[c] = std::sqrt(var / static_cast<float>(n));
            if (osm_height_std_[c] < 1e-4f) osm_height_std_[c] = 0.2f;  // avoid division by zero
            osm_height_valid_[c] = true;
        }
    }

    void SemanticBKIOctoMap::compute_osm_prior_vec(float x, float y,
                                                    float osm_vec[N_OSM_PRIOR_COLS]) const {
        osm_vec[0] = compute_osm_road_prior(x, y);
        osm_vec[1] = compute_osm_parking_prior(x, y);
        osm_vec[2] = compute_osm_grassland_prior(x, y);
        osm_vec[3] = compute_osm_tree_prior(x, y);
        osm_vec[4] = compute_osm_forest_prior(x, y);
        osm_vec[5] = compute_osm_building_prior(x, y);
        osm_vec[6] = compute_osm_fence_prior(x, y);
        osm_vec[7] = compute_osm_stairs_prior(x, y);
        // "none" = 1 when no OSM geometry covers this point, 0 when fully covered
        float max_geom = 0.f;
        for (int c = 0; c < 8; ++c)
            if (osm_vec[c] > max_geom) max_geom = osm_vec[c];
        osm_vec[8] = 1.0f - max_geom;
    }

    void SemanticBKIOctoMap::apply_osm_prior_to_ybars(std::vector<float> &ybars,
                                                      float x, float y, float z, float scale) const {
        if (!osm_cm_loaded_ || osm_prior_strength_ <= 0.0f || scale <= 0.0f) return;

        float osm_vec[N_OSM_PRIOR_COLS];
        compute_osm_prior_vec(x, y, osm_vec);

        if (use_osm_height_filter_) {
            for (int c = 0; c < 8; ++c) {
                if (!osm_height_valid_[c]) continue;
                float mean = osm_height_mean_[c];
                float stdv = osm_height_std_[c];
                float k = osm_height_std_multiplier_;
                float lo = mean - k * stdv;
                float hi = mean + k * stdv;
                float w = (z >= lo && z <= hi) ? 1.0f : 0.0f;
                osm_vec[c] *= w;
            }
        }

        // p_super[row] = sum_j(M[row][j] * osm_vec[j])
        // M values in [-1, 1]; negative decreases likelihood, positive increases.
        std::vector<float> p_super(osm_cm_rows_, 0.f);
        for (int r = 0; r < osm_cm_rows_; ++r)
            for (int c = 0; c < N_OSM_PRIOR_COLS; ++c)
                p_super[r] += osm_cm_[r][c] * osm_vec[c];

        // Add OSM contribution directly to ybars so it accumulates with every
        // observation, maintaining a constant proportion of the total evidence.
        float effective = osm_prior_strength_ * scale;
        int num_class = static_cast<int>(ybars.size());
        for (int r = 0; r < osm_cm_rows_; ++r) {
            const auto &labels = osm_cm_row_to_labels_[r];
            if (labels.empty() || p_super[r] == 0.f) continue;
            float share = effective * p_super[r] / static_cast<float>(labels.size());
            for (int lbl : labels) {
                if (lbl >= 0 && lbl < num_class)
                    ybars[lbl] += share;
            }
        }
    }

    float SemanticBKIOctoMap::compute_osm_building_prior(float x, float y) const {
        if (osm_buildings_.empty()) return 0.f;
        float min_positive_d = std::numeric_limits<float>::max();
        for (const auto &poly : osm_buildings_) {
            float signed_d = distance_to_polygon_boundary(x, y, poly);
            if (signed_d <= 0.f) return 1.f;  // inside a building
            if (signed_d < min_positive_d) min_positive_d = signed_d;
        }
        return osm_prior_from_signed_distance(min_positive_d, osm_decay_meters_);
    }

    float SemanticBKIOctoMap::compute_osm_road_prior(float x, float y) const {
        if (osm_roads_.empty()) return 0.f;
        float min_d = std::numeric_limits<float>::max();
        for (const auto &road : osm_roads_) {
            float d = distance_to_polyline(x, y, road);
            if (d < min_d) min_d = d;
        }
        return osm_prior_from_distance(min_d, osm_decay_meters_);
    }

    float SemanticBKIOctoMap::compute_osm_grassland_prior(float x, float y) const {
        if (osm_grasslands_.empty()) return 0.f;
        float min_positive_d = std::numeric_limits<float>::max();
        for (const auto &poly : osm_grasslands_) {
            float signed_d = distance_to_polygon_boundary(x, y, poly);
            if (signed_d <= 0.f) return 1.f;  // inside grassland
            if (signed_d < min_positive_d) min_positive_d = signed_d;
        }
        return osm_prior_from_signed_distance(min_positive_d, osm_decay_meters_);
    }

    float SemanticBKIOctoMap::compute_osm_forest_prior(float x, float y) const {
        if (osm_forests_.empty()) return 0.f;
        float min_positive_d = std::numeric_limits<float>::max();
        for (const auto &poly : osm_forests_) {
            float signed_d = distance_to_polygon_boundary(x, y, poly);
            if (signed_d <= 0.f) return 1.f;  // inside forest
            if (signed_d < min_positive_d) min_positive_d = signed_d;
        }
        return osm_prior_from_signed_distance(min_positive_d, osm_decay_meters_);
    }

    float SemanticBKIOctoMap::compute_osm_tree_prior(float x, float y) const {
        float max_prior = 0.f;
        // Check tree polygons (forests/woods)
        if (!osm_trees_.empty()) {
            float min_positive_d = std::numeric_limits<float>::max();
            for (const auto &poly : osm_trees_) {
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
        if (!osm_tree_points_.empty()) {
            float min_signed_d = std::numeric_limits<float>::max();
            for (const auto& pt : osm_tree_points_) {
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
        if (osm_parking_.empty()) return 0.f;
        float max_prior = 0.f;
        float min_positive_d = std::numeric_limits<float>::max();
        for (const auto &poly : osm_parking_) {
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
        if (osm_fences_.empty()) return 0.f;
        float min_d = std::numeric_limits<float>::max();
        for (const auto &fence : osm_fences_) {
            float d = distance_to_polyline(x, y, fence);
            if (d < min_d) min_d = d;
        }
        return osm_prior_from_distance(min_d, osm_decay_meters_);
    }

    float SemanticBKIOctoMap::compute_osm_stairs_prior(float x, float y) const {
        if (osm_stairs_.empty()) return 0.f;
        float max_prior = 0.f;
        float min_positive_d = std::numeric_limits<float>::max();
        const float hw = osm_stairs_width_ * 0.5f;
        const float eps = 1e-6f;
        
        // Convert each stair polyline segment into a rectangle polygon and check if point is inside
        for (const auto &stair : osm_stairs_) {
            if (stair.coords.size() < 2) continue;
            
            // For each segment in the polyline, create a rectangle
            for (size_t i = 0; i < stair.coords.size() - 1; ++i) {
                float x1 = stair.coords[i].first;
                float y1 = stair.coords[i].second;
                float x2 = stair.coords[i + 1].first;
                float y2 = stair.coords[i + 1].second;
                float dx = x2 - x1;
                float dy = y2 - y1;
                float L = std::sqrt(dx * dx + dy * dy);
                if (L < eps) continue;
                
                // Compute rectangle corners (perpendicular to segment)
                float nx = -dy / L;
                float ny = dx / L;
                float c1x = x1 + hw * nx, c1y = y1 + hw * ny;
                float c2x = x1 - hw * nx, c2y = y1 - hw * ny;
                float c3x = x2 - hw * nx, c3y = y2 - hw * ny;
                float c4x = x2 + hw * nx, c4y = y2 + hw * ny;
                
                // Create rectangle polygon (closed: 4 corners + first corner again)
                Geometry2D rect;
                rect.coords.push_back({c1x, c1y});
                rect.coords.push_back({c2x, c2y});
                rect.coords.push_back({c3x, c3y});
                rect.coords.push_back({c4x, c4y});
                rect.coords.push_back({c1x, c1y});  // Close the polygon
                
                // Check if point is inside this rectangle
                float signed_d = distance_to_polygon_boundary(x, y, rect);
                if (signed_d <= 0.f) {
                    max_prior = 1.f;  // Inside rectangle, max prior
                    break;
                }
                if (signed_d < min_positive_d) min_positive_d = signed_d;
            }
            if (max_prior >= 1.f) break;  // Already found inside, no need to check more
        }
        
        if (max_prior >= 1.f) return 1.f;
        if (min_positive_d < std::numeric_limits<float>::max()) {
            float poly_prior = osm_prior_from_signed_distance(min_positive_d, osm_decay_meters_);
            if (poly_prior > max_prior) max_prior = poly_prior;
        }
        return max_prior;
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

        if (use_osm_height_filter_ && osm_prior_strength_ > 0.0f)
            compute_osm_height_stats_from_cloud(cloud);

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

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

            vector<float> block_x, block_y;
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());
                block_y.push_back(it->second);
            
            
            //std::cout << search(it->first.x(), it->first.y(), it->first.z()) << std::endl;
            }

            SemanticBKI3f *bgk = new SemanticBKI3f(SemanticOcTreeNode::num_class, SemanticOcTreeNode::sf2, SemanticOcTreeNode::ell);
            bgk->train(block_x, block_y);
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
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (block_arr.find(key) == block_arr.end())
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            Block *block = block_arr[key];
            vector<float> xs;
            for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it) {
                point3f p = block->get_loc(leaf_it);
                xs.push_back(p.x());
                xs.push_back(p.y());
                xs.push_back(p.z());
            }
            //std::cout << "xs size: "<<xs.size() << std::endl;

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
                    point3f loc = block->get_loc(leaf_it);

                    float ybar_sum = 0.f;
                    for (auto v : ybars[j]) ybar_sum += std::abs(v);
                    apply_osm_prior_to_ybars(ybars[j], loc.x(), loc.y(), loc.z(), std::min(ybar_sum, 1.0f));

                    node.update(ybars[j]);
                    node.set_osm_building(compute_osm_building_prior(loc.x(), loc.y()));
                    node.set_osm_road(compute_osm_road_prior(loc.x(), loc.y()));
                    node.set_osm_grassland(compute_osm_grassland_prior(loc.x(), loc.y()));
                    node.set_osm_tree(compute_osm_tree_prior(loc.x(), loc.y()));
                    node.set_osm_parking(compute_osm_parking_prior(loc.x(), loc.y()));
                    node.set_osm_fence(compute_osm_fence_prior(loc.x(), loc.y()));
                    node.set_osm_stairs(compute_osm_stairs_prior(loc.x(), loc.y()));
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

    void SemanticBKIOctoMap::insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin,
                                      float ds_resolution, float free_res, float max_range,
                                      const std::vector<float> &point_weights) {
        GPPointCloud xy;
        get_training_data(cloud, origin, ds_resolution, free_res, max_range, xy, point_weights);

        if (xy.size() == 0) return;

        if (use_osm_height_filter_ && osm_prior_strength_ > 0.0f)
            compute_osm_height_stats_from_cloud(cloud);

        point3f lim_min, lim_max;
        bbox(xy, lim_min, lim_max);

        vector<BlockHashKey> blocks;
        get_blocks_in_bbox(lim_min, lim_max, blocks);

        for (auto it = xy.cbegin(); it != xy.cend(); ++it) {
            float p[] = {it->first.x(), it->first.y(), it->first.z()};
            rtree.Insert(p, p, const_cast<GPPointType *>(&*it));
        }

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
            for (auto it = block_xy.cbegin(); it != block_xy.cend(); ++it) {
                block_x.push_back(it->first.x());
                block_x.push_back(it->first.y());
                block_x.push_back(it->first.z());
                block_y.push_back(it->second);
                block_w.push_back(it->weight);
            }

            SemanticBKI3f *bgk = new SemanticBKI3f(SemanticOcTreeNode::num_class, SemanticOcTreeNode::sf2, SemanticOcTreeNode::ell);
            bgk->train(block_x, block_y, block_w);
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
#ifdef OPENMP
#pragma omp critical
#endif
            {
                if (block_arr.find(key) == block_arr.end())
                    block_arr.emplace(key, new Block(hash_key_to_block(key)));
            };
            Block *block = block_arr[key];
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
                bgk->second->predict(xs, ybars);
                int j = 0;
                for (auto leaf_it = block->begin_leaf(); leaf_it != block->end_leaf(); ++leaf_it, ++j) {
                    SemanticOcTreeNode &node = leaf_it.get_node();
                    point3f loc = block->get_loc(leaf_it);

                    float ybar_sum = 0.f;
                    for (auto v : ybars[j]) ybar_sum += std::abs(v);
                    apply_osm_prior_to_ybars(ybars[j], loc.x(), loc.y(), loc.z(), std::min(ybar_sum, 1.0f));

                    node.update(ybars[j]);
                    node.set_osm_building(compute_osm_building_prior(loc.x(), loc.y()));
                    node.set_osm_road(compute_osm_road_prior(loc.x(), loc.y()));
                    node.set_osm_grassland(compute_osm_grassland_prior(loc.x(), loc.y()));
                    node.set_osm_tree(compute_osm_tree_prior(loc.x(), loc.y()));
                    node.set_osm_parking(compute_osm_parking_prior(loc.x(), loc.y()));
                    node.set_osm_fence(compute_osm_fence_prior(loc.x(), loc.y()));
                    node.set_osm_stairs(compute_osm_stairs_prior(loc.x(), loc.y()));
                }
            }
        }

        for (auto it = bgk_arr.begin(); it != bgk_arr.end(); ++it)
            delete it->second;
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
