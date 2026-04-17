#pragma once

#include <array>
#include <memory>
#include <unordered_map>
#include <vector>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "rtree.h"
#include "bkiblock.h"
#include "bkioctree_node.h"
#include "osm_geometry.h"

namespace osm_bki {

    /// PCL PointCloud types as input
    typedef pcl::PointXYZL PCLPointType;
    typedef pcl::PointCloud<PCLPointType> PCLPointCloud;

    /*
     * @brief BGKOctoMap
     *
     * Bayesian Generalized Kernel Inference for Occupancy Map Prediction
     * The space is partitioned by Blocks in which OcTrees with fixed
     * depth are rooted. Occupancy values in one Block is predicted by 
     * its ExtendedBlock via Bayesian generalized kernel inference.
     */
    class SemanticBKIOctoMap {
    public:
        /// Number of OSM prior categories (matches columns of optimized confusion matrices):
        /// roads, sidewalks, cycleways, parking, grasslands, trees, forest, buildings, fences, none.
        static constexpr int N_OSM_PRIOR_COLS = 10;

        /// Types used internally
        typedef std::vector<point3f> PointCloud;
        struct GPPointType {
            point3f first;
            float second;
            float weight;
            /// Optional soft (multiclass probability) label for counting sensor model; size = num_class.
            std::shared_ptr<std::vector<float>> soft_probs;
            GPPointType() : first(), second(0), weight(1.0f), soft_probs() {}
            GPPointType(const point3f& p, float label, float w = 1.0f,
                        std::shared_ptr<std::vector<float>> soft = nullptr)
                : first(p), second(label), weight(w), soft_probs(std::move(soft)) {}
        };
        typedef std::vector<GPPointType> GPPointCloud;
        typedef RTree<GPPointType *, float, 3, float> MyRTree;

    public:
        SemanticBKIOctoMap();

        /*
         * @param resolution (default 0.1m)
         * @param block_depth maximum depth of OcTree (default 4)
         * @param sf2 signal variance in GPs (default 1.0)
         * @param ell length-scale in GPs (default 1.0)
         * @param noise noise variance in GPs (default 0.01)
         * @param l length-scale in logistic regression function (default 100)
         * @param min_var minimum variance in Occupancy (default 0.001)
         * @param max_var maximum variance in Occupancy (default 1000)
         * @param max_known_var maximum variance for Occuapncy to be classified as KNOWN State (default 0.02)
         * @param free_thresh free threshold for Occupancy probability (default 0.3)
         * @param occupied_thresh occupied threshold for Occupancy probability (default 0.7)
         */
        SemanticBKIOctoMap(float resolution,
                unsigned short block_depth,
                int num_class,
                float sf2,
                float ell,
                float prior,
                float var_thresh,
                float free_thresh,
                float occupied_thresh);

        ~SemanticBKIOctoMap();

        /// Set resolution.
        void set_resolution(float resolution);

        /// Set block max depth.
        void set_block_depth(unsigned short max_depth);

        /// Get resolution.
        inline float get_resolution() const { return resolution; }

        /// Get block max depth.
        inline float get_block_depth() const { return block_depth; }

        /*
         * @brief Insert PCL PointCloud into BGKOctoMaps.
         * @param cloud one scan in PCLPointCloud format
         * @param origin sensor origin in the scan
         * @param ds_resolution downsampling resolution for PCL VoxelGrid filtering (-1 if no downsampling)
         * @param free_res resolution for sampling free training points along sensor beams (default 2.0)
         * @param max_range maximum range for beams to be considered as valid measurements (-1 if no limitation)
         */
        void insert_pointcloud_csm(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                               float free_res = 2.0f,
                               float max_range = -1);


        void insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                               float free_res = 2.0f,
                               float max_range = -1);

        /// Weighted variant: per-point weights discount uncertain training points
        /// in the BKI kernel.  point_weights must have the same size as cloud.
        void insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                               float free_res, float max_range,
                               const std::vector<float> &point_weights);

        /// Weighted + soft labels: when multiclass_probs is non-null and matches cloud size,
        /// uses counting sensor model with soft counts (train_soft + predict_csm).
        void insert_pointcloud(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                               float free_res, float max_range,
                               const std::vector<float> &point_weights,
                               const std::vector<std::vector<float>> *multiclass_probs);

        /// Get bounding box of the map.
        void get_bbox(point3f &lim_min, point3f &lim_max) const;

        class RayCaster {
        public:
            RayCaster(const SemanticBKIOctoMap *map, const point3f &start, const point3f &end) : map(map) {
                assert(map != nullptr);

                _block_key = block_to_hash_key(start);
                block = map->search(_block_key);
                lim = static_cast<unsigned short>(pow(2, map->block_depth - 1));

                if (block != nullptr) {
                    block->get_index(start, x, y, z);
                    block_lim = block->get_center();
                    block_size = block->size;
                    current_p = start;
                    resolution = map->resolution;

                    int x0 = static_cast<int>((start.x() / resolution));
                    int y0 = static_cast<int>((start.y() / resolution));
                    int z0 = static_cast<int>((start.z() / resolution));
                    int x1 = static_cast<int>((end.x() / resolution));
                    int y1 = static_cast<int>((end.y() / resolution));
                    int z1 = static_cast<int>((end.z() / resolution));
                    dx = abs(x1 - x0);
                    dy = abs(y1 - y0);
                    dz = abs(z1 - z0);
                    n = 1 + dx + dy + dz;
                    x_inc = x1 > x0 ? 1 : (x1 == x0 ? 0 : -1);
                    y_inc = y1 > y0 ? 1 : (y1 == y0 ? 0 : -1);
                    z_inc = z1 > z0 ? 1 : (z1 == z0 ? 0 : -1);
                    xy_error = dx - dy;
                    xz_error = dx - dz;
                    yz_error = dy - dz;
                    dx *= 2;
                    dy *= 2;
                    dz *= 2;
                } else {
                    n = 0;
                }
            }

            inline bool end() const { return n <= 0; }

            bool next(point3f &p, SemanticOcTreeNode &node, BlockHashKey &block_key, OcTreeHashKey &node_key) {
                assert(!end());
                bool valid = false;
                unsigned short index = x + y * lim + z * lim * lim;
                node_key = Block::index_map[index];
                block_key = _block_key;
                if (block != nullptr) {
                    valid = true;
                    node = (*block)[node_key];
                    current_p = block->get_point(x, y, z);
                    p = current_p;
                } else {
                    p = current_p;
                }

                if (xy_error > 0 && xz_error > 0) {
                    x += x_inc;
                    current_p.x() += x_inc * resolution;
                    xy_error -= dy;
                    xz_error -= dz;
                    if (x >= lim || x < 0) {
                        block_lim.x() += x_inc * block_size;
                        _block_key = block_to_hash_key(block_lim);
                        block = map->search(_block_key);
                        x = x_inc > 0 ? 0 : lim - 1;
                    }
                } else if (xy_error < 0 && yz_error > 0) {
                    y += y_inc;
                    current_p.y() += y_inc * resolution;
                    xy_error += dx;
                    yz_error -= dz;
                    if (y >= lim || y < 0) {
                        block_lim.y() += y_inc * block_size;
                        _block_key = block_to_hash_key(block_lim);
                        block = map->search(_block_key);
                        y = y_inc > 0 ? 0 : lim - 1;
                    }
                } else if (yz_error < 0 && xz_error < 0) {
                    z += z_inc;
                    current_p.z() += z_inc * resolution;
                    xz_error += dx;
                    yz_error += dy;
                    if (z >= lim || z < 0) {
                        block_lim.z() += z_inc * block_size;
                        _block_key = block_to_hash_key(block_lim);
                        block = map->search(_block_key);
                        z = z_inc > 0 ? 0 : lim - 1;
                    }
                } else if (xy_error == 0) {
                    x += x_inc;
                    y += y_inc;
                    n -= 2;
                    current_p.x() += x_inc * resolution;
                    current_p.y() += y_inc * resolution;
                    if (x >= lim || x < 0) {
                        block_lim.x() += x_inc * block_size;
                        _block_key = block_to_hash_key(block_lim);
                        block = map->search(_block_key);
                        x = x_inc > 0 ? 0 : lim - 1;
                    }
                    if (y >= lim || y < 0) {
                        block_lim.y() += y_inc * block_size;
                        _block_key = block_to_hash_key(block_lim);
                        block = map->search(_block_key);
                        y = y_inc > 0 ? 0 : lim - 1;
                    }
                }
                n--;
                return valid;
            }

        private:
            const SemanticBKIOctoMap *map;
            Block *block;
            point3f block_lim;
            float block_size, resolution;
            int dx, dy, dz, error, n;
            int x_inc, y_inc, z_inc, xy_error, xz_error, yz_error;
            unsigned short index, x, y, z, lim;
            BlockHashKey _block_key;
            point3f current_p;
        };

        /// LeafIterator for iterating all leaf nodes in blocks
        class LeafIterator : public std::iterator<std::forward_iterator_tag, SemanticOcTreeNode> {
        public:
            LeafIterator(const SemanticBKIOctoMap *map) {
                assert(map != nullptr);

                block_it = map->block_arr.cbegin();
                end_block = map->block_arr.cend();

                if (map->block_arr.size() > 0) {
                    leaf_it = block_it->second->begin_leaf();
                    end_leaf = block_it->second->end_leaf();
                } else {
                    leaf_it = SemanticOcTree::LeafIterator();
                    end_leaf = SemanticOcTree::LeafIterator();
                }
            }

            // just for initializing end iterator
            LeafIterator(std::unordered_map<BlockHashKey, Block *>::const_iterator block_it,
                         SemanticOcTree::LeafIterator leaf_it)
                    : block_it(block_it), leaf_it(leaf_it), end_block(block_it), end_leaf(leaf_it) { }

            bool operator==(const LeafIterator &other) {
                return (block_it == other.block_it) && (leaf_it == other.leaf_it);
            }

            bool operator!=(const LeafIterator &other) {
                return !(this->operator==(other));
            }

            LeafIterator operator++(int) {
                LeafIterator result(*this);
                ++(*this);
                return result;
            }

            LeafIterator &operator++() {
                ++leaf_it;
                if (leaf_it == end_leaf) {
                    ++block_it;
                    if (block_it != end_block) {
                        leaf_it = block_it->second->begin_leaf();
                        end_leaf = block_it->second->end_leaf();
                    }
                }
                return *this;
            }

            SemanticOcTreeNode &operator*() const {
                return *leaf_it;
            }

            std::vector<point3f> get_pruned_locs() const {
                std::vector<point3f> pruned_locs;
                point3f center = get_loc();
                float size = get_size();
                float x0 = center.x() - size * 0.5 + Block::resolution * 0.5;
                float y0 = center.y() - size * 0.5 + Block::resolution * 0.5;
                float z0 = center.z() - size * 0.5 + Block::resolution * 0.5;
                float x1 = center.x() + size * 0.5;
                float y1 = center.y() + size * 0.5;
                float z1 = center.z() + size * 0.5;
                for (float x = x0; x < x1; x += Block::resolution) {
                    for (float y = y0; y < y1; y += Block::resolution) {
                        for (float z = z0; z < z1; z += Block::resolution) {
                            pruned_locs.emplace_back(x, y, z);
                        }
                    }
                }
                return pruned_locs;
            }

            inline SemanticOcTreeNode &get_node() const {
                return operator*();
            }

            inline point3f get_loc() const {
                return block_it->second->get_loc(leaf_it);
            }

            inline float get_size() const {
                return block_it->second->get_size(leaf_it);
            }

        private:
            std::unordered_map<BlockHashKey, Block *>::const_iterator block_it;
            std::unordered_map<BlockHashKey, Block *>::const_iterator end_block;

            SemanticOcTree::LeafIterator leaf_it;
            SemanticOcTree::LeafIterator end_leaf;
        };

        /// @return the beginning of leaf iterator
        inline LeafIterator begin_leaf() const { return LeafIterator(this); }

        /// @return the end of leaf iterator
        inline LeafIterator end_leaf() const { return LeafIterator(block_arr.cend(), SemanticOcTree::LeafIterator()); }

        SemanticOcTreeNode search(point3f p) const;

        SemanticOcTreeNode search(float x, float y, float z) const;

        Block *search(BlockHashKey key) const;

        inline float get_block_size() const { return block_size; }

        /// OSM semantics: set 2D geometries (same frame as map) and decay distance for prior dropoff.
        void set_osm_buildings(const std::vector<Geometry2D> &buildings);
        void set_osm_roads(const std::vector<Geometry2D> &roads);
        void set_osm_sidewalks(const std::vector<Geometry2D> &sidewalks);
        void set_osm_cycleways(const std::vector<Geometry2D> &cycleways);
        void set_osm_grasslands(const std::vector<Geometry2D> &grasslands);
        void set_osm_trees(const std::vector<Geometry2D> &trees);
        void set_osm_forests(const std::vector<Geometry2D> &forests);
        void set_osm_tree_points(const std::vector<std::pair<float, float>> &tree_points);
        void set_osm_tree_point_radius(float radius_m);
        void set_osm_parking(const std::vector<Geometry2D> &parking);
        void set_osm_fences(const std::vector<Geometry2D> &fences);
        void set_osm_road_width(float width_m);
        void set_osm_sidewalk_width(float width_m);
        void set_osm_cycleway_width(float width_m);
        void set_osm_fence_width(float width_m);
        void set_osm_decay_meters(float decay_m);

        /// OSM confusion matrix: set pre-parsed matrix and label mappings.
        /// K_pred rows (semantic super-classes) x K_prior cols (OSM categories).
        /// Column order: [roads, sidewalks, cycleways, parking, grasslands, trees, forest, buildings, fences, none].
        /// Values in [-1, 1]: negative = decrease likelihood, positive = increase.
        /// "none" column is active (1.0) when no OSM geometry covers the point.
        /// @param matrix  rows x N_OSM_PRIOR_COLS values, outer index = row
        /// @param row_to_labels  for each row, list of raw label IDs that map to it
        void set_osm_confusion_matrix(const std::vector<std::vector<float>> &matrix,
                                      const std::vector<std::vector<int>> &row_to_labels);
        void set_osm_prior_strength(float strength);

        /// Set strength for spatially varying Dirichlet prior from OSM.
        /// Controls how much OSM biases unobserved voxels toward expected classes.
        /// 0.0 = uniform prior everywhere (standard BKI), higher = stronger OSM bias.
        void set_osm_dirichlet_prior_strength(float strength);

        /// Initialize OSM-derived Dirichlet priors for all leaf nodes in a block.
        /// Only affects unclassified nodes (new blocks). Called during insert_pointcloud.
        void init_osm_prior_for_block(Block *block);

        /// OSM height filter: fixed-metric bins measured upward from the per-scan
        /// bottom-most point along a fixed reference "up" axis (the +z of the first
        /// scan's lidar, set once via set_osm_height_up_ref). Multiplies OSM priors
        /// by rows of the height confusion matrix.
        void set_osm_height_filter_enabled(bool enabled);
        void set_osm_height_confusion_matrix(const std::vector<std::vector<float>> &matrix);
        /// Set the bin step (meters per row of the confusion matrix). Matches the
        /// osm_height_bin_step_meters field written by the optimizer.
        void set_osm_height_bin_step(float step_meters);
        /// Set the fixed reference up axis (unit 3-vector in the map frame). Typically
        /// the +z of the first scan's lidar. Called once before the insert loop.
        void set_osm_height_up_ref(float ux, float uy, float uz);

        /// Set scan radius extension factor for OSM geometry pre-filtering.
        /// The filter radius = max_xy_distance * extension_factor + osm_decay_meters.
        void set_osm_scan_radius_extension(float factor) { osm_scan_radius_extension_ = factor; }

        /// Pre-filter OSM geometry to only polygons/points overlapping a circle.
        /// Call once per scan before any per-point OSM queries.
        void filter_osm_for_scan(float center_x, float center_y, float max_xy_dist);

        /// Clear per-scan filtered geometry (called after scan processing).
        void clear_osm_scan_filter();

        /// OSM priors for visualization: compute on-the-fly (building, road, grassland, tree, parking, fence).
        void get_osm_priors_for_visualization(float x, float y, float &building, float &road, float &grassland,
                                              float &tree, float &parking, float &fence) const;

    private:
        void compute_osm_prior_vec(float x, float y, float osm_vec[N_OSM_PRIOR_COLS]) const;

        void apply_osm_prior_to_ybars(std::vector<float> &ybars, float x, float y, float z, float scale) const;

        /// Compute per-class semantic kernel weights from OSM prior at a training point.
        /// Returns a vector of size num_class where each element is a multiplicative
        /// factor for that class's contribution. Classes supported by OSM get boosted
        /// (> 1.0), others stay at 1.0 (no suppression). Height filtering modulates
        /// the OSM prior before computing per-class support.
        /// Fills k_vec with all 1.0s when OSM is disabled (backward compatible).
        void compute_osm_semantic_kernel(float x, float y, float z,
                                         std::vector<float> &k_vec) const;

        /// Compute OSM priors at (x,y): building (polygon), road (polyline), grassland (polygon), tree (polygon + points), parking (polygon), fence (polyline), stairs (polyline with width).
        float compute_osm_building_prior(float x, float y) const;
        float compute_osm_road_prior(float x, float y) const;
        float compute_osm_grassland_prior(float x, float y) const;
        float compute_osm_tree_prior(float x, float y) const;
        float compute_osm_forest_prior(float x, float y) const;
        float compute_osm_parking_prior(float x, float y) const;
        float compute_osm_fence_prior(float x, float y) const;
        float compute_osm_sidewalk_prior(float x, float y) const;
        float compute_osm_cycleway_prior(float x, float y) const;

    private:
        /// @return true if point is inside a bounding box given min and max limits.
        inline bool gp_point_in_bbox(const GPPointType &p, const point3f &lim_min, const point3f &lim_max) const {
            return (p.first.x() > lim_min.x() && p.first.x() < lim_max.x() &&
                    p.first.y() > lim_min.y() && p.first.y() < lim_max.y() &&
                    p.first.z() > lim_min.z() && p.first.z() < lim_max.z());
        }

        /// Get the bounding box of a pointcloud.
        void bbox(const GPPointCloud &cloud, point3f &lim_min, point3f &lim_max) const;

        /// Get all block indices inside a bounding box.
        void get_blocks_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                std::vector<BlockHashKey> &blocks) const;

        /// Get all points inside a bounding box assuming pointcloud has been inserted in rtree before.
        int get_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max,
                                  GPPointCloud &out);

        /// @return true if point exists inside a bounding box assuming pointcloud has been inserted in rtree before.
        int has_gp_points_in_bbox(const point3f &lim_min, const point3f &lim_max);

        /// Get all points inside a bounding box (block) assuming pointcloud has been inserted in rtree before.
        int get_gp_points_in_bbox(const BlockHashKey &key, GPPointCloud &out);

        /// @return true if point exists inside a bounding box (block) assuming pointcloud has been inserted in rtree before.
        int has_gp_points_in_bbox(const BlockHashKey &key);

        /// Get all points inside an extended block assuming pointcloud has been inserted in rtree before.
        int get_gp_points_in_bbox(const ExtendedBlock &block, GPPointCloud &out);

        /// @return true if point exists inside an extended block assuming pointcloud has been inserted in rtree before.
        int has_gp_points_in_bbox(const ExtendedBlock &block);

        /// RTree callback function
        static bool count_callback(GPPointType *p, void *arg);

        /// RTree callback function
        static bool search_callback(GPPointType *p, void *arg);

        /// Downsample PCLPointCloud using PCL VoxelGrid Filtering.
        void downsample(const PCLPointCloud &in, PCLPointCloud &out, float ds_resolution) const;

        /// Sample free training points along sensor beams.
        void beam_sample(const point3f &hits, const point3f &origin, PointCloud &frees,
                         float free_resolution) const;

        /// Get training data from one sensor scan.
        void get_training_data(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                               float free_resolution, float max_range, GPPointCloud &xy) const;

        /// Weighted variant: manual voxel downsampling that preserves per-point weights.
        void get_training_data(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                               float free_resolution, float max_range, GPPointCloud &xy,
                               const std::vector<float> &point_weights) const;

        /// Weighted + soft labels: like above but attaches mean soft probs per voxel for counting sensor model.
        void get_training_data(const PCLPointCloud &cloud, const point3f &origin, float ds_resolution,
                               float free_resolution, float max_range, GPPointCloud &xy,
                               const std::vector<float> &point_weights,
                               const std::vector<std::vector<float>> *multiclass_probs) const;

        float resolution;
        float block_size;
        unsigned short block_depth;
        std::unordered_map<BlockHashKey, Block *> block_arr;
        MyRTree rtree;
        std::vector<Geometry2D> osm_buildings_;
        std::vector<Geometry2D> osm_roads_;
        std::vector<Geometry2D> osm_sidewalks_;
        std::vector<Geometry2D> osm_cycleways_;
        std::vector<Geometry2D> osm_grasslands_;
        std::vector<Geometry2D> osm_trees_;
        std::vector<Geometry2D> osm_forests_;
        std::vector<std::pair<float, float>> osm_tree_points_;
        float osm_tree_point_radius_{5.0f};  // Radius (m) for tree point circles; prior projected same as polygons
        std::vector<Geometry2D> osm_parking_;
        std::vector<Geometry2D> osm_fences_;
        float osm_road_width_{6.0f};         // Width (m) for road polyline bands
        float osm_sidewalk_width_{2.0f};     // Width (m) for sidewalk polyline bands
        float osm_cycleway_width_{2.0f};     // Width (m) for cycleway polyline bands
        float osm_fence_width_{0.6f};        // Width (m) for fence polyline bands
        float osm_decay_meters_;

        // OSM confusion matrix for semantic-OSM prior fusion
        bool osm_cm_loaded_{false};
        float osm_prior_strength_{0.0f};
        float osm_dirichlet_prior_strength_{0.0f};
        float osm_scan_radius_extension_{1.2f};  // multiply max_xy_dist by this to get filter radius

        // Per-scan filtered OSM geometry (populated by filter_osm_for_scan, cleared after scan)
        bool osm_scan_filtered_{false};
        std::vector<Geometry2D> active_buildings_;
        std::vector<Geometry2D> active_roads_;
        std::vector<Geometry2D> active_sidewalks_;
        std::vector<Geometry2D> active_cycleways_;
        std::vector<Geometry2D> active_grasslands_;
        std::vector<Geometry2D> active_trees_;
        std::vector<Geometry2D> active_forests_;
        std::vector<std::pair<float, float>> active_tree_points_;
        std::vector<Geometry2D> active_parking_;
        std::vector<Geometry2D> active_fences_;
        int osm_cm_rows_{0};      // K_pred (number of semantic super-classes)
        float osm_cm_[13][N_OSM_PRIOR_COLS]{};  // confusion matrix [row][col], max 13 rows
        // For each confusion matrix row, list of raw label IDs (SemanticKITTI) that map to it
        std::vector<std::vector<int>> osm_cm_row_to_labels_;

        // OSM height filter: per-scan min/max z, bin count, height confusion matrix [bin][common_class_row]
        // Variant A: height multipliers are applied AFTER the OSM->common projection, indexed by
        // common-class row (same index space as osm_cm_ rows), so columns == osm_cm_rows_.
        bool use_osm_height_filter_{false};
        bool osm_height_cm_loaded_{false};
        int osm_height_num_bins_{0};
        float osm_height_step_meters_{1.0f};
        float osm_height_up_ref_x_{0.f};
        float osm_height_up_ref_y_{0.f};
        float osm_height_up_ref_z_{1.f};
        bool osm_height_up_ref_set_{false};
        float osm_height_z_base_{0.f};  // per-scan: min(up_ref · p) over training points
        std::vector<std::vector<float>> osm_height_cm_{};
    };

}
