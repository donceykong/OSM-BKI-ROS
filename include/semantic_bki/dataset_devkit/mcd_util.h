#pragma once

#include <fstream>
#include <sstream>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <memory>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include "bkioctomap.h"
#include "markerarray_pub.h"
#include "osm_geometry.h"

// ---------------------------------------------------------------------------
// Common taxonomy (13 classes) — loaded from labels_common.yaml at runtime.
// ---------------------------------------------------------------------------
static constexpr int N_COMMON = 13;

struct MulticlassResult {
  pcl::PointCloud<pcl::PointXYZL>::Ptr cloud;
  std::vector<float> variances;
  int n_classes = 0;
};

/// Voxel key for semantic uncertainty accumulation (input observation uncertainty per voxel)
struct VoxelKey {
  int x = 0, y = 0, z = 0;
  bool operator<(const VoxelKey& o) const {
    return std::tie(x, y, z) < std::tie(o.x, o.y, o.z);
  }
};

/// Convert IEEE 754 half-precision (uint16) to single-precision float.
static inline float half_to_float(uint16_t h) {
    uint32_t sign     = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x03FFu;

    if (exponent == 0) {
        if (mantissa == 0) {
            float f; uint32_t r = sign;
            std::memcpy(&f, &r, sizeof(f));
            return f;
        }
        while (!(mantissa & 0x0400u)) { mantissa <<= 1; exponent--; }
        exponent++; mantissa &= ~0x0400u;
    } else if (exponent == 31) {
        uint32_t r = sign | 0x7F800000u | (mantissa << 13);
        float f; std::memcpy(&f, &r, sizeof(f));
        return f;
    }

    exponent += (127 - 15);
    uint32_t r = sign | (exponent << 23) | (mantissa << 13);
    float f; std::memcpy(&f, &r, sizeof(f));
    return f;
}

class MCDData {
  public:
    MCDData(rclcpp::Node::SharedPtr node,
             double resolution, double block_depth,
             double sf2, double ell,
             int num_class, double free_thresh,
             double occupied_thresh, float var_thresh, 
             double ds_resolution,
             double free_resolution, double max_range,
             std::string map_topic,
             float prior)
      : node_(node)
      , resolution_(resolution)
      , num_class_(num_class)
      , ds_resolution_(ds_resolution)
      , free_resolution_(free_resolution)
      , max_range_(max_range)
      , tf_broadcaster_(node)
      , tf_buffer_(std::make_shared<tf2_ros::Buffer>(node->get_clock()))
      , tf_listener_(*tf_buffer_) {
        if (!node_) {
          RCLCPP_WARN_STREAM(rclcpp::get_logger("mcd_util"), "WARNING: MCDData constructor: node_ is null!");
        }
        RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: MCDData constructor: Creating octomap");
        map_ = new semantic_bki::SemanticBKIOctoMap(resolution, block_depth, num_class, sf2, ell, prior, var_thresh, free_thresh, occupied_thresh);
        if (!map_) {
          RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Failed to create SemanticBKIOctoMap!");
        } else {
          RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Octomap created successfully");
        }
        
        RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Creating MarkerArrayPub");
        m_pub_ = new semantic_bki::MarkerArrayPub(node_, map_topic, resolution);
        if (!m_pub_) {
          RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Failed to create MarkerArrayPub!");
        } else {
          RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: MarkerArrayPub created successfully");
        }
        
        // Publisher for individual scan point clouds
        RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Creating pointcloud publisher");
        pointcloud_pub_ = node_->create_publisher<sensor_msgs::msg::PointCloud2>("/mcd_scan_pointcloud", 10);
        if (!pointcloud_pub_) {
          RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Failed to create pointcloud publisher!");
        } else {
          RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Pointcloud publisher created successfully");
        }
        
        // Identity transformation for MCD (poses are already in world frame)
        init_trans_to_ground_ = Eigen::Matrix4d::Identity();
        // Body to LiDAR transformation (must be loaded from calibration - no default identity)
        // Will be set by load_calibration_from_params() - if not set, will error
        body_to_lidar_tf_ = Eigen::Matrix4d::Zero();  // Set to zero to detect if not loaded
        original_first_pose_ = Eigen::Matrix4d::Identity();  // Will be set when poses are loaded
        scan_indices_.clear();
        inferred_use_multiclass_ = false;
        gt_use_multiclass_ = false;
        common_label_config_loaded_ = false;
        use_uncertainty_filter_ = false;
        confusion_matrix_loaded_ = false;
        uncertainty_filter_mode_ = "confusion_matrix";
        uncertainty_drop_percent_ = 10.0f;
        uncertainty_min_weight_ = 0.1f;
        total_points_processed_ = 0;
        total_points_filtered_ = 0;
        std::memset(confusion_matrix_, 0, sizeof(confusion_matrix_));
        std::memset(class_precision_, 0, sizeof(class_precision_));
        RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: MCDData constructor completed");
      }

    bool read_lidar_poses(const std::string lidar_pose_name) {
      RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: read_lidar_poses: Opening file: " << lidar_pose_name);
      std::ifstream fPoses;
      fPoses.open(lidar_pose_name.c_str());
      if (!fPoses.is_open()) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Cannot open pose file " << lidar_pose_name);
        RCLCPP_ERROR_STREAM(node_->get_logger(), "Cannot open pose file " << lidar_pose_name);
        return false;
      }
      RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Pose file opened successfully");
      
      // Skip header line if present
      std::string header_line;
      std::getline(fPoses, header_line);
      
      // Check if header contains column names (common in CSV)
      bool has_header = (header_line.find("num") != std::string::npos || 
                         header_line.find("timestamp") != std::string::npos ||
                         header_line.find("x") != std::string::npos);
      
      if (!has_header) {
        // No header, rewind to beginning
        fPoses.close();
        fPoses.open(lidar_pose_name.c_str());
      }

      while (!fPoses.eof()) {
        std::string s;
        std::getline(fPoses, s);
        
        // Skip empty lines and comments
        if (s.empty() || s[0] == '#') {
          continue;
        }

        std::stringstream ss(s);
        std::string token;
        std::vector<double> values;
        
        // Parse CSV line (handles comma-separated or space-separated)
        char delimiter = ',';
        if (s.find(',') == std::string::npos) {
          delimiter = ' ';
        }
        
        while (std::getline(ss, token, delimiter)) {
          try {
            double val = std::stod(token);
            values.push_back(val);
          } catch (...) {
            // Skip invalid tokens
            continue;
          }
        }

        // Expect at least 8 values: num, timestamp, x, y, z, qx, qy, qz, qw
        // Or 9 values if first is index
        if (values.size() < 8) {
          continue;
        }

        // Extract scan index (num) - first column in CSV
        int scan_index = (int)values[0];
        
        // Extract pose values (skip num and timestamp, they're at indices 0 and 1)
        double x = values[2];
        double y = values[3];
        double z = values[4];
        double qx = values[5];
        double qy = values[6];
        double qz = values[7];
        double qw = values.size() > 8 ? values[8] : 1.0;  // Default qw to 1.0 if not provided

        // Convert quaternion to rotation matrix
        Eigen::Quaterniond quat(qw, qx, qy, qz);
        quat.normalize();
        
        // Build transformation matrix
        Eigen::Matrix4d t_matrix = Eigen::Matrix4d::Identity();
        t_matrix.block<3, 3>(0, 0) = quat.toRotationMatrix();
        t_matrix(0, 3) = x;
        t_matrix(1, 3) = y;
        t_matrix(2, 3) = z;
        
        lidar_poses_.push_back(t_matrix);
        scan_indices_.push_back(scan_index);
      }
      
      fPoses.close();
      RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Finished reading pose file, loaded " << lidar_poses_.size() << " poses");
      
      if (lidar_poses_.empty()) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: No poses loaded from " << lidar_pose_name);
        RCLCPP_ERROR_STREAM(node_->get_logger(), "No poses loaded from " << lidar_pose_name);
        return false;
      }
      
      // Store original first pose before transformation (needed for OSM data alignment)
      original_first_pose_ = lidar_poses_[0];
      
      // Make all poses relative to the first pose (set first pose to origin)
      // This means: transform all poses by the inverse of the first pose
      Eigen::Matrix4d first_pose_inverse = lidar_poses_[0].inverse();
      
      RCLCPP_INFO_STREAM(node_->get_logger(), "First pose before alignment:");
      RCLCPP_INFO_STREAM(node_->get_logger(), "  Translation: [" << lidar_poses_[0](0,3) << ", " << lidar_poses_[0](1,3) << ", " << lidar_poses_[0](2,3) << "]");
      
      // Transform all poses to be relative to the first pose
      for (size_t i = 0; i < lidar_poses_.size(); ++i) {
        lidar_poses_[i] = first_pose_inverse * lidar_poses_[i];
      }
      
      // Verify first pose is now at origin
      RCLCPP_INFO_STREAM(node_->get_logger(), "After alignment - First pose should be identity:");
      RCLCPP_INFO_STREAM(node_->get_logger(), "  Translation: [" << lidar_poses_[0](0,3) << ", " << lidar_poses_[0](1,3) << ", " << lidar_poses_[0](2,3) << "]");
      RCLCPP_INFO_STREAM(node_->get_logger(), "Loaded " << lidar_poses_.size() << " poses from " << lidar_pose_name << " (all relative to first pose)");
      
      return true;
    }
    
    // Get the original first pose (before transformation to origin)
    // This is needed to align OSM data with the same coordinate frame
    Eigen::Matrix4d getOriginalFirstPose() const {
      return original_first_pose_;
    }

    /// Enable multiclass for inferred labels: read per-class confidence scores and take argmax.
    void set_inferred_multiclass_mode(bool use_mc, const std::string& multiclass_dir) {
      inferred_use_multiclass_ = use_mc;
      inferred_multiclass_dir_ = multiclass_dir;
      if (use_mc) {
        RCLCPP_INFO_STREAM(node_->get_logger(),
            "Inferred labels multiclass mode enabled. Scores dir: " << multiclass_dir);
      }
    }

    /// Enable multiclass for GT labels (e.g. one-hot or per-class scores). Uses gt_label_dir.
    void set_gt_multiclass_mode(bool use_mc) {
      gt_use_multiclass_ = use_mc;
      if (use_mc) {
        RCLCPP_INFO_STREAM(node_->get_logger(),
            "GT labels multiclass mode enabled.");
      }
    }

    /// Load learning_map_inv from a label config YAML (e.g. labels_semkitti.yaml) for inferred labels.
    bool load_label_config(const std::string& yaml_path) {
      return load_learning_map_inv_(yaml_path, learning_map_inv_, "inferred");
    }

    /// Load learning_map_inv for GT labels (when GT is multiclass from a model).
    bool load_gt_label_config(const std::string& yaml_path) {
      return load_learning_map_inv_(yaml_path, gt_learning_map_inv_, "GT");
    }

  private:
    bool load_learning_map_inv_(const std::string& yaml_path,
                                std::map<int, int>& out_map,
                                const std::string& label) {
      try {
        YAML::Node cfg = YAML::LoadFile(yaml_path);
        if (!cfg["learning_map_inv"]) {
          RCLCPP_ERROR_STREAM(node_->get_logger(),
              "No 'learning_map_inv' key in " << yaml_path);
          return false;
        }
        out_map.clear();
        for (auto it = cfg["learning_map_inv"].begin();
             it != cfg["learning_map_inv"].end(); ++it) {
          int class_idx = it->first.as<int>();
          int label_id  = it->second.as<int>();
          out_map[class_idx] = label_id;
        }
        RCLCPP_INFO_STREAM(node_->get_logger(),
            "Loaded " << label << " learning_map_inv with " << out_map.size()
            << " entries from " << yaml_path);
        return true;
      } catch (const std::exception& e) {
        RCLCPP_ERROR_STREAM(node_->get_logger(),
            "Failed to load label config: " << e.what());
        return false;
      }
    }

  public:

    /// Load common taxonomy mappings from labels_common.yaml.
    /// @param yaml_path       Path to labels_common.yaml.
    /// @param inferred_key    "mcd", "semkitti", or "kitti360" — picks <key>_to_common for inferred labels.
    /// @param gt_key          "mcd", "semkitti", or "kitti360" — picks <key>_to_common for GT labels.
    bool load_common_label_config(const std::string& yaml_path,
                                  const std::string& inferred_key,
                                  const std::string& gt_key) {
      try {
        YAML::Node cfg = YAML::LoadFile(yaml_path);
        std::string inf_map_key = inferred_key + "_to_common";
        std::string gt_map_key  = gt_key + "_to_common";

        if (!cfg[inf_map_key]) {
          RCLCPP_ERROR_STREAM(node_->get_logger(),
              "No '" << inf_map_key << "' key in " << yaml_path);
          return false;
        }
        if (!cfg[gt_map_key]) {
          RCLCPP_ERROR_STREAM(node_->get_logger(),
              "No '" << gt_map_key << "' key in " << yaml_path);
          return false;
        }

        inferred_to_common_.clear();
        for (auto it = cfg[inf_map_key].begin(); it != cfg[inf_map_key].end(); ++it)
          inferred_to_common_[it->first.as<int>()] = it->second.as<int>();

        gt_to_common_.clear();
        for (auto it = cfg[gt_map_key].begin(); it != cfg[gt_map_key].end(); ++it)
          gt_to_common_[it->first.as<int>()] = it->second.as<int>();

        common_label_config_loaded_ = true;
        RCLCPP_INFO_STREAM(node_->get_logger(),
            "Loaded common label config from " << yaml_path
            << ": inferred mapping '" << inf_map_key << "' (" << inferred_to_common_.size()
            << " entries), GT mapping '" << gt_map_key << "' (" << gt_to_common_.size()
            << " entries)");
        return true;
      } catch (const std::exception& e) {
        RCLCPP_ERROR_STREAM(node_->get_logger(),
            "Failed to load common label config: " << e.what());
        return false;
      }
    }

    void set_uncertainty_filter(bool enabled, const std::string& labels_key,
                               const std::string& mode = "confusion_matrix",
                               float drop_percent = 10.0f,
                               float min_weight = 0.1f) {
      use_uncertainty_filter_ = enabled;
      inferred_labels_key_ = labels_key;
      uncertainty_filter_mode_ = mode;
      uncertainty_drop_percent_ = drop_percent;
      uncertainty_min_weight_ = min_weight;
      if (enabled) {
        RCLCPP_INFO_STREAM(node_->get_logger(),
            "Uncertainty filtering enabled (mode=" << mode
            << ", labels_key=" << labels_key
            << (mode == "top_percent"
                ? ", drop_percent=" + std::to_string(drop_percent)
                  + ", min_weight=" + std::to_string(min_weight)
                : "")
            << ")");
      }
    }

    /// Load a pre-computed confusion matrix from YAML (rows=predicted, cols=true).
    /// Computes per-class precision and stores it for filtering.
    bool load_confusion_matrix(const std::string& yaml_path) {
      try {
        YAML::Node root = YAML::LoadFile(yaml_path);
        if (!root["confusion_matrix"]) {
          RCLCPP_ERROR_STREAM(node_->get_logger(),
              "No 'confusion_matrix' key in " << yaml_path);
          return false;
        }
        std::memset(confusion_matrix_, 0, sizeof(confusion_matrix_));
        auto cm = root["confusion_matrix"];
        for (auto it = cm.begin(); it != cm.end(); ++it) {
          int pred_cls = it->first.as<int>();
          if (pred_cls < 0 || pred_cls >= N_COMMON) continue;
          auto row = it->second;
          // Columns are classes 1..12, stored as a sequence of 12 values
          if (row.size() != 12) {
            RCLCPP_WARN_STREAM(node_->get_logger(),
                "Confusion matrix row " << pred_cls << " has " << row.size()
                << " columns (expected 12), skipping");
            continue;
          }
          for (int c = 0; c < 12; ++c) {
            confusion_matrix_[pred_cls][c + 1] = row[c].as<int>();
          }
        }

        // Compute per-class precision
        RCLCPP_INFO_STREAM(node_->get_logger(),
            "Loaded confusion matrix from " << yaml_path);
        RCLCPP_INFO_STREAM(node_->get_logger(), "Per-class precision:");
        for (int cls = 1; cls < N_COMMON; ++cls) {
          int row_total = 0;
          for (int c = 0; c < N_COMMON; ++c) row_total += confusion_matrix_[cls][c];
          float prec = (row_total > 0)
              ? static_cast<float>(confusion_matrix_[cls][cls]) / row_total
              : 0.0f;
          class_precision_[cls] = prec;
          RCLCPP_INFO_STREAM(node_->get_logger(),
              "  class " << cls << ": precision=" << (prec * 100.0f) << "%");
        }
        confusion_matrix_loaded_ = true;
        return true;
      } catch (const std::exception& e) {
        RCLCPP_ERROR_STREAM(node_->get_logger(),
            "Failed to load confusion matrix: " << e.what());
        return false;
      }
    }

    /// Set map visualization color mode: semantic class or OSM prior (building/road/grassland/tree).
    void set_color_mode(semantic_bki::MapColorMode mode) {
      if (m_pub_) m_pub_->set_color_mode(mode);
    }

    /// Enable variance visualization on a separate topic. Jet colormap: blue=low variance, red=high.
    void set_publish_variance(bool enabled, const std::string& topic) {
      publish_variance_ = enabled;
      variance_topic_ = topic;
      if (enabled && !variance_pub_) {
        variance_pub_ = new semantic_bki::MarkerArrayPub(node_, topic, static_cast<float>(resolution_));
        RCLCPP_INFO_STREAM(node_->get_logger(), "Variance visualization enabled on topic: " << topic);
      }
    }

    /// Enable semantic uncertainty (input observation) map visualization. Voxel map with jet colormap: blue=confident, red=uncertain.
    /// Same pattern as variance map. Requires inferred_use_multiclass.
    void set_publish_semantic_uncertainty(bool enabled, const std::string& topic) {
      publish_semantic_uncertainty_ = enabled;
      semantic_uncertainty_topic_ = topic;
      if (enabled && !semantic_uncertainty_pub_) {
        semantic_uncertainty_pub_ = new semantic_bki::MarkerArrayPub(node_, topic, static_cast<float>(resolution_));
        RCLCPP_INFO_STREAM(node_->get_logger(), "Semantic uncertainty map visualization enabled on topic: " << topic);
      }
    }

    void set_osm_buildings(const std::vector<semantic_bki::Geometry2D> &buildings) {
      if (map_) map_->set_osm_buildings(buildings);
    }
    void set_osm_roads(const std::vector<semantic_bki::Geometry2D> &roads) {
      if (map_) map_->set_osm_roads(roads);
    }
    void set_osm_sidewalks(const std::vector<semantic_bki::Geometry2D> &sidewalks) {
      if (map_) map_->set_osm_sidewalks(sidewalks);
    }
    void set_osm_cycleways(const std::vector<semantic_bki::Geometry2D> &cycleways) {
      if (map_) map_->set_osm_cycleways(cycleways);
    }
    void set_osm_grasslands(const std::vector<semantic_bki::Geometry2D> &grasslands) {
      if (map_) map_->set_osm_grasslands(grasslands);
    }
    void set_osm_trees(const std::vector<semantic_bki::Geometry2D> &trees) {
      if (map_) map_->set_osm_trees(trees);
    }
    void set_osm_forests(const std::vector<semantic_bki::Geometry2D> &forests) {
      if (map_) map_->set_osm_forests(forests);
    }
    void set_osm_tree_points(const std::vector<std::pair<float, float>> &tree_points) {
      if (map_) map_->set_osm_tree_points(tree_points);
    }

    void set_osm_tree_point_radius(float radius_m) {
      if (map_) map_->set_osm_tree_point_radius(radius_m);
    }

    void set_osm_parking(const std::vector<semantic_bki::Geometry2D> &parking) {
      if (map_) map_->set_osm_parking(parking);
    }

    void set_osm_fences(const std::vector<semantic_bki::Geometry2D> &fences) {
      if (map_) map_->set_osm_fences(fences);
    }
    void set_osm_walls(const std::vector<semantic_bki::Geometry2D> &walls) {
      if (map_) map_->set_osm_walls(walls);
    }
    void set_osm_stairs(const std::vector<semantic_bki::Geometry2D> &stairs) {
      if (map_) map_->set_osm_stairs(stairs);
    }
    void set_osm_water(const std::vector<semantic_bki::Geometry2D> &water) {
      if (map_) map_->set_osm_water(water);
    }
    void set_osm_pole_points(const std::vector<std::pair<float, float>> &pole_points) {
      if (map_) map_->set_osm_pole_points(pole_points);
    }
    void set_osm_stairs_width(float width_m) {
      if (map_) map_->set_osm_stairs_width(width_m);
    }
    void set_osm_decay_meters(float decay_m) {
      if (map_) map_->set_osm_decay_meters(decay_m);
    }
    void set_osm_prior_strength(float strength) {
      if (map_) map_->set_osm_prior_strength(strength);
    }

    void set_osm_height_filter_enabled(bool enabled) {
      if (map_) map_->set_osm_height_filter_enabled(enabled);
    }

    /// Enable OSM prior map on a separate topic. Priors computed on-the-fly (not stored in octree).
    void set_publish_osm_prior_map(bool enabled, const std::string& topic, semantic_bki::MapColorMode osm_color_mode) {
      publish_osm_prior_map_ = enabled;
      osm_prior_map_color_mode_ = osm_color_mode;
      if (enabled && !osm_prior_map_pub_) {
        osm_prior_map_pub_ = new semantic_bki::MarkerArrayPub(node_, topic, static_cast<float>(resolution_));
        RCLCPP_INFO_STREAM(node_->get_logger(), "OSM prior map visualization enabled on topic: " << topic);
      }
    }

    bool load_osm_confusion_matrix(const std::string &yaml_path) {
      if (!map_) return false;
      try {
        YAML::Node root = YAML::LoadFile(yaml_path);
        if (!root["confusion_matrix"] || !root["label_to_matrix_idx"])
          return false;

        auto cm_node = root["confusion_matrix"];
        auto lm_node = root["label_to_matrix_idx"];

        // Determine row count from label_to_matrix_idx
        int max_row = -1;
        for (auto it = lm_node.begin(); it != lm_node.end(); ++it)
          max_row = std::max(max_row, it->second.as<int>());
        int n_rows = max_row + 1;

        // Parse confusion matrix rows (keyed by common class ID)
        // 14 columns: [roads, sidewalks, cycleways, parking, grasslands, trees, forest, buildings, fences, walls, stairs, water, poles, none]
        static constexpr int N_OSM_COLS = 14;
        std::vector<std::vector<float>> matrix(n_rows, std::vector<float>(N_OSM_COLS, 0.f));
        for (auto it = cm_node.begin(); it != cm_node.end(); ++it) {
          int common_class = it->first.as<int>();
          int row = -1;
          if (lm_node[common_class]) row = lm_node[common_class].as<int>();
          if (row < 0 || row >= n_rows) continue;
          auto vals = it->second;
          for (int c = 0; c < std::min(static_cast<int>(vals.size()), N_OSM_COLS); ++c)
            matrix[row][c] = vals[c].as<float>();
        }

        // Labels in ybars are now common taxonomy indices, so each confusion
        // matrix row maps directly to its common class ID.
        std::vector<std::vector<int>> row_to_labels(n_rows);
        for (auto it = lm_node.begin(); it != lm_node.end(); ++it) {
          int common_class = it->first.as<int>();
          int row = it->second.as<int>();
          if (common_class > 0 && row >= 0 && row < n_rows)
            row_to_labels[row].push_back(common_class);
        }

        map_->set_osm_confusion_matrix(matrix, row_to_labels);
        RCLCPP_INFO_STREAM(node_->get_logger(),
            "OSM confusion matrix loaded: " << n_rows << " rows x " << N_OSM_COLS << " cols");
        return true;
      } catch (const std::exception &e) {
        RCLCPP_WARN_STREAM(node_->get_logger(),
            "Failed to load OSM confusion matrix: " << e.what());
        return false;
      }
    }

    /// Load OSM height confusion matrix (rows=height bins, cols=OSM categories). Optional; used when osm_height_filtering enabled.
    bool load_osm_height_confusion_matrix(const std::string &yaml_path) {
      if (!map_) return false;
      try {
        YAML::Node root = YAML::LoadFile(yaml_path);
        if (!root["osm_height_confusion_matrix"]) return false;

        auto cm_node = root["osm_height_confusion_matrix"];
        static constexpr int N_OSM_COLS = 14;
        int max_bin = 0;
        for (auto it = cm_node.begin(); it != cm_node.end(); ++it)
          max_bin = std::max(max_bin, it->first.as<int>());
        std::vector<std::vector<float>> matrix(max_bin, std::vector<float>(N_OSM_COLS, 0.f));
        for (auto it = cm_node.begin(); it != cm_node.end(); ++it) {
          int bin_idx = it->first.as<int>();
          if (bin_idx < 1 || bin_idx > max_bin) continue;
          auto vals = it->second;
          for (int c = 0; c < std::min(static_cast<int>(vals.size()), N_OSM_COLS); ++c)
            matrix[bin_idx - 1][c] = vals[c].as<float>();
        }
        if (matrix.empty()) return false;
        map_->set_osm_height_confusion_matrix(matrix);
        RCLCPP_INFO_STREAM(node_->get_logger(),
            "OSM height confusion matrix loaded: " << matrix.size() << " bins x " << N_OSM_COLS << " cols");
        return true;
      } catch (const std::exception &e) {
        RCLCPP_WARN_STREAM(node_->get_logger(),
            "Failed to load OSM height confusion matrix: " << e.what());
        return false;
      }
    }

    /// Return true if both the lidar bin and label/multiclass file exist for the given scan file number.
    bool scan_and_label_exist(const std::string& input_data_dir, const std::string& input_label_dir, int scan_file_num) {
      char scan_id_c[256];
      std::snprintf(scan_id_c, sizeof(scan_id_c), "%010d", scan_file_num);
      std::string scan_name = input_data_dir + "/" + std::string(scan_id_c) + ".bin";
      FILE* fp = std::fopen(scan_name.c_str(), "rb");
      if (!fp) return false;
      std::fclose(fp);

      std::string label_name;
      if (inferred_use_multiclass_) {
        label_name = inferred_multiclass_dir_ + "/" + std::string(scan_id_c) + ".bin";
      } else {
        label_name = input_label_dir + "/" + std::string(scan_id_c) + ".bin";
      }
      FILE* fp_label = std::fopen(label_name.c_str(), "rb");
      if (!fp_label) return false;
      std::fclose(fp_label);
      return true;
    }

    bool process_scans(std::string input_data_dir, std::string input_label_dir, int scan_num, int skip_frames, bool query, bool visualize) {
      if (!map_) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: process_scans: map_ is null!");
        return false;
      }
      if (!m_pub_) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: process_scans: m_pub_ is null!");
        return false;
      }
      if (lidar_poses_.empty()) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: process_scans: No poses loaded!");
        return false;
      }
      
      // Build list of pose indices where both lidar bin and label file exist (matching file names)
      std::vector<int> valid_pose_indices;
      valid_pose_indices.reserve(lidar_poses_.size());
      for (int pose_idx = 0; pose_idx < static_cast<int>(lidar_poses_.size()); ++pose_idx) {
        int scan_file_num = scan_indices_[pose_idx];
        if (scan_and_label_exist(input_data_dir, input_label_dir, scan_file_num))
          valid_pose_indices.push_back(pose_idx);
      }
      RCLCPP_INFO_STREAM(node_->get_logger(), "Found " << valid_pose_indices.size() << " scans with both lidar and label files (out of " << lidar_poses_.size() << " poses). Applying scan_num=" << scan_num << ", skip_frames=" << skip_frames);
      
      // Apply scan_num and skip_frames to the valid set: take every (skip_frames+1)-th valid scan, up to scan_num
      int valid_count = 0;
      std::vector<int> indices_to_process;
      for (size_t i = 0; i < valid_pose_indices.size(); ++i) {
        if (skip_frames > 0 && valid_count % (skip_frames + 1) != 0) {
          valid_count++;
          continue;
        }
        if (static_cast<int>(indices_to_process.size()) >= scan_num)
          break;
        indices_to_process.push_back(valid_pose_indices[i]);
        valid_count++;
      }
      
      semantic_bki::point3f origin;
      int insertion_count = 0;
      
      for (size_t list_idx = 0; list_idx < indices_to_process.size(); ++list_idx) {
        int pose_idx = indices_to_process[list_idx];
        int scan_file_num = scan_indices_[pose_idx];
        
        char scan_id_c[256];
        sprintf(scan_id_c, "%010d", scan_file_num);
        std::string scan_name = input_data_dir + "/" + std::string(scan_id_c) + ".bin";

        pcl::PointCloud<pcl::PointXYZL>::Ptr cloud;
        pcl::PointCloud<pcl::PointXYZL>::ConstPtr orig_cloud_for_unc;
        std::vector<float> orig_variances_copy;  // copy must outlive mc_result for accumulation
        int orig_n_classes_for_unc = 0;
        if (inferred_use_multiclass_) {
          std::string mc_name = inferred_multiclass_dir_ + "/" + std::string(scan_id_c) + ".bin";
          MulticlassResult mc_result = mcd2pcl_multiclass(scan_name, mc_name);
          cloud = mc_result.cloud;
          if (publish_semantic_uncertainty_ && mc_result.n_classes > 1 && !mc_result.variances.empty()) {
            orig_cloud_for_unc = mc_result.cloud;
            orig_variances_copy = mc_result.variances;  // copy: mc_result is destroyed at block end
            orig_n_classes_for_unc = mc_result.n_classes;
          }
          // Compute per-point weights from uncertainty for kernel discounting
          if (use_uncertainty_filter_ && cloud && !cloud->points.empty() &&
              mc_result.n_classes > 1) {

            float max_var = static_cast<float>(mc_result.n_classes - 1) /
                            (static_cast<float>(mc_result.n_classes) * mc_result.n_classes);
            size_t n_pts = cloud->points.size();

            std::vector<float> uncertainties(n_pts);
            for (size_t pi = 0; pi < n_pts; ++pi) {
              uncertainties[pi] = 1.0f - std::min(mc_result.variances[pi] / max_var, 1.0f);
            }

            if (uncertainty_filter_mode_ == "top_percent") {
              // Discount the top N% most uncertain points via kernel weights.
              // Points below the threshold get weight 1.0 (full influence).
              // Points above get weight ramping from 1.0 down to 0.0, so
              // confident neighbors can dominate through kernel smoothing.
              float keep_fraction = 1.0f - uncertainty_drop_percent_ / 100.0f;
              size_t keep_rank = static_cast<size_t>(keep_fraction * n_pts);
              if (keep_rank >= n_pts) keep_rank = n_pts - 1;

              std::vector<float> sorted_unc(uncertainties);
              std::nth_element(sorted_unc.begin(),
                               sorted_unc.begin() + keep_rank,
                               sorted_unc.end());
              float threshold = sorted_unc[keep_rank];

              scan_point_weights_.resize(n_pts);
              int n_discounted = 0;
              for (size_t pi = 0; pi < n_pts; ++pi) {
                if (uncertainties[pi] <= threshold) {
                  scan_point_weights_[pi] = 1.0f;
                } else {
                  // Linearly ramp from 1.0 at threshold to uncertainty_min_weight_ at uncertainty=1.0
                  float denom = (1.0f - threshold);
                  float t = (denom > 1e-6f)
                      ? std::min((uncertainties[pi] - threshold) / denom, 1.0f)
                      : 1.0f;
                  scan_point_weights_[pi] = 1.0f - t * (1.0f - uncertainty_min_weight_);
                  n_discounted++;
                }
              }

              total_points_processed_ += static_cast<int>(n_pts);
              total_points_filtered_ += n_discounted;
              if (n_discounted > 0 && (list_idx < 5 || list_idx % 50 == 0)) {
                RCLCPP_INFO_STREAM(node_->get_logger(),
                    "Scan " << list_idx << ": discounted " << n_discounted << "/"
                    << n_pts << " points (cumulative: "
                    << total_points_filtered_ << "/" << total_points_processed_ << ")");
              }

            } else if (confusion_matrix_loaded_) {
              // Per-class precision: filter (drop) points entirely.
              // Labels are already in common taxonomy after ingestion.
              pcl::PointCloud<pcl::PointXYZL>::Ptr filtered(new pcl::PointCloud<pcl::PointXYZL>);
              filtered->points.reserve(n_pts);
              int n_dropped = 0;

              for (size_t pi = 0; pi < n_pts; ++pi) {
                int pred_common = static_cast<int>(cloud->points[pi].label);
                float precision = (pred_common > 0 && pred_common < N_COMMON)
                    ? class_precision_[pred_common] : 1.0f;
                if (uncertainties[pi] <= precision || pred_common == 0) {
                  filtered->points.push_back(cloud->points[pi]);
                } else {
                  n_dropped++;
                }
              }

              if (n_dropped > 0) {
                total_points_processed_ += static_cast<int>(n_pts);
                total_points_filtered_ += n_dropped;
                if (list_idx < 5 || list_idx % 50 == 0) {
                  RCLCPP_INFO_STREAM(node_->get_logger(),
                      "Scan " << list_idx << ": filtered " << n_dropped << "/"
                      << n_pts << " points (cumulative: "
                      << total_points_filtered_ << "/" << total_points_processed_ << ")");
                }
              }

              filtered->width = static_cast<uint32_t>(filtered->points.size());
              filtered->height = 1;
              filtered->is_dense = false;
              cloud = filtered;
              scan_point_weights_.clear();
            }
          } else {
            scan_point_weights_.clear();
          }
        } else {
          std::string label_name = input_label_dir + "/" + std::string(scan_id_c) + ".bin";
          cloud = mcd2pcl(scan_name, label_name);
        }
        if (!cloud) {
          RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: mcd2pcl returned null pointer for scan " << scan_file_num);
          continue;
        }
        if (cloud->points.empty()) {
          RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Empty point cloud at scan file " << scan_file_num << " (pose index " << pose_idx << "), skipping");
          continue;
        }
  
        Eigen::Matrix4d transform = lidar_poses_[pose_idx];  // This is body_to_world from pose
        
        // Verify body-to-lidar transform is loaded (should not be zero matrix)
        if (body_to_lidar_tf_.isZero(1e-10)) {
          RCLCPP_FATAL_STREAM(node_->get_logger(), "ERROR: body_to_lidar_tf_ is not initialized! Calibration must be loaded before processing scans.");
          RCLCPP_FATAL_STREAM(node_->get_logger(), "Call load_calibration_from_params() and ensure it returns true.");
          exit(1);
        }
        
        // Apply body-to-lidar transformation
        // The poses in CSV are body/IMU poses, need to transform from lidar frame to world frame
        // Following Python code: transform_matrix = body_to_world @ lidar_to_body
        // where lidar_to_body = inv(body_to_lidar_tf)
        Eigen::Matrix4d lidar_to_body = body_to_lidar_tf_.inverse();
        Eigen::Matrix4d lidar_to_map = transform * lidar_to_body;  // T_lidar_to_map = body_to_world * lidar_to_body
        
        // Publish TF transform from 'map' to 'lidar' frame
        geometry_msgs::msg::TransformStamped t;
        t.header.stamp = node_->now();
        t.header.frame_id = "map";
        t.child_frame_id = "lidar";
        
        Eigen::Matrix3d rotation = lidar_to_map.block<3, 3>(0, 0);
        Eigen::Vector3d translation = lidar_to_map.block<3, 1>(0, 3);
        
        Eigen::Quaterniond quat(rotation);
        t.transform.translation.x = translation(0);
        t.transform.translation.y = translation(1);
        t.transform.translation.z = translation(2);
        t.transform.rotation.x = quat.x();
        t.transform.rotation.y = quat.y();
        t.transform.rotation.z = quat.z();
        t.transform.rotation.w = quat.w();
        
        tf_broadcaster_.sendTransform(t);
        
        // Debug: Print transform info for first few processed scans
        if (list_idx < 3) {
          RCLCPP_INFO_STREAM(node_->get_logger(), "Scan " << list_idx << " (pose_idx " << pose_idx << ") Transform info:");
          RCLCPP_INFO_STREAM(node_->get_logger(), "  Body-to-world translation from CSV: [" << transform(0,3) << ", " << transform(1,3) << ", " << transform(2,3) << "]");
          RCLCPP_INFO_STREAM(node_->get_logger(), "  Lidar-to-map translation: [" << lidar_to_map(0,3) << ", " << lidar_to_map(1,3) << ", " << lidar_to_map(2,3) << "]");
          RCLCPP_INFO_STREAM(node_->get_logger(), "  TF translation (from lidar_to_map): [" << translation(0) << ", " << translation(1) << ", " << translation(2) << "]");
          
          // Verify: transform origin from lidar frame should give lidar_to_map translation
          Eigen::Vector4d lidar_origin(0, 0, 0, 1);
          Eigen::Vector4d map_origin_test = lidar_to_map * lidar_origin;
          RCLCPP_INFO_STREAM(node_->get_logger(), "  Verification - lidar origin in map coords: [" << map_origin_test(0) << ", " << map_origin_test(1) << ", " << map_origin_test(2) << "]");
        }
        
        // Publish individual scan as PointCloud2 (in lidar frame, before transformation)
        sensor_msgs::msg::PointCloud2 cloud_msg;
        pcl::toROSMsg(*cloud, cloud_msg);
        cloud_msg.header.frame_id = "lidar";
        cloud_msg.header.stamp = node_->now();
        pointcloud_pub_->publish(cloud_msg);

        // Accumulate semantic uncertainty per voxel (for map visualization)
        if (orig_cloud_for_unc && !orig_variances_copy.empty() && publish_semantic_uncertainty_) {
          float max_var = static_cast<float>(orig_n_classes_for_unc - 1) /
              (static_cast<float>(orig_n_classes_for_unc) * orig_n_classes_for_unc);
          pcl::PointCloud<pcl::PointXYZL> cloud_map;
          pcl::transformPointCloud(*orig_cloud_for_unc, cloud_map, lidar_to_map.cast<float>());
          for (size_t pi = 0; pi < cloud_map.points.size(); ++pi) {
            float u = 1.0f - std::min(orig_variances_copy[pi] / max_var, 1.0f);
            int kx = static_cast<int>(std::floor(cloud_map.points[pi].x / resolution_));
            int ky = static_cast<int>(std::floor(cloud_map.points[pi].y / resolution_));
            int kz = static_cast<int>(std::floor(cloud_map.points[pi].z / resolution_));
            VoxelKey key{kx, ky, kz};
            auto& p = semantic_uncertainty_acc_[key];
            p.first += u;
            p.second += 1;
          }
        }

        if (insertion_count == 0) {
          RCLCPP_INFO_STREAM(node_->get_logger(), "Published PointCloud2 with " << cloud->points.size() << " points in frame 'lidar'");
        }
        
        // Now transform cloud to world frame for map insertion
        pcl::transformPointCloud(*cloud, *cloud, lidar_to_map);
        
        origin.x() = transform(0, 3);
        origin.y() = transform(1, 3);
        origin.z() = transform(2, 3);
        
        try {
          if (!scan_point_weights_.empty() && scan_point_weights_.size() == cloud->points.size()) {
            map_->insert_pointcloud(*cloud, origin, ds_resolution_, free_resolution_, max_range_, scan_point_weights_);
          } else {
            map_->insert_pointcloud(*cloud, origin, ds_resolution_, free_resolution_, max_range_);
          }
        } catch (const std::exception& e) {
          RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Exception during insert_pointcloud: " << e.what());
          continue;
        }
        insertion_count++;
        
        // Skip query/visualize for first insertion to avoid potential segfaults with empty/initializing octree
        if (insertion_count == 1) {
          RCLCPP_DEBUG_STREAM(node_->get_logger(), "Skipping query/visualize for first insertion");
          continue;
        }
        
        if (query) {
          // Query previous scans (use pose indices, not file numbers)
          // Original ROS1 logic: for (int query_pose_idx = pose_idx - 10; query_pose_idx >= 0 && query_pose_idx <= pose_idx; ++query_pose_idx)
          for (int query_pose_idx = pose_idx - 10; query_pose_idx >= 0 && query_pose_idx <= pose_idx; ++query_pose_idx) {
            query_scan(input_data_dir, input_label_dir, query_pose_idx);
          }
        }

        if (visualize) {
          publish_map();
          // Small delay to allow rviz to process the visualization
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
      }
      
      // Final publish after all scans are processed
      if (visualize) {
        RCLCPP_INFO_STREAM(node_->get_logger(), "All scans processed. Publishing final map visualization...");
        publish_map();
      }
      
      return true;
    }

    void publish_map() {
      if (!m_pub_) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: publish_map: m_pub_ is null!");
        return;
      }
      if (!map_) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: publish_map: map_ is null!");
        return;
      }

      m_pub_->clear_map(resolution_);
      
      // Check if map is empty before iterating - get iterators separately to catch segfault location
      try {
        auto begin_it = map_->begin_leaf();
        
        auto end_it = map_->end_leaf();
        
        if (begin_it == end_it) {
          RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Map is empty (begin == end), nothing to publish");
          m_pub_->publish();
          return;
        }
      } catch (const std::exception& e) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Exception getting iterators: " << e.what());
        return;
      } catch (...) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Unknown exception getting iterators");
        return;
      }

      // First pass: main semantic map (always semantic colors) + compute min/max variance for variance visualization
      float min_var = std::numeric_limits<float>::max();
      float max_var = std::numeric_limits<float>::lowest();
      int voxel_count = 0;
      int iter_count = 0;
      try {
        auto loop_begin = map_->begin_leaf();
        auto loop_end = map_->end_leaf();
        
        auto it = loop_begin;
        
        while (it != loop_end) {
          iter_count++;
          
          try {
            auto node = it.get_node();
            
              if (node.get_state() == semantic_bki::State::OCCUPIED) {
              semantic_bki::point3f p = it.get_loc();
              float size = it.get_size();
              if (publish_variance_) {
                std::vector<float> vars(num_class_);
                node.get_vars(vars);
                int semantics = node.get_semantics();
                if (semantics >= 0 && semantics < num_class_) {
                  float v = vars[semantics];
                  if (v > max_var) max_var = v;
                  if (v < min_var) min_var = v;
                }
              }
              m_pub_->insert_point3d_semantics(p.x(), p.y(), p.z(), size, node.get_semantics(), 2);
              voxel_count++;
            }
            
            if (iter_count % 1000 == 0) {
              RCLCPP_DEBUG_STREAM(node_->get_logger(), "Processed " << iter_count << " nodes");
            }
            
            // Increment iterator
            ++it;
          } catch (const std::exception& e) {
            RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Exception at iteration " << iter_count << ": " << e.what());
            break;
          } catch (...) {
            RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Unknown exception at iteration " << iter_count);
            break;
          }
        }
      } catch (const std::exception& e) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Exception during map iteration: " << e.what());
        return;
      }
      m_pub_->publish();

      // Second pass: publish variance map if enabled
      if (publish_variance_ && variance_pub_) {
        if (min_var > max_var) min_var = max_var = 0.f;  // avoid div by zero
        variance_pub_->clear_map(static_cast<float>(resolution_));
        for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
          auto node = it.get_node();
          if (node.get_state() == semantic_bki::State::OCCUPIED) {
            semantic_bki::point3f p = it.get_loc();
            int semantics = node.get_semantics();
            std::vector<float> vars(num_class_);
            node.get_vars(vars);
            if (semantics >= 0 && semantics < num_class_)
              variance_pub_->insert_point3d_variance(p.x(), p.y(), p.z(), min_var, max_var, it.get_size(), vars[semantics]);
          }
        }
        variance_pub_->publish();
      }

      // OSM prior map pass: publish OSM priors on separate topic (computed on-the-fly, not stored in octree)
      if (publish_osm_prior_map_ && osm_prior_map_pub_ && map_) {
        osm_prior_map_pub_->clear_map(static_cast<float>(resolution_));
        osm_prior_map_pub_->set_color_mode(osm_prior_map_color_mode_);
        semantic_bki::MapColorMode mode = osm_prior_map_color_mode_;
        for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
          auto node = it.get_node();
          if (node.get_state() != semantic_bki::State::OCCUPIED) continue;
          semantic_bki::point3f p = it.get_loc();
          float size = it.get_size();
          float building, road, grassland, tree, parking, fence, stairs;
          map_->get_osm_priors_for_visualization(p.x(), p.y(), building, road, grassland, tree, parking, fence, stairs);
          if (mode == semantic_bki::MapColorMode::OSMBlend) {
            osm_prior_map_pub_->insert_point3d_osm_blend(p.x(), p.y(), p.z(), size,
                building, road, grassland, tree, parking, fence, stairs);
          } else {
            int prior_type = 0;
            float value = 0.f;
            switch (mode) {
              case semantic_bki::MapColorMode::OSMBuilding:   prior_type = 0; value = building; break;
              case semantic_bki::MapColorMode::OSMRoad:      prior_type = 1; value = road; break;
              case semantic_bki::MapColorMode::OSMGrassland:  prior_type = 2; value = grassland; break;
              case semantic_bki::MapColorMode::OSMTree:     prior_type = 3; value = tree; break;
              case semantic_bki::MapColorMode::OSMParking:  prior_type = 4; value = parking; break;
              case semantic_bki::MapColorMode::OSMFence:    prior_type = 5; value = fence; break;
              case semantic_bki::MapColorMode::OSStairs:    prior_type = 6; value = stairs; break;
              default: prior_type = 0; value = building; break;
            }
            osm_prior_map_pub_->insert_point3d_osm_prior(p.x(), p.y(), p.z(), size, value, prior_type);
          }
        }
        osm_prior_map_pub_->publish();
      }

      // Fourth pass: publish semantic uncertainty map if enabled (input observation uncertainty per voxel)
      if (publish_semantic_uncertainty_ && semantic_uncertainty_pub_ && !semantic_uncertainty_acc_.empty()) {
        float min_unc = 1.f;
        float max_unc = 0.f;
        for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
          auto node = it.get_node();
          if (node.get_state() != semantic_bki::State::OCCUPIED) continue;
          semantic_bki::point3f p = it.get_loc();
          int kx = static_cast<int>(std::floor(p.x() / resolution_));
          int ky = static_cast<int>(std::floor(p.y() / resolution_));
          int kz = static_cast<int>(std::floor(p.z() / resolution_));
          VoxelKey key{kx, ky, kz};
          auto fit = semantic_uncertainty_acc_.find(key);
          if (fit == semantic_uncertainty_acc_.end()) continue;
          float avg_u = fit->second.first / static_cast<float>(fit->second.second);
          if (avg_u < min_unc) min_unc = avg_u;
          if (avg_u > max_unc) max_unc = avg_u;
        }
        if (min_unc > max_unc) min_unc = max_unc = 0.f;
        semantic_uncertainty_pub_->clear_map(static_cast<float>(resolution_));
        for (auto it = map_->begin_leaf(); it != map_->end_leaf(); ++it) {
          auto node = it.get_node();
          if (node.get_state() != semantic_bki::State::OCCUPIED) continue;
          semantic_bki::point3f p = it.get_loc();
          int kx = static_cast<int>(std::floor(p.x() / resolution_));
          int ky = static_cast<int>(std::floor(p.y() / resolution_));
          int kz = static_cast<int>(std::floor(p.z() / resolution_));
          VoxelKey key{kx, ky, kz};
          auto fit = semantic_uncertainty_acc_.find(key);
          if (fit == semantic_uncertainty_acc_.end()) continue;
          float avg_u = fit->second.first / static_cast<float>(fit->second.second);
          semantic_uncertainty_pub_->insert_point3d_variance(p.x(), p.y(), p.z(), min_unc, max_unc, it.get_size(), avg_u);
        }
        semantic_uncertainty_pub_->publish();
      }
    }
    
    // Load colors from ROS parameters
    bool load_colors_from_params() {
      if (m_pub_) {
        return m_pub_->load_colors_from_params(node_);
      }
      return false;
    }
    
    // Load colors directly from YAML file
    bool load_colors_from_yaml(const std::string& yaml_file_path) {
      RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: MCDData::load_colors_from_yaml: Starting, file=" << yaml_file_path);
      if (!m_pub_) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: m_pub_ is null, cannot load colors!");
        return false;
      }
      RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: m_pub_ is valid, calling load_colors_from_yaml");
      bool result = m_pub_->load_colors_from_yaml(yaml_file_path);
      RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: MCDData::load_colors_from_yaml: Result=" << result);
      return result;
    }

    void set_up_evaluation(const std::string gt_label_dir, const std::string evaluation_result_dir) {
      gt_label_dir_ = gt_label_dir;
      evaluation_result_dir_ = evaluation_result_dir;
    }

    bool load_calibration_from_params() {
      // Load body-to-lidar transform from ROS parameters
      // Expected path: body/os_sensor/T (from hhs_calib.yaml)
      // Format: T is a list of 4 lists, each containing 4 doubles
      
      RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: load_calibration_from_params: Starting");
      try {
        // First, try to load from calibration_file parameter (YAML file path)
        std::string calib_file;
        if (node_->get_parameter("calibration_file", calib_file)) {
          RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Found calibration_file parameter: " << calib_file);
          RCLCPP_INFO_STREAM(node_->get_logger(), "Loading calibration from YAML file: " << calib_file);
          bool result = load_calibration_from_yaml(calib_file);
          RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: load_calibration_from_yaml returned: " << result);
          return result;
        }
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: calibration_file parameter not found");
        
        // Fallback: try to get body parameter directly from ROS parameters
        if (node_->has_parameter("body")) {
          RCLCPP_WARN_STREAM(node_->get_logger(), "Calibration loading from ROS parameters not fully implemented. Please use calibration_file parameter.");
          return false;
        }
        
        RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: 'calibration_file' parameter not found in ROS parameter server!");
        RCLCPP_ERROR_STREAM(node_->get_logger(), "Make sure calibration_file parameter is set in the launch file pointing to hhs_calib.yaml.");
        return false;
      } catch (const std::exception& e) {
        RCLCPP_ERROR_STREAM(node_->get_logger(), "Error loading calibration: " << e.what());
        return false;
      }
    }
    
    bool load_calibration_from_yaml(const std::string& yaml_file) {
      RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: load_calibration_from_yaml: Loading file: " << yaml_file);
      try {
        YAML::Node yaml_node = YAML::LoadFile(yaml_file);
        RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: YAML file loaded successfully");
        if (!yaml_node["body"]) {
          RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: 'body' key not found in YAML file!");
          RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: 'body' key not found in YAML file!");
          return false;
        }
        RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Found 'body' key in YAML");
        
        YAML::Node body_node = yaml_node["body"];
        if (!body_node["os_sensor"]) {
          RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: 'body/os_sensor' not found in calibration!");
          return false;
        }
        
        YAML::Node os_sensor_node = body_node["os_sensor"];
        if (!os_sensor_node["T"]) {
          RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: 'body/os_sensor/T' not found in calibration!");
          return false;
        }
        
        YAML::Node T_node = os_sensor_node["T"];
        if (!T_node.IsSequence() || T_node.size() != 4) {
          RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: 'body/os_sensor/T' must be an array of 4 arrays!");
          return false;
        }
        
        // Parse the 4x4 matrix
        for (int i = 0; i < 4; ++i) {
          if (!T_node[i].IsSequence() || T_node[i].size() != 4) {
            RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: Row " << i << " of body/os_sensor/T must be an array of 4 elements!");
            return false;
          }
          for (int j = 0; j < 4; ++j) {
            body_to_lidar_tf_(i, j) = T_node[i][j].as<double>();
          }
        }
        
        RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Successfully parsed calibration transform matrix");
        RCLCPP_INFO_STREAM(node_->get_logger(), "Successfully loaded body-to-lidar transform from body/os_sensor/T");
        RCLCPP_INFO_STREAM(node_->get_logger(), "Transform matrix:");
        RCLCPP_INFO_STREAM(node_->get_logger(), "  [" << body_to_lidar_tf_(0, 0) << ", " << body_to_lidar_tf_(0, 1) << ", " << body_to_lidar_tf_(0, 2) << ", " << body_to_lidar_tf_(0, 3) << "]");
        RCLCPP_INFO_STREAM(node_->get_logger(), "  [" << body_to_lidar_tf_(1, 0) << ", " << body_to_lidar_tf_(1, 1) << ", " << body_to_lidar_tf_(1, 2) << ", " << body_to_lidar_tf_(1, 3) << "]");
        RCLCPP_INFO_STREAM(node_->get_logger(), "  [" << body_to_lidar_tf_(2, 0) << ", " << body_to_lidar_tf_(2, 1) << ", " << body_to_lidar_tf_(2, 2) << ", " << body_to_lidar_tf_(2, 3) << "]");
        RCLCPP_INFO_STREAM(node_->get_logger(), "  [" << body_to_lidar_tf_(3, 0) << ", " << body_to_lidar_tf_(3, 1) << ", " << body_to_lidar_tf_(3, 2) << ", " << body_to_lidar_tf_(3, 3) << "]");
        
        RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: load_calibration_from_yaml completed successfully");
        return true;
      } catch (const std::exception& e) {
        RCLCPP_ERROR_STREAM(node_->get_logger(), "Error loading calibration from YAML: " << e.what());
        return false;
      }
    }

    void query_scan(std::string input_data_dir, std::string /* input_label_dir */, int pose_idx) {
      if (pose_idx < 0 || pose_idx >= (int)lidar_poses_.size()) {
        return;
      }
      
      if (!map_) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "Cannot query scan: map_ is null");
        return;
      }

      try {
        // Get the actual scan file number from CSV
        int scan_file_num = scan_indices_[pose_idx];
        
        // Use 10-digit format for MCD file naming
        char scan_id_c[256];
        sprintf(scan_id_c, "%010d", scan_file_num);
        // Helper function to join paths (handles trailing slashes)
        auto join_path = [](const std::string& dir, const std::string& file) -> std::string {
          if (dir.empty()) return file;
          if (dir.back() == '/') return dir + file;
          return dir + "/" + file;
        };
        std::string scan_name = join_path(input_data_dir, std::string(scan_id_c) + ".bin");
        std::string gt_name = join_path(gt_label_dir_, std::string(scan_id_c) + ".bin");
        std::string result_name = join_path(evaluation_result_dir_, std::string(scan_id_c) + ".txt");

        pcl::PointCloud<pcl::PointXYZL>::Ptr cloud;
        if (gt_use_multiclass_)
          cloud = mcd2pcl_gt_multiclass(scan_name, gt_name);
        else
          cloud = mcd2pcl(scan_name, gt_name, /*use_gt_mapping=*/true);
        if (cloud->points.empty()) {
          return;
        }

        Eigen::Matrix4d transform = lidar_poses_[pose_idx];  // This is body_to_world from pose
        
        // Apply body-to-lidar transformation (same as in process_scans)
        // Following Python code: transform_matrix = body_to_world @ lidar_to_body
        Eigen::Matrix4d lidar_to_body = body_to_lidar_tf_.inverse();
        Eigen::Matrix4d new_transform = transform * lidar_to_body;  // body_to_world * lidar_to_body
        pcl::transformPointCloud(*cloud, *cloud, new_transform);

        // Create directory if it doesn't exist
        size_t last_slash = result_name.find_last_of('/');
        if (last_slash != std::string::npos) {
          std::string dir_path = result_name.substr(0, last_slash);
          // Try to create directory (mkdir -p equivalent)
          std::string mkdir_cmd = "mkdir -p " + dir_path;
          int result = std::system(mkdir_cmd.c_str());
          if (result != 0) {
            RCLCPP_WARN_STREAM(node_->get_logger(), "Failed to create directory: " << dir_path);
          }
        }
        
        std::ofstream result_file;
        result_file.open(result_name);
        if (!result_file.is_open()) {
          RCLCPP_WARN_STREAM(node_->get_logger(), "Cannot open result file: " << result_name);
          return;
        }
        
        for (int i = 0; i < (int)cloud->points.size(); ++i) {
          try {
            semantic_bki::SemanticOcTreeNode node = map_->search(cloud->points[i].x, cloud->points[i].y, cloud->points[i].z);
            int pred_label = 0;
            if (node.get_state() == semantic_bki::State::OCCUPIED)
              pred_label = node.get_semantics();
            result_file << (int)cloud->points[i].label << " " << pred_label << "\n";
          } catch (const std::exception& e) {
            RCLCPP_WARN_STREAM(node_->get_logger(), "Exception searching map for point " << i << ": " << e.what());
            continue;
          }
        }
        result_file.close();
      } catch (const std::exception& e) {
        RCLCPP_ERROR_STREAM(node_->get_logger(), "Exception in query_scan(): " << e.what());
      } catch (...) {
        RCLCPP_ERROR_STREAM(node_->get_logger(), "Unknown exception in query_scan()");
      }
    }

  
  private:
    rclcpp::Node::SharedPtr node_;
    double resolution_;
    int num_class_;
    double ds_resolution_;
    double free_resolution_;
    double max_range_;
    semantic_bki::SemanticBKIOctoMap* map_;
    semantic_bki::MarkerArrayPub* m_pub_;
    semantic_bki::MarkerArrayPub* variance_pub_{nullptr};
    bool publish_variance_{false};
    semantic_bki::MarkerArrayPub* osm_prior_map_pub_{nullptr};
    bool publish_osm_prior_map_{false};
    semantic_bki::MapColorMode osm_prior_map_color_mode_{semantic_bki::MapColorMode::OSMBlend};
    std::string variance_topic_;
    semantic_bki::MarkerArrayPub* semantic_uncertainty_pub_{nullptr};
    bool publish_semantic_uncertainty_{false};
    std::string semantic_uncertainty_topic_;
    std::map<VoxelKey, std::pair<float, int>> semantic_uncertainty_acc_;  // per-voxel: (sum_uncertainty, count)
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pointcloud_pub_;  // Publisher for individual scan point clouds
    tf2_ros::TransformBroadcaster tf_broadcaster_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::ofstream pose_file_;
    std::vector<Eigen::Matrix4d> lidar_poses_;
    std::vector<int> scan_indices_;  // Maps pose index to actual scan file number (from CSV "num" column)
    std::string gt_label_dir_;
    std::string evaluation_result_dir_;
    Eigen::Matrix4d init_trans_to_ground_;
    Eigen::Matrix4d body_to_lidar_tf_;
    Eigen::Matrix4d original_first_pose_;

    // Multiclass settings (inferred and GT)
    bool inferred_use_multiclass_;
    std::string inferred_multiclass_dir_;
    std::map<int, int> learning_map_inv_;   // for inferred: model output index → raw label
    bool gt_use_multiclass_;
    std::map<int, int> gt_learning_map_inv_;  // for GT: model output index → raw label

    // Common taxonomy mappings (loaded from labels_common.yaml)
    bool common_label_config_loaded_;
    std::map<int, int> inferred_to_common_;  // raw inferred label → common class index
    std::map<int, int> gt_to_common_;        // raw GT label → common class index

    // Uncertainty filtering
    bool use_uncertainty_filter_;
    bool confusion_matrix_loaded_;
    std::string inferred_labels_key_;  // "mcd", "semkitti", or "kitti360"
    std::string uncertainty_filter_mode_;  // "confusion_matrix" or "top_percent"
    float uncertainty_drop_percent_;       // for top_percent mode: discount this % of most uncertain points
    float uncertainty_min_weight_;         // minimum kernel weight for the most uncertain points
    int confusion_matrix_[N_COMMON][N_COMMON];
    float class_precision_[N_COMMON];
    int total_points_processed_;
    int total_points_filtered_;
    std::vector<float> scan_point_weights_;

    pcl::PointCloud<pcl::PointXYZL>::Ptr mcd2pcl(std::string fn, std::string fn_label, bool use_gt_mapping = false) {
      // Open scan file
      FILE* fp = std::fopen(fn.c_str(), "rb");
      if (!fp) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Cannot open scan file: " << fn);
        return pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
      }

      // Open label file
      FILE* fp_label = std::fopen(fn_label.c_str(), "rb");
      if (!fp_label) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Cannot open label file: " << fn_label);
        std::fclose(fp);
        return pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
      }

      // Get file size for scan (x, y, z, intensity = 4 floats per point)
      std::fseek(fp, 0L, SEEK_END);
      size_t sz = std::ftell(fp);
      std::rewind(fp);
      int n_hits = static_cast<int>(sz / (sizeof(float) * 4));

      // Get label file size (expected: n_hits * sizeof(uint32_t) per label)
      std::fseek(fp_label, 0L, SEEK_END);
      long label_file_sz = std::ftell(fp_label);
      std::rewind(fp_label);
      int num_labels_in_file = static_cast<int>(label_file_sz / sizeof(uint32_t));
      // RCLCPP_INFO_STREAM(node_->get_logger(), "\n\n\n\n     Scan: " << fn << " \n\nlabel file: " << fn_label << " number of labels: " << num_labels_in_file << "\n\n\n\n");

      // Preallocate point cloud for better performance (avoids reallocation)
      pcl::PointCloud<pcl::PointXYZL>::Ptr pc(new pcl::PointCloud<pcl::PointXYZL>);
      if (!pc) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Failed to allocate point cloud!");
        std::fclose(fp);
        std::fclose(fp_label);
        return pcl::PointCloud<pcl::PointXYZL>::Ptr(new pcl::PointCloud<pcl::PointXYZL>);
      }
      pc->points.reserve(n_hits);  // Preallocate to avoid reallocation overhead
      pc->width = n_hits;
      pc->height = 1;
      pc->is_dense = false;

      // Read data in a tighter loop; collect unique class IDs for logging
      int points_read = 0;
      std::set<int> unique_labels;
      for (int i = 0; i < n_hits; i++) {
        pcl::PointXYZL point;
        float intensity;
        uint32_t label;

        // Read point data (x, y, z, intensity as floats)
        if (fread(&point.x, sizeof(float), 1, fp) != 1) break;
        if (fread(&point.y, sizeof(float), 1, fp) != 1) break;
        if (fread(&point.z, sizeof(float), 1, fp) != 1) break;
        if (fread(&intensity, sizeof(float), 1, fp) != 1) break;

        // Read label (uint32) and map to common taxonomy
        if (fread(&label, sizeof(uint32_t), 1, fp_label) != 1) break;

        int raw = static_cast<int>(label);
        if (common_label_config_loaded_) {
          const auto& mapping = use_gt_mapping ? gt_to_common_ : inferred_to_common_;
          auto it = mapping.find(raw);
          point.label = (it != mapping.end()) ? it->second : 0;
        } else {
          point.label = raw;
        }
        unique_labels.insert(point.label);
        pc->points.push_back(point);
        points_read++;
      }
      
      std::fclose(fp);
      std::fclose(fp_label);
      
      // // Print list of unique class IDs in this label file (should match Python inspect_label_file.py)
      // std::ostringstream oss;
      // oss << "Unique classes in label file (" << fn_label << "): ";
      // for (auto it = unique_labels.begin(); it != unique_labels.end(); ++it) {
      //   if (it != unique_labels.begin()) oss << ", ";
      //   oss << *it;
      // }
      // RCLCPP_INFO_STREAM(node_->get_logger(), oss.str());
      // // Sanity: first few label values (compare with Python inspect_label_file.py)
      // if (points_read >= 5) {
      //   std::ostringstream dbg;
      //   dbg << "First 5 label values (uint32): " << pc->points[0].label << ", " << pc->points[1].label << ", " << pc->points[2].label << ", " << pc->points[3].label << ", " << pc->points[4].label;
      //   RCLCPP_INFO_STREAM(node_->get_logger(), dbg.str());
      // }
      
      // // RED error if label count does not match point count
      // if (points_read != n_hits || num_labels_in_file != n_hits) {
      //   RCLCPP_ERROR_STREAM(node_->get_logger(),
      //     "ERROR: Label count does not match point count! scan=" << fn
      //     << " label_file=" << fn_label
      //     << " points_in_scan=" << n_hits
      //     << " labels_in_file=" << num_labels_in_file
      //     << " points_read=" << points_read);
      // }
      
      return pc;
    }

    /// Read lidar scan + multiclass confidence scores (float16), take argmax,
    /// apply learning_map_inv.  Also computes per-point variance of the class
    /// probability distribution (used for uncertainty filtering).
    MulticlassResult mcd2pcl_multiclass(
        const std::string& fn, const std::string& fn_multiclass) {

      MulticlassResult result;
      result.cloud.reset(new pcl::PointCloud<pcl::PointXYZL>);

      FILE* fp = std::fopen(fn.c_str(), "rb");
      if (!fp) {
        RCLCPP_WARN_STREAM(node_->get_logger(),
            "Cannot open scan file: " << fn);
        return result;
      }

      FILE* fp_mc = std::fopen(fn_multiclass.c_str(), "rb");
      if (!fp_mc) {
        RCLCPP_WARN_STREAM(node_->get_logger(),
            "Cannot open multiclass file: " << fn_multiclass);
        std::fclose(fp);
        return result;
      }

      std::fseek(fp, 0L, SEEK_END);
      size_t scan_sz = std::ftell(fp);
      std::rewind(fp);
      int n_points = static_cast<int>(scan_sz / (sizeof(float) * 4));

      std::fseek(fp_mc, 0L, SEEK_END);
      size_t mc_sz = std::ftell(fp_mc);
      std::rewind(fp_mc);
      int n_mc_values = static_cast<int>(mc_sz / sizeof(uint16_t));

      if (n_points == 0 || n_mc_values == 0) {
        std::fclose(fp);
        std::fclose(fp_mc);
        return result;
      }

      int n_classes = n_mc_values / n_points;
      if (n_mc_values != n_points * n_classes) {
        RCLCPP_ERROR_STREAM(node_->get_logger(),
            "Multiclass file size mismatch: " << n_mc_values
            << " values for " << n_points << " points (not divisible)");
        std::fclose(fp);
        std::fclose(fp_mc);
        return result;
      }
      result.n_classes = n_classes;

      RCLCPP_INFO_STREAM(node_->get_logger(),
          "Multiclass: " << n_points << " points x " << n_classes << " classes");

      std::vector<uint16_t> mc_raw(n_mc_values);
      if (std::fread(mc_raw.data(), sizeof(uint16_t), n_mc_values, fp_mc)
          != static_cast<size_t>(n_mc_values)) {
        RCLCPP_ERROR_STREAM(node_->get_logger(),
            "Failed to read multiclass data from " << fn_multiclass);
        std::fclose(fp);
        std::fclose(fp_mc);
        return result;
      }
      std::fclose(fp_mc);

      auto& pc = result.cloud;
      pc->points.reserve(n_points);
      pc->width = n_points;
      pc->height = 1;
      pc->is_dense = false;
      result.variances.reserve(n_points);

      for (int i = 0; i < n_points; i++) {
        pcl::PointXYZL point;
        float intensity;
        if (std::fread(&point.x, sizeof(float), 1, fp) != 1) break;
        if (std::fread(&point.y, sizeof(float), 1, fp) != 1) break;
        if (std::fread(&point.z, sizeof(float), 1, fp) != 1) break;
        if (std::fread(&intensity, sizeof(float), 1, fp) != 1) break;

        const uint16_t* row_ptr = mc_raw.data() + static_cast<size_t>(i) * n_classes;

        // Convert float16 → float32, find argmax, and compute variance
        int best_class = 0;
        float best_val = half_to_float(row_ptr[0]);
        float sum = best_val;
        float sum_sq = best_val * best_val;
        for (int c = 1; c < n_classes; c++) {
          float v = half_to_float(row_ptr[c]);
          sum += v;
          sum_sq += v * v;
          if (v > best_val) {
            best_val = v;
            best_class = c;
          }
        }
        float mean = sum / n_classes;
        float variance = sum_sq / n_classes - mean * mean;
        if (variance < 0.0f) variance = 0.0f;
        result.variances.push_back(variance);

        // Network class index → raw label (via learning_map_inv) → common class
        int raw_label = best_class;
        if (!learning_map_inv_.empty()) {
          auto it_inv = learning_map_inv_.find(best_class);
          if (it_inv != learning_map_inv_.end()) raw_label = it_inv->second;
        }
        int common_label = raw_label;
        if (common_label_config_loaded_) {
          auto it_c = inferred_to_common_.find(raw_label);
          if (it_c != inferred_to_common_.end()) common_label = it_c->second;
          else common_label = 0;
        }
        point.label = static_cast<uint32_t>(common_label);
        pc->points.push_back(point);
      }

      std::fclose(fp);
      pc->width = static_cast<uint32_t>(pc->points.size());
      return result;
    }

    /// Read GT from multiclass/one-hot format (float16, n_points * n_classes).
    /// Takes argmax, applies gt_learning_map_inv, then gt_to_common. Returns cloud with labels in common taxonomy.
    pcl::PointCloud<pcl::PointXYZL>::Ptr mcd2pcl_gt_multiclass(
        const std::string& fn_scan, const std::string& fn_gt_multiclass) {
      pcl::PointCloud<pcl::PointXYZL>::Ptr pc(new pcl::PointCloud<pcl::PointXYZL>);
      FILE* fp = std::fopen(fn_scan.c_str(), "rb");
      if (!fp) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "Cannot open scan file: " << fn_scan);
        return pc;
      }
      FILE* fp_gt = std::fopen(fn_gt_multiclass.c_str(), "rb");
      if (!fp_gt) {
        RCLCPP_WARN_STREAM(node_->get_logger(), "Cannot open GT multiclass file: " << fn_gt_multiclass);
        std::fclose(fp);
        return pc;
      }
      std::fseek(fp, 0L, SEEK_END);
      size_t scan_sz = std::ftell(fp);
      std::rewind(fp);
      int n_points = static_cast<int>(scan_sz / (sizeof(float) * 4));
      std::fseek(fp_gt, 0L, SEEK_END);
      size_t gt_sz = std::ftell(fp_gt);
      std::rewind(fp_gt);
      int n_mc = static_cast<int>(gt_sz / sizeof(uint16_t));
      if (n_points == 0 || n_mc == 0) {
        std::fclose(fp);
        std::fclose(fp_gt);
        return pc;
      }
      int n_classes = n_mc / n_points;
      if (n_mc != n_points * n_classes) {
        RCLCPP_ERROR_STREAM(node_->get_logger(), "GT multiclass file size mismatch");
        std::fclose(fp);
        std::fclose(fp_gt);
        return pc;
      }
      std::vector<uint16_t> mc_raw(static_cast<size_t>(n_mc));
      if (std::fread(mc_raw.data(), sizeof(uint16_t), n_mc, fp_gt) != static_cast<size_t>(n_mc)) {
        std::fclose(fp);
        std::fclose(fp_gt);
        return pc;
      }
      std::fclose(fp_gt);
      pc->points.reserve(n_points);
      pc->width = n_points;
      pc->height = 1;
      pc->is_dense = false;
      for (int i = 0; i < n_points; i++) {
        pcl::PointXYZL point;
        float intensity;
        if (std::fread(&point.x, sizeof(float), 1, fp) != 1) break;
        if (std::fread(&point.y, sizeof(float), 1, fp) != 1) break;
        if (std::fread(&point.z, sizeof(float), 1, fp) != 1) break;
        if (std::fread(&intensity, sizeof(float), 1, fp) != 1) break;
        const uint16_t* row = mc_raw.data() + static_cast<size_t>(i) * n_classes;
        int best_class = 0;
        float best_val = half_to_float(row[0]);
        for (int c = 1; c < n_classes; c++) {
          float v = half_to_float(row[c]);
          if (v > best_val) { best_val = v; best_class = c; }
        }
        int raw_label = best_class;
        if (!gt_learning_map_inv_.empty()) {
          auto it = gt_learning_map_inv_.find(best_class);
          if (it != gt_learning_map_inv_.end()) raw_label = it->second;
        }
        int common_label = raw_label;
        if (common_label_config_loaded_) {
          auto it_c = gt_to_common_.find(raw_label);
          if (it_c != gt_to_common_.end()) common_label = it_c->second;
          else common_label = 0;
        }
        point.label = static_cast<uint32_t>(common_label);
        pc->points.push_back(point);
      }
      std::fclose(fp);
      pc->width = static_cast<uint32_t>(pc->points.size());
      return pc;
    }

    /// Read GT label file (uint32 per point). Returns empty vector on failure.
    std::vector<uint32_t> read_gt_labels(const std::string& dir, int scan_file_num) {
      char buf[256];
      std::snprintf(buf, sizeof(buf), "%010d", scan_file_num);
      std::string path = dir + "/" + std::string(buf) + ".bin";
      FILE* fp = std::fopen(path.c_str(), "rb");
      if (!fp) return {};
      std::fseek(fp, 0L, SEEK_END);
      size_t sz = std::ftell(fp);
      std::rewind(fp);
      int n = static_cast<int>(sz / sizeof(uint32_t));
      std::vector<uint32_t> labels(n);
      if (std::fread(labels.data(), sizeof(uint32_t), n, fp) != static_cast<size_t>(n))
        labels.clear();
      std::fclose(fp);
      return labels;
    }
};
