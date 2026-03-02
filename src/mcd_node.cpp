#include <string>
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>
#include <cstdlib>
#include <fstream>

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>

#include "bkioctomap.h"
#include "markerarray_pub.h"
#include "mcd_util.h"
#include "osm_visualizer.h"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("mcd_node");
    
    if (!node) {
        RCLCPP_WARN_STREAM(rclcpp::get_logger("mcd_node"), "WARNING: Failed to create ROS2 node!");
        return 1;
    }
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Node created successfully");

    std::string map_topic("/occupied_cells_vis_array");
    int block_depth = 4;
    double sf2 = 1.0;
    double ell = 1.0;
    float prior = 1.0f;
    float var_thresh = 1.0f;
    double free_thresh = 0.3;
    double occupied_thresh = 0.7;
    double resolution = 0.1;
    int num_class = 2;
    double free_resolution = 0.5;
    double ds_resolution = 0.1;
    int scan_num = 0;
    double max_range = -1;
    int skip_frames = 0;
    
    // MCD Dataset
    std::string dir;
    std::string input_data_prefix;
    std::string input_label_prefix;
    std::string lidar_pose_file;
    std::string gt_label_prefix;
    std::string evaluation_result_prefix;
    bool query = false;
    bool visualize = false;

    // Declare parameters
    node->declare_parameter<std::string>("map_topic", map_topic);
    node->declare_parameter<int>("block_depth", block_depth);
    node->declare_parameter<double>("sf2", sf2);
    node->declare_parameter<double>("ell", ell);
    node->declare_parameter<float>("prior", prior);
    node->declare_parameter<float>("var_thresh", var_thresh);
    node->declare_parameter<double>("free_thresh", free_thresh);
    node->declare_parameter<double>("occupied_thresh", occupied_thresh);
    node->declare_parameter<double>("resolution", resolution);
    node->declare_parameter<int>("num_class", num_class);
    node->declare_parameter<double>("free_resolution", free_resolution);
    node->declare_parameter<double>("ds_resolution", ds_resolution);
    node->declare_parameter<int>("scan_num", scan_num);
    node->declare_parameter<double>("max_range", max_range);
    node->declare_parameter<int>("skip_frames", skip_frames);
    node->declare_parameter<std::string>("dir", dir);
    node->declare_parameter<std::string>("input_data_prefix", input_data_prefix);
    node->declare_parameter<std::string>("input_label_prefix", input_label_prefix);
    node->declare_parameter<std::string>("lidar_pose_file", lidar_pose_file);
    node->declare_parameter<std::string>("gt_label_prefix", gt_label_prefix);
    node->declare_parameter<std::string>("evaluation_result_prefix", evaluation_result_prefix);
    node->declare_parameter<bool>("query", query);
    node->declare_parameter<bool>("visualize", visualize);
    node->declare_parameter<std::string>("colors_file", "");
    node->declare_parameter<std::string>("calibration_file", "");
    node->declare_parameter<std::string>("osm_file", "");
    node->declare_parameter<double>("osm_origin_lat", 0.0);
    node->declare_parameter<double>("osm_origin_lon", 0.0);
    node->declare_parameter<double>("osm_decay_meters", 2.0);
    node->declare_parameter<double>("osm_tree_point_radius_meters", 5.0);
    node->declare_parameter<bool>("inferred_use_multiclass", false);
    node->declare_parameter<bool>("gt_use_multiclass", false);
    node->declare_parameter<bool>("use_uncertainty_filter", false);
    node->declare_parameter<std::string>("inferred_labels_key", "mcd");
    node->declare_parameter<std::string>("gt_labels_key", "mcd");
    node->declare_parameter<std::string>("confusion_matrix_file", "");
    node->declare_parameter<std::string>("uncertainty_filter_mode", "confusion_matrix");
    node->declare_parameter<double>("uncertainty_drop_percent", 10.0);
    node->declare_parameter<double>("uncertainty_min_weight", 0.1);
    node->declare_parameter<std::string>("osm_confusion_matrix_file", "");
    node->declare_parameter<double>("osm_prior_strength", 0.0);
    node->declare_parameter<bool>("publish_osm_prior_map", false);
    node->declare_parameter<std::string>("osm_prior_map_color_mode", "osm_blend");
    node->declare_parameter<std::string>("osm_prior_map_topic", "/semantic_osm_prior_map");
    node->declare_parameter<bool>("publish_variance", false);
    node->declare_parameter<std::string>("variance_topic", "/semantic_bki_variance");
    node->declare_parameter<bool>("publish_semantic_uncertainty", false);
    node->declare_parameter<std::string>("semantic_uncertainty_topic", "/semantic_uncertainty_cloud");

    // Get parameters
    node->get_parameter<std::string>("map_topic", map_topic);
    node->get_parameter<int>("block_depth", block_depth);
    node->get_parameter<double>("sf2", sf2);
    node->get_parameter<double>("ell", ell);
    node->get_parameter<float>("prior", prior);
    node->get_parameter<float>("var_thresh", var_thresh);
    node->get_parameter<double>("free_thresh", free_thresh);
    node->get_parameter<double>("occupied_thresh", occupied_thresh);
    node->get_parameter<double>("resolution", resolution);
    node->get_parameter<int>("num_class", num_class);
    node->get_parameter<double>("free_resolution", free_resolution);
    node->get_parameter<double>("ds_resolution", ds_resolution);
    node->get_parameter<int>("scan_num", scan_num);
    node->get_parameter<double>("max_range", max_range);
    node->get_parameter<int>("skip_frames", skip_frames);
    node->get_parameter<std::string>("dir", dir);
    node->get_parameter<std::string>("input_data_prefix", input_data_prefix);
    node->get_parameter<std::string>("input_label_prefix", input_label_prefix);
    node->get_parameter<std::string>("lidar_pose_file", lidar_pose_file);
    node->get_parameter<std::string>("gt_label_prefix", gt_label_prefix);
    node->get_parameter<std::string>("evaluation_result_prefix", evaluation_result_prefix);
    node->get_parameter<bool>("query", query);
    node->get_parameter<bool>("visualize", visualize);
    
    // Color configuration
    std::string colors_file;
    node->get_parameter<std::string>("colors_file", colors_file);
    
    // Calibration file
    std::string calibration_file;
    node->get_parameter<std::string>("calibration_file", calibration_file);

    std::string osm_file;
    double osm_origin_lat, osm_origin_lon, osm_decay_meters, osm_tree_point_radius_meters;
    node->get_parameter<std::string>("osm_file", osm_file);
    node->get_parameter<double>("osm_origin_lat", osm_origin_lat);
    node->get_parameter<double>("osm_origin_lon", osm_origin_lon);
    node->get_parameter<double>("osm_decay_meters", osm_decay_meters);
    node->get_parameter<double>("osm_tree_point_radius_meters", osm_tree_point_radius_meters);
    
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: All parameters retrieved. dir=" << dir << ", lidar_pose_file=" << lidar_pose_file << ", calibration_file=" << calibration_file);

    RCLCPP_INFO_STREAM(node->get_logger(), "Parameters:" << std::endl <<
      "block_depth: " << block_depth << std::endl <<
      "sf2: " << sf2 << std::endl <<
      "ell: " << ell << std::endl <<
      "prior:" << prior << std::endl <<
      "var_thresh: " << var_thresh << std::endl <<
      "free_thresh: " << free_thresh << std::endl <<
      "occupied_thresh: " << occupied_thresh << std::endl <<
      "resolution: " << resolution << std::endl <<
      "num_class: " << num_class << std::endl << 
      "free_resolution: " << free_resolution << std::endl <<
      "ds_resolution: " << ds_resolution << std::endl <<
      "scan_num: " << scan_num << std::endl <<
      "max_range: " << max_range << std::endl <<
      "skip_frames: " << skip_frames << std::endl <<

      "MCD:" << std::endl <<
      "dir: " << dir << std::endl <<
      "input_data_prefix: " << input_data_prefix << std::endl <<
      "input_label_prefix: " << input_label_prefix << std::endl <<
      "lidar_pose_file: " << lidar_pose_file << std::endl <<
      "gt_label_prefix: " << gt_label_prefix << std::endl <<
      "evaluation_result_prefix: " << evaluation_result_prefix << std::endl <<
      "query: " << query << std::endl <<
      "visualize:" << visualize
      );

    
    // Publish static transform for "map" frame (required for RViz visualization)
    // This ensures the map frame exists even before any scans are processed
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to publish static transform");
    try {
        tf2_ros::StaticTransformBroadcaster static_tf_broadcaster(node);
        geometry_msgs::msg::TransformStamped static_transform;
        static_transform.header.stamp = node->now();
        static_transform.header.frame_id = "map";
        static_transform.child_frame_id = "odom";  // Common base frame, or use "base_link" if preferred
        static_transform.transform.translation.x = 0.0;
        static_transform.transform.translation.y = 0.0;
        static_transform.transform.translation.z = 0.0;
        static_transform.transform.rotation.x = 0.0;
        static_transform.transform.rotation.y = 0.0;
        static_transform.transform.rotation.z = 0.0;
        static_transform.transform.rotation.w = 1.0;
        static_tf_broadcaster.sendTransform(static_transform);
        RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Static transform published successfully");
        RCLCPP_INFO(node->get_logger(), "Published static transform: map -> odom (identity)");
    } catch (const std::exception& e) {
        RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Exception publishing static transform: " << e.what());
        RCLCPP_ERROR_STREAM(node->get_logger(), "Exception publishing static transform: " << e.what());
    }
    
    ///////// Build Map /////////////////////
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to create MCDData object");
    MCDData mcd_data(node, resolution, block_depth, sf2, ell, num_class, free_thresh, occupied_thresh, var_thresh, ds_resolution, free_resolution, max_range, map_topic, prior);
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: MCDData object created successfully");
    
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to read lidar poses from: " << (dir + '/' + lidar_pose_file));
    if (!mcd_data.read_lidar_poses(dir + '/' + lidar_pose_file)) {
        RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Failed to read lidar poses!");
        return 1;
    }
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Lidar poses read successfully");
    
    // Load body-to-lidar calibration from hhs_calib.yaml (if provided via ROS parameters)
    // The calibration file can be loaded via rosparam or included in the launch file
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to load calibration from params");
    if (!mcd_data.load_calibration_from_params()) {
      RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Failed to load body-to-lidar calibration!");
      RCLCPP_FATAL(node->get_logger(), "Failed to load body-to-lidar calibration! Cannot proceed without calibration.");
      return 1;
    }
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Calibration loaded successfully");

    // Load common taxonomy label mappings (raw labels → common class indices)
    {
      std::string inferred_labels_key, gt_labels_key;
      node->get_parameter<std::string>("inferred_labels_key", inferred_labels_key);
      node->get_parameter<std::string>("gt_labels_key", gt_labels_key);

      std::string common_label_path;
      size_t dp = dir.rfind("/data/");
      if (dp != std::string::npos) {
        common_label_path = dir.substr(0, dp) + "/config/datasets/labels_common.yaml";
      } else {
        common_label_path = ament_index_cpp::get_package_share_directory("semantic_bki")
                            + "/config/datasets/labels_common.yaml";
      }
      if (!mcd_data.load_common_label_config(common_label_path, inferred_labels_key, gt_labels_key)) {
        RCLCPP_FATAL_STREAM(node->get_logger(),
            "Failed to load common label config from " << common_label_path
            << ". Cannot proceed without label mappings.");
        return 1;
      }
    }

    // Multiclass setup: inferred and/or GT can use per-class scores (float16) instead of single uint32 labels.
    // learning_map_inv is inferred from inferred_labels_key / gt_labels_key (labels_semkitti.yaml, labels_mcd.yaml, or labels_kitti360.yaml).
    auto resolve_label_config_path = [&](const std::string& labels_key) -> std::string {
      std::string f;
      if (labels_key == "mcd")
        f = "labels_mcd.yaml";
      else if (labels_key == "kitti360")
        f = "labels_kitti360.yaml";
      else
        f = "labels_semkitti.yaml";
      size_t dp = dir.rfind("/data/");
      if (dp != std::string::npos)
        return dir.substr(0, dp) + "/config/datasets/" + f;
      return ament_index_cpp::get_package_share_directory("semantic_bki") + "/config/datasets/" + f;
    };

    bool inferred_use_multiclass = false, gt_use_multiclass = false;
    std::string inferred_labels_key_val, gt_labels_key_val;
    node->get_parameter<bool>("inferred_use_multiclass", inferred_use_multiclass);
    node->get_parameter<bool>("gt_use_multiclass", gt_use_multiclass);
    node->get_parameter<std::string>("inferred_labels_key", inferred_labels_key_val);
    node->get_parameter<std::string>("gt_labels_key", gt_labels_key_val);

    if (inferred_use_multiclass) {
      mcd_data.set_inferred_multiclass_mode(true, dir + '/' + input_label_prefix);
      if (!mcd_data.load_label_config(resolve_label_config_path(inferred_labels_key_val))) {
        RCLCPP_WARN_STREAM(node->get_logger(),
            "Failed to load inferred label config. Argmax indices will be used as-is.");
      }
      bool use_uncertainty_filter = false;
      std::string confusion_matrix_file, uncertainty_filter_mode;
      double uncertainty_drop_percent, uncertainty_min_weight;
      node->get_parameter<bool>("use_uncertainty_filter", use_uncertainty_filter);
      node->get_parameter<std::string>("confusion_matrix_file", confusion_matrix_file);
      node->get_parameter<std::string>("uncertainty_filter_mode", uncertainty_filter_mode);
      node->get_parameter<double>("uncertainty_drop_percent", uncertainty_drop_percent);
      node->get_parameter<double>("uncertainty_min_weight", uncertainty_min_weight);
      mcd_data.set_uncertainty_filter(use_uncertainty_filter, inferred_labels_key_val,
                                      uncertainty_filter_mode,
                                      static_cast<float>(uncertainty_drop_percent),
                                      static_cast<float>(uncertainty_min_weight));
      if (use_uncertainty_filter && !confusion_matrix_file.empty()) {
        size_t dp = dir.rfind("/data/");
        std::string cm_path = (dp != std::string::npos)
            ? dir.substr(0, dp) + "/config/datasets/" + confusion_matrix_file
            : confusion_matrix_file;
        if (!mcd_data.load_confusion_matrix(cm_path)) {
          RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load confusion matrix. Uncertainty filtering disabled.");
        }
      }
    }

    if (gt_use_multiclass) {
      mcd_data.set_gt_multiclass_mode(true);
      if (!mcd_data.load_gt_label_config(resolve_label_config_path(gt_labels_key_val))) {
        RCLCPP_WARN_STREAM(node->get_logger(),
            "Failed to load GT label config. Argmax indices will be used as-is.");
      }
    }

    // Load colors from YAML file specified in colors_file parameter
    // Load directly into MarkerArrayPub instead of using ROS parameters
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to load colors");
    if (colors_file.empty()) {
      RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: No colors_file specified in dataset config. Using default hardcoded colors.");
    } else {
      // Derive config path from dir (<pkg_root>/data/<dataset>)
      std::string colors_file_path;
      size_t dp = dir.rfind("/data/");
      if (dp != std::string::npos) {
        colors_file_path = dir.substr(0, dp) + "/config/datasets/" + colors_file;
      } else {
        colors_file_path = ament_index_cpp::get_package_share_directory("semantic_bki")
                           + "/config/datasets/" + colors_file;
      }
      {
        RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Colors file path: " << colors_file_path);
        RCLCPP_INFO_STREAM(node->get_logger(), "Loading colors from file specified in config: " << colors_file_path);
        // Load colors directly into MarkerArrayPub (bypasses ROS parameter system)
        if (mcd_data.load_colors_from_yaml(colors_file_path)) {
          RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Colors loaded successfully");
          RCLCPP_INFO_STREAM(node->get_logger(), "Successfully loaded colors from: " << colors_file_path);
        } else {
          RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Failed to load colors from: " << colors_file_path << ". Using default hardcoded colors.");
        }
      }
    }
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Color loading completed");

    // Main map always uses semantic colors
    mcd_data.set_color_mode(semantic_bki::MapColorMode::Semantic);

    mcd_data.set_osm_decay_meters(static_cast<float>(osm_decay_meters));
    mcd_data.set_osm_tree_point_radius(static_cast<float>(osm_tree_point_radius_meters));

    // Optional: load OSM geometries for voxel priors (same frame as map)
    if (!osm_file.empty()) {
      std::string full_osm_path = osm_file;
      if (osm_file[0] != '/' && !dir.empty()) {
        full_osm_path = dir + "/" + osm_file;
      }
      semantic_bki::OSMVisualizer osm_vis(node, "");
      if (osm_vis.loadFromOSM(full_osm_path, osm_origin_lat, osm_origin_lon)) {
        osm_vis.transformToFirstPoseOrigin(mcd_data.getOriginalFirstPose());
        mcd_data.set_osm_buildings(osm_vis.getBuildings());
        mcd_data.set_osm_roads(osm_vis.getRoads());
        mcd_data.set_osm_sidewalks(osm_vis.getSidewalks());
        mcd_data.set_osm_cycleways(osm_vis.getCycleways());
        mcd_data.set_osm_grasslands(osm_vis.getGrasslands());
        mcd_data.set_osm_trees(osm_vis.getTrees());
        mcd_data.set_osm_forests(osm_vis.getForests());
        mcd_data.set_osm_tree_points(osm_vis.getTreePoints());
        mcd_data.set_osm_parking(osm_vis.getParking());
        mcd_data.set_osm_fences(osm_vis.getFences());
        mcd_data.set_osm_walls(osm_vis.getWalls());
        mcd_data.set_osm_stairs(osm_vis.getStairs());
        mcd_data.set_osm_water(osm_vis.getWater());
        mcd_data.set_osm_pole_points(osm_vis.getPolePoints());
        mcd_data.set_osm_stairs_width(osm_vis.getStairsWidth());
        RCLCPP_INFO_STREAM(node->get_logger(), "Loaded OSM geometries for voxel priors: "
            << osm_vis.getBuildings().size() << " buildings, " << osm_vis.getRoads().size() << " roads, "
            << osm_vis.getSidewalks().size() << " sidewalks, " << osm_vis.getCycleways().size() << " cycleways, "
            << osm_vis.getParking().size() << " parking, " << osm_vis.getFences().size() << " fences, "
            << osm_vis.getWalls().size() << " walls, " << osm_vis.getStairs().size() << " stairs, "
            << osm_vis.getGrasslands().size() << " grasslands, " << osm_vis.getTrees().size() << " trees, "
            << osm_vis.getForests().size() << " forests, " << osm_vis.getTreePoints().size() << " tree points, "
            << osm_vis.getWater().size() << " water, " << osm_vis.getPolePoints().size() << " pole points (decay=" << osm_decay_meters << " m)");
      } else {
        RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load OSM file for priors: " << full_osm_path);
      }
    }

    // OSM confusion matrix for semantic-OSM prior fusion
    {
      std::string osm_cm_file;
      double osm_prior_str;
      node->get_parameter<std::string>("osm_confusion_matrix_file", osm_cm_file);
      node->get_parameter<double>("osm_prior_strength", osm_prior_str);
      bool publish_variance = false;
      std::string variance_topic = "/semantic_bki_variance";
      node->get_parameter<bool>("publish_variance", publish_variance);
      node->get_parameter<std::string>("variance_topic", variance_topic);
      mcd_data.set_publish_variance(publish_variance, variance_topic);
      bool publish_semantic_uncertainty = false;
      std::string semantic_uncertainty_topic = "/semantic_uncertainty_cloud";
      node->get_parameter<bool>("publish_semantic_uncertainty", publish_semantic_uncertainty);
      node->get_parameter<std::string>("semantic_uncertainty_topic", semantic_uncertainty_topic);
      mcd_data.set_publish_semantic_uncertainty(publish_semantic_uncertainty, semantic_uncertainty_topic);
      mcd_data.set_osm_prior_strength(static_cast<float>(osm_prior_str));

      bool publish_osm_prior_map = false;
      std::string osm_prior_map_color_mode_str = "osm_blend";
      std::string osm_prior_map_topic = "/semantic_osm_prior_map";
      node->get_parameter<bool>("publish_osm_prior_map", publish_osm_prior_map);
      node->get_parameter<std::string>("osm_prior_map_color_mode", osm_prior_map_color_mode_str);
      node->get_parameter<std::string>("osm_prior_map_topic", osm_prior_map_topic);
      semantic_bki::MapColorMode osm_color_mode = semantic_bki::MapColorMode::OSMBlend;
      if (osm_prior_map_color_mode_str == "osm_building") {
        osm_color_mode = semantic_bki::MapColorMode::OSMBuilding;
      } else if (osm_prior_map_color_mode_str == "osm_road") {
        osm_color_mode = semantic_bki::MapColorMode::OSMRoad;
      } else if (osm_prior_map_color_mode_str == "osm_grassland") {
        osm_color_mode = semantic_bki::MapColorMode::OSMGrassland;
      } else if (osm_prior_map_color_mode_str == "osm_tree") {
        osm_color_mode = semantic_bki::MapColorMode::OSMTree;
      } else if (osm_prior_map_color_mode_str == "osm_parking") {
        osm_color_mode = semantic_bki::MapColorMode::OSMParking;
      } else if (osm_prior_map_color_mode_str == "osm_fence") {
        osm_color_mode = semantic_bki::MapColorMode::OSMFence;
      } else if (osm_prior_map_color_mode_str == "osm_blend") {
        osm_color_mode = semantic_bki::MapColorMode::OSMBlend;
      }
      mcd_data.set_publish_osm_prior_map(publish_osm_prior_map, osm_prior_map_topic, osm_color_mode);
      if (!osm_cm_file.empty() && osm_prior_str > 0.0) {
        std::string cm_path;
        size_t data_pos = dir.rfind("/data/");
        if (data_pos != std::string::npos) {
          cm_path = dir.substr(0, data_pos) + "/config/datasets/" + osm_cm_file;
        } else {
          cm_path = osm_cm_file;
        }
        if (mcd_data.load_osm_confusion_matrix(cm_path)) {
          RCLCPP_INFO_STREAM(node->get_logger(),
              "Loaded OSM confusion matrix from " << cm_path
              << " (strength=" << osm_prior_str << ")");
        } else {
          RCLCPP_WARN_STREAM(node->get_logger(),
              "Failed to load OSM confusion matrix from " << cm_path);
        }
      }
    }

    // Process scans
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to set up evaluation");
    mcd_data.set_up_evaluation(dir + '/' + gt_label_prefix, dir + '/' + evaluation_result_prefix);
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Evaluation setup completed");
    
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to process scans. input_data_prefix=" << input_data_prefix << ", scan_num=" << scan_num);
    mcd_data.process_scans(dir + '/' + input_data_prefix, dir + '/' + input_label_prefix, scan_num, skip_frames, query, visualize);
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Scan processing completed, about to spin");
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Node pointer: " << node.get());
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Starting rclcpp::spin(node)...");
    
    rclcpp::spin(node);
    
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: rclcpp::spin() returned (node shutdown)");
    
    rclcpp::shutdown();
    return 0;
}
