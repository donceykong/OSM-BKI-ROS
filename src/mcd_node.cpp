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
    double keyframe_dist = 0.0;
    
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
    node->declare_parameter<double>("keyframe_dist", keyframe_dist);
    node->declare_parameter<std::string>("dir", dir);
    node->declare_parameter<std::string>("sequence_name", "");
    node->declare_parameter<std::string>("input_data_suffix", "");
    node->declare_parameter<std::string>("input_label_suffix", "");
    node->declare_parameter<std::string>("lidar_pose_suffix", "");
    node->declare_parameter<std::string>("gt_label_suffix", "");
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
    node->declare_parameter<std::string>("config_datasets_dir", "");
    node->declare_parameter<std::string>("osm_confusion_matrix_file", "");
    node->declare_parameter<double>("osm_prior_strength", 0.0);
    node->declare_parameter<double>("osm_dirichlet_prior_strength", 0.0);
    node->declare_parameter<double>("osm_scan_radius_extension", 1.2);
    node->declare_parameter<bool>("osm_height_filtering", false);
    node->declare_parameter<std::string>("height_filter_type", "discreet");
    node->declare_parameter<double>("height_kernel_lambda", 0.0);
    node->declare_parameter<double>("height_kernel_dead_zone", 0.0);
    node->declare_parameter<bool>("height_kernel_redistribute", false);
    node->declare_parameter<double>("height_kernel_gate", 0.0);
    node->declare_parameter<double>("sensor_mounting_height", 0.0);
    node->declare_parameter<std::vector<double>>("height_kernel_mu", std::vector<double>());
    node->declare_parameter<std::vector<double>>("height_kernel_tau", std::vector<double>());
    node->declare_parameter<bool>("publish_osm_height_bins_scan", false);
    node->declare_parameter<bool>("publish_osm_height_bins_map", false);
    node->declare_parameter<double>("osm_height_bins_step_meters", 5.0);
    node->declare_parameter<double>("osm_height_bins_map_leaf_size", 0.5);
    node->declare_parameter<std::string>("osm_height_bins_scan_topic", "/osm_height_bins_scan");
    node->declare_parameter<std::string>("osm_height_bins_map_topic", "/osm_height_bins_map");
    node->declare_parameter<bool>("publish_osm_prior_map", false);
    node->declare_parameter<std::string>("osm_prior_map_color_mode", "osm_blend");
    node->declare_parameter<std::string>("osm_prior_map_topic", "/semantic_osm_prior_map");
    node->declare_parameter<bool>("publish_variance", false);
    node->declare_parameter<std::string>("variance_topic", "/osm_bki_variance");
    node->declare_parameter<bool>("publish_semantic_uncertainty", false);
    node->declare_parameter<std::string>("semantic_uncertainty_topic", "/semantic_uncertainty_cloud");
    node->declare_parameter<bool>("publish_static_tf", true);
    node->declare_parameter<bool>("use_pose_index_as_scan_id", false);
    node->declare_parameter<bool>("use_common_taxonomy", true);

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
    node->get_parameter<double>("keyframe_dist", keyframe_dist);
    node->get_parameter<std::string>("dir", dir);
    std::string sequence_name, input_data_suffix, input_label_suffix, lidar_pose_suffix, gt_label_suffix;
    node->get_parameter<std::string>("sequence_name", sequence_name);
    node->get_parameter<std::string>("input_data_suffix", input_data_suffix);
    node->get_parameter<std::string>("input_label_suffix", input_label_suffix);
    node->get_parameter<std::string>("lidar_pose_suffix", lidar_pose_suffix);
    node->get_parameter<std::string>("gt_label_suffix", gt_label_suffix);
    node->get_parameter<std::string>("input_data_prefix", input_data_prefix);
    node->get_parameter<std::string>("input_label_prefix", input_label_prefix);
    node->get_parameter<std::string>("lidar_pose_file", lidar_pose_file);
    node->get_parameter<std::string>("gt_label_prefix", gt_label_prefix);
    node->get_parameter<std::string>("evaluation_result_prefix", evaluation_result_prefix);
    // Build paths from sequence_name + suffix when sequence-based config is used
    if (!sequence_name.empty() && !input_data_suffix.empty()) {
      input_data_prefix = sequence_name + "/" + input_data_suffix;
      if (!lidar_pose_suffix.empty()) lidar_pose_file = sequence_name + "/" + lidar_pose_suffix;
      if (!input_label_suffix.empty()) input_label_prefix = sequence_name + "/" + input_label_suffix;
      if (!gt_label_suffix.empty()) gt_label_prefix = sequence_name + "/" + gt_label_suffix;
      if (!evaluation_result_prefix.empty()) evaluation_result_prefix = sequence_name + "/" + evaluation_result_prefix;
    }
    node->get_parameter<bool>("query", query);
    node->get_parameter<bool>("visualize", visualize);
    
    // Color configuration
    std::string colors_file;
    node->get_parameter<std::string>("colors_file", colors_file);

    std::string config_datasets_dir;
    node->get_parameter<std::string>("config_datasets_dir", config_datasets_dir);
    
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
      "keyframe_dist: " << keyframe_dist << std::endl <<

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

    
    bool publish_static_tf = true;
    node->get_parameter<bool>("publish_static_tf", publish_static_tf);
    if (publish_static_tf) {
        try {
            tf2_ros::StaticTransformBroadcaster static_tf_broadcaster(node);
            geometry_msgs::msg::TransformStamped static_transform;
            static_transform.header.stamp = node->now();
            static_transform.header.frame_id = "map";
            static_transform.child_frame_id = "odom";
            static_transform.transform.translation.x = 0.0;
            static_transform.transform.translation.y = 0.0;
            static_transform.transform.translation.z = 0.0;
            static_transform.transform.rotation.x = 0.0;
            static_transform.transform.rotation.y = 0.0;
            static_transform.transform.rotation.z = 0.0;
            static_transform.transform.rotation.w = 1.0;
            static_tf_broadcaster.sendTransform(static_transform);
            RCLCPP_INFO(node->get_logger(), "Published static transform: map -> odom (identity)");
        } catch (const std::exception& e) {
            RCLCPP_WARN_STREAM(node->get_logger(), "WARNING: Exception publishing static transform: " << e.what());
        }
    } else {
        RCLCPP_INFO(node->get_logger(), "Skipping static TF (publish_static_tf=false).");
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
    bool use_pose_index_as_scan_id = false;
    node->get_parameter<bool>("use_pose_index_as_scan_id", use_pose_index_as_scan_id);
    if (use_pose_index_as_scan_id)
        mcd_data.apply_pose_index_as_scan_id();
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

    // Load common taxonomy label mappings (raw labels → common class indices).
    // When use_common_taxonomy is false, skip loading and use network class indices directly;
    // set num_class in config to the network's number of classes (e.g. 29 for MCD).
    bool use_common_taxonomy = true;
    node->get_parameter<bool>("use_common_taxonomy", use_common_taxonomy);
    // Prefer config from source (config_datasets_dir); else use install share
    std::string pkg_config_datasets;
    if (!config_datasets_dir.empty()) {
      pkg_config_datasets = config_datasets_dir;
      if (pkg_config_datasets.back() != '/') pkg_config_datasets += '/';
    } else {
      pkg_config_datasets = ament_index_cpp::get_package_share_directory("osm_bki") + "/config/datasets/";
    }

    if (use_common_taxonomy) {
      std::string inferred_labels_key, gt_labels_key;
      node->get_parameter<std::string>("inferred_labels_key", inferred_labels_key);
      node->get_parameter<std::string>("gt_labels_key", gt_labels_key);

      std::string common_label_path = pkg_config_datasets + "labels_common.yaml";
      if (!mcd_data.load_common_label_config(common_label_path, inferred_labels_key, gt_labels_key)) {
        RCLCPP_FATAL_STREAM(node->get_logger(),
            "Failed to load common label config from " << common_label_path
            << ". Cannot proceed without label mappings.");
        return 1;
      }
    } else {
      RCLCPP_INFO(node->get_logger(), "use_common_taxonomy=false: using network class indices (set num_class to network n_classes)");
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
      return pkg_config_datasets + f;
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
        std::string cm_path = pkg_config_datasets + confusion_matrix_file;
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
      std::string colors_file_path = pkg_config_datasets + colors_file;
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
    mcd_data.set_color_mode(osm_bki::MapColorMode::Semantic);

    mcd_data.set_osm_decay_meters(static_cast<float>(osm_decay_meters));
    mcd_data.set_osm_tree_point_radius(static_cast<float>(osm_tree_point_radius_meters));

    // Optional: load OSM geometries for voxel priors (same frame as map)
    if (!osm_file.empty()) {
      std::string full_osm_path = osm_file;
      if (osm_file[0] != '/' && !dir.empty()) {
        full_osm_path = dir + "/" + osm_file;
      }
      osm_bki::OSMVisualizer osm_vis(node, "");
      // Apply OSM geometry widths from ROS params (from osm_bki.yaml via launch, flattened as
      // "osm_geometry_parameters.<name>") BEFORE loadFromOSM so RawNetLines pick up the
      // configured widths rather than the hardcoded OSMVisualizer defaults.
      {
        auto apply_width = [&](const std::string& name, auto setter) {
          if (!node->has_parameter(name)) {
            node->declare_parameter<double>(name, 0.0);
          }
          double v = 0.0;
          node->get_parameter(name, v);
          if (v > 0.0) setter(static_cast<float>(v));
        };
        apply_width("osm_geometry_parameters.road_width_meters",     [&](float w){ osm_vis.setRoadWidth(w); });
        apply_width("osm_geometry_parameters.sidewalk_width_meters", [&](float w){ osm_vis.setSidewalkWidth(w); });
        apply_width("osm_geometry_parameters.cycleway_width_meters", [&](float w){ osm_vis.setCyclewayWidth(w); });
        apply_width("osm_geometry_parameters.fence_width_meters",    [&](float w){ osm_vis.setFenceWidth(w); });
        RCLCPP_INFO_STREAM(node->get_logger(), "OSM widths (m): road=" << osm_vis.getRoadWidth()
          << " sidewalk=" << osm_vis.getSidewalkWidth()
          << " cycleway=" << osm_vis.getCyclewayWidth()
          << " fence=" << osm_vis.getFenceWidth());
      }
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
        mcd_data.set_osm_road_width(osm_vis.getRoadWidth());
        mcd_data.set_osm_sidewalk_width(osm_vis.getSidewalkWidth());
        mcd_data.set_osm_cycleway_width(osm_vis.getCyclewayWidth());
        mcd_data.set_osm_fence_width(osm_vis.getFenceWidth());
        RCLCPP_INFO_STREAM(node->get_logger(), "Loaded OSM geometries for voxel priors: "
            << osm_vis.getBuildings().size() << " buildings, " << osm_vis.getRoads().size() << " roads, "
            << osm_vis.getSidewalks().size() << " sidewalks, " << osm_vis.getCycleways().size() << " cycleways, "
            << osm_vis.getParking().size() << " parking, " << osm_vis.getFences().size() << " fences, "
            << osm_vis.getGrasslands().size() << " grasslands, " << osm_vis.getTrees().size() << " trees, "
            << osm_vis.getForests().size() << " forests, " << osm_vis.getTreePoints().size() << " tree points (decay=" << osm_decay_meters << " m)");
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
      bool osm_height_filtering = false;
      node->get_parameter<bool>("osm_height_filtering", osm_height_filtering);
      bool publish_variance = false;
      std::string variance_topic = "/osm_bki_variance";
      node->get_parameter<bool>("publish_variance", publish_variance);
      node->get_parameter<std::string>("variance_topic", variance_topic);
      mcd_data.set_publish_variance(publish_variance, variance_topic);
      bool publish_semantic_uncertainty = false;
      std::string semantic_uncertainty_topic = "/semantic_uncertainty_cloud";
      node->get_parameter<bool>("publish_semantic_uncertainty", publish_semantic_uncertainty);
      node->get_parameter<std::string>("semantic_uncertainty_topic", semantic_uncertainty_topic);
      mcd_data.set_publish_semantic_uncertainty(publish_semantic_uncertainty, semantic_uncertainty_topic);
      mcd_data.set_osm_prior_strength(static_cast<float>(osm_prior_str));
      double osm_dirichlet_str = 0.0;
      node->get_parameter<double>("osm_dirichlet_prior_strength", osm_dirichlet_str);
      mcd_data.set_osm_dirichlet_prior_strength(static_cast<float>(osm_dirichlet_str));
      double osm_scan_ext = 1.2;
      node->get_parameter<double>("osm_scan_radius_extension", osm_scan_ext);
      mcd_data.set_osm_scan_radius_extension(static_cast<float>(osm_scan_ext));
      mcd_data.set_osm_height_filter_enabled(osm_height_filtering);

      // Height filter mode selection (discreet = current per-bin CM, gaussian = per-class Gaussian).
      std::string height_filter_type = "discreet";
      node->get_parameter<std::string>("height_filter_type", height_filter_type);
      bool use_gaussian_height = osm_height_filtering && (height_filter_type == "gaussian");
      mcd_data.set_height_filter_mode_gaussian(use_gaussian_height);
      if (use_gaussian_height) {
        double hk_lambda = 0.0, hk_dead_zone = 0.0, hk_gate = 0.0, sensor_height = 0.0;
        bool hk_redistribute = false;
        std::vector<double> mu_d, tau_d;
        node->get_parameter<double>("height_kernel_lambda", hk_lambda);
        node->get_parameter<double>("height_kernel_dead_zone", hk_dead_zone);
        node->get_parameter<bool>("height_kernel_redistribute", hk_redistribute);
        node->get_parameter<double>("height_kernel_gate", hk_gate);
        node->get_parameter<double>("sensor_mounting_height", sensor_height);
        node->get_parameter<std::vector<double>>("height_kernel_mu", mu_d);
        node->get_parameter<std::vector<double>>("height_kernel_tau", tau_d);
        std::vector<float> mu_f(mu_d.begin(), mu_d.end());
        std::vector<float> tau_f(tau_d.begin(), tau_d.end());
        mcd_data.set_height_kernel_params(static_cast<float>(hk_lambda), mu_f, tau_f,
                                          static_cast<float>(hk_dead_zone), hk_redistribute,
                                          static_cast<float>(hk_gate),
                                          static_cast<float>(sensor_height));
        RCLCPP_INFO_STREAM(node->get_logger(),
            "Height filter mode: gaussian (lambda=" << hk_lambda
            << ", dead_zone=" << hk_dead_zone << ", gate=" << hk_gate
            << ", sensor_height=" << sensor_height
            << ", " << mu_f.size() << " mu / " << tau_f.size() << " tau)");
      } else if (osm_height_filtering) {
        RCLCPP_INFO_STREAM(node->get_logger(), "Height filter mode: discreet (CM-based)");
      }

      bool publish_osm_height_bins_scan = false;
      bool publish_osm_height_bins_map = false;
      double osm_height_bins_step_meters = 5.0;
      double osm_height_bins_map_leaf_size = 0.5;
      std::string osm_height_bins_scan_topic = "/osm_height_bins_scan";
      std::string osm_height_bins_map_topic = "/osm_height_bins_map";
      node->get_parameter<bool>("publish_osm_height_bins_scan", publish_osm_height_bins_scan);
      node->get_parameter<bool>("publish_osm_height_bins_map", publish_osm_height_bins_map);
      node->get_parameter<double>("osm_height_bins_step_meters", osm_height_bins_step_meters);
      node->get_parameter<double>("osm_height_bins_map_leaf_size", osm_height_bins_map_leaf_size);
      node->get_parameter<std::string>("osm_height_bins_scan_topic", osm_height_bins_scan_topic);
      node->get_parameter<std::string>("osm_height_bins_map_topic", osm_height_bins_map_topic);
      mcd_data.set_publish_height_bins(publish_osm_height_bins_scan, publish_osm_height_bins_map,
                                       static_cast<float>(osm_height_bins_step_meters),
                                       static_cast<float>(osm_height_bins_map_leaf_size),
                                       osm_height_bins_scan_topic, osm_height_bins_map_topic);

      bool publish_osm_prior_map = false;
      std::string osm_prior_map_color_mode_str = "osm_blend";
      std::string osm_prior_map_topic = "/semantic_osm_prior_map";
      node->get_parameter<bool>("publish_osm_prior_map", publish_osm_prior_map);
      node->get_parameter<std::string>("osm_prior_map_color_mode", osm_prior_map_color_mode_str);
      node->get_parameter<std::string>("osm_prior_map_topic", osm_prior_map_topic);
      osm_bki::MapColorMode osm_color_mode = osm_bki::MapColorMode::OSMBlend;
      if (osm_prior_map_color_mode_str == "osm_building") {
        osm_color_mode = osm_bki::MapColorMode::OSMBuilding;
      } else if (osm_prior_map_color_mode_str == "osm_road") {
        osm_color_mode = osm_bki::MapColorMode::OSMRoad;
      } else if (osm_prior_map_color_mode_str == "osm_grassland") {
        osm_color_mode = osm_bki::MapColorMode::OSMGrassland;
      } else if (osm_prior_map_color_mode_str == "osm_tree") {
        osm_color_mode = osm_bki::MapColorMode::OSMTree;
      } else if (osm_prior_map_color_mode_str == "osm_parking") {
        osm_color_mode = osm_bki::MapColorMode::OSMParking;
      } else if (osm_prior_map_color_mode_str == "osm_fence") {
        osm_color_mode = osm_bki::MapColorMode::OSMFence;
      } else if (osm_prior_map_color_mode_str == "osm_sidewalk") {
        osm_color_mode = osm_bki::MapColorMode::OSMSidewalk;
      } else if (osm_prior_map_color_mode_str == "osm_cycleway") {
        osm_color_mode = osm_bki::MapColorMode::OSMCycleway;
      } else if (osm_prior_map_color_mode_str == "osm_forest") {
        osm_color_mode = osm_bki::MapColorMode::OSMForest;
      } else if (osm_prior_map_color_mode_str == "osm_blend") {
        osm_color_mode = osm_bki::MapColorMode::OSMBlend;
      }
      mcd_data.set_publish_osm_prior_map(publish_osm_prior_map, osm_prior_map_topic, osm_color_mode);
      // Widths and decay come from ROS params (osm_geometry_parameters.*) applied before
      // loadFromOSM; do not re-load from the confusion-matrix yaml here.
      if (!osm_cm_file.empty() && osm_prior_str > 0.0) {
        std::string cm_path = pkg_config_datasets + osm_cm_file;
        if (mcd_data.load_osm_confusion_matrix(cm_path)) {
          RCLCPP_INFO_STREAM(node->get_logger(),
              "Loaded OSM confusion matrix from " << cm_path
              << " (strength=" << osm_prior_str << ")");
        } else {
          RCLCPP_WARN_STREAM(node->get_logger(),
              "Failed to load OSM confusion matrix from " << cm_path);
        }
        if (osm_height_filtering && !use_gaussian_height) {
          if (mcd_data.load_osm_height_confusion_matrix(cm_path)) {
            RCLCPP_INFO_STREAM(node->get_logger(),
                "Loaded OSM height confusion matrix from " << cm_path);
          } else {
            RCLCPP_WARN_STREAM(node->get_logger(),
                "Failed to load OSM height confusion matrix; height filtering disabled");
          }
        }
      }
    }

    // Process scans
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to set up evaluation");
    mcd_data.set_up_evaluation(dir + '/' + gt_label_prefix, dir + '/' + evaluation_result_prefix);
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Evaluation setup completed");
    
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: About to process scans. input_data_prefix=" << input_data_prefix << ", scan_num=" << scan_num);
    mcd_data.process_scans(dir + '/' + input_data_prefix, dir + '/' + input_label_prefix, scan_num, keyframe_dist, query, visualize);
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Scan processing completed, about to spin");
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Node pointer: " << node.get());
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: Starting rclcpp::spin(node)...");
    
    rclcpp::spin(node);
    
    RCLCPP_WARN_STREAM(node->get_logger(), "CHECKPOINT: rclcpp::spin() returned (node shutdown)");
    
    rclcpp::shutdown();
    return 0;
}
