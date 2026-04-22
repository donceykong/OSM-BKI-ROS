#include <string>
#include <iostream>
#include <memory>
#include <cstdlib>
#include <fstream>

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <yaml-cpp/yaml.h>

#include "bkioctomap.h"
#include "markerarray_pub.h"
#include "mcd_util.h"
#include "osm_visualizer.h"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("kitti360_node");

    if (!node) {
        RCLCPP_ERROR_STREAM(rclcpp::get_logger("kitti360_node"), "Failed to create ROS2 node!");
        return 1;
    }

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

    std::string dir;
    std::string input_data_prefix;
    std::string input_label_prefix;
    std::string lidar_pose_file;
    std::string gt_label_prefix;
    std::string evaluation_result_prefix;
    bool query = false;
    bool publish_semantic_occ_map = false;

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
    node->declare_parameter<bool>("publish_semantic_occ_map", publish_semantic_occ_map);
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
    node->declare_parameter<std::string>("inferred_labels_key", "kitti360");
    node->declare_parameter<std::string>("gt_labels_key", "kitti360");
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
    node->declare_parameter<std::vector<double>>("height_kernel_dead_zone", std::vector<double>{});
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
    node->declare_parameter<double>("osm_prior_map_z", 0.0);
    node->declare_parameter<bool>("publish_osm_converted_map", false);
    node->declare_parameter<std::string>("osm_converted_map_topic", "/semantic_osm_converted_map");
    node->declare_parameter<bool>("publish_variance", false);
    node->declare_parameter<std::string>("variance_topic", "/osm_bki_variance");
    node->declare_parameter<bool>("publish_semantic_uncertainty", false);
    node->declare_parameter<std::string>("semantic_uncertainty_topic", "/semantic_uncertainty_cloud");
    node->declare_parameter<bool>("use_common_taxonomy", true);

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
    if (!sequence_name.empty() && !input_data_suffix.empty()) {
        input_data_prefix = sequence_name + "/" + input_data_suffix;
        if (!lidar_pose_suffix.empty()) lidar_pose_file = sequence_name + "/" + lidar_pose_suffix;
        if (!input_label_suffix.empty()) input_label_prefix = sequence_name + "/" + input_label_suffix;
        if (!gt_label_suffix.empty()) gt_label_prefix = sequence_name + "/" + gt_label_suffix;
        if (!evaluation_result_prefix.empty()) evaluation_result_prefix = sequence_name + "/" + evaluation_result_prefix;
    }
    node->get_parameter<bool>("query", query);
    node->get_parameter<bool>("publish_semantic_occ_map", publish_semantic_occ_map);

    std::string colors_file, calibration_file, osm_file, config_datasets_dir;
    double osm_origin_lat, osm_origin_lon, osm_decay_meters, osm_tree_point_radius_meters;
    node->get_parameter<std::string>("colors_file", colors_file);
    node->get_parameter<std::string>("calibration_file", calibration_file);
    node->get_parameter<std::string>("config_datasets_dir", config_datasets_dir);
    node->get_parameter<std::string>("osm_file", osm_file);
    node->get_parameter<double>("osm_origin_lat", osm_origin_lat);
    node->get_parameter<double>("osm_origin_lon", osm_origin_lon);
    node->get_parameter<double>("osm_decay_meters", osm_decay_meters);
    node->get_parameter<double>("osm_tree_point_radius_meters", osm_tree_point_radius_meters);

    RCLCPP_INFO_STREAM(node->get_logger(), "KITTI360: dir=" << dir << ", lidar_pose_file=" << lidar_pose_file << ", calibration_file=" << (calibration_file.empty() ? "(identity)" : calibration_file));

    // No static TF for KITTI360 (poses are already in world frame, no base-to-lidar TF needed).

    MCDData mcd_data(node, resolution, block_depth, sf2, ell, num_class, free_thresh, occupied_thresh, var_thresh, ds_resolution, free_resolution, max_range, map_topic, prior);

    std::string pose_path = dir + "/" + lidar_pose_file;
    RCLCPP_INFO_STREAM(node->get_logger(), "Reading KITTI360 poses from: " << pose_path);
    if (!mcd_data.read_lidar_poses_kitti360(pose_path)) {
        RCLCPP_ERROR_STREAM(node->get_logger(), "Failed to read KITTI360 poses!");
        return 1;
    }

    if (!mcd_data.load_calibration_from_params()) {
        RCLCPP_FATAL(node->get_logger(), "Failed to load calibration (use empty calibration_file for identity).");
        return 1;
    }

    std::string pkg_config_datasets;
    if (!config_datasets_dir.empty()) {
        pkg_config_datasets = config_datasets_dir;
        if (pkg_config_datasets.back() != '/') pkg_config_datasets += '/';
    } else {
        pkg_config_datasets = ament_index_cpp::get_package_share_directory("osm_bki") + "/config/datasets/";
    }

    bool use_common_taxonomy = true;
    node->get_parameter<bool>("use_common_taxonomy", use_common_taxonomy);
    if (use_common_taxonomy) {
        std::string inferred_labels_key, gt_labels_key;
        node->get_parameter<std::string>("inferred_labels_key", inferred_labels_key);
        node->get_parameter<std::string>("gt_labels_key", gt_labels_key);
        std::string common_label_path = pkg_config_datasets + "labels_common.yaml";
        if (!mcd_data.load_common_label_config(common_label_path, inferred_labels_key, gt_labels_key)) {
            RCLCPP_FATAL_STREAM(node->get_logger(), "Failed to load common label config from " << common_label_path);
            return 1;
        }
    } else {
        RCLCPP_INFO(node->get_logger(), "use_common_taxonomy=false: using network class indices (set num_class to network n_classes)");
    }
    auto resolve_label_config_path = [&](const std::string& labels_key) -> std::string {
        std::string f = (labels_key == "mcd") ? "labels_mcd.yaml" : (labels_key == "kitti360") ? "labels_kitti360.yaml" : "labels_semkitti.yaml";
        return pkg_config_datasets + f;
    };

    bool inferred_use_multiclass = false, gt_use_multiclass = false;
    std::string inferred_labels_key_val, gt_labels_key_val;
    node->get_parameter<bool>("inferred_use_multiclass", inferred_use_multiclass);
    node->get_parameter<bool>("gt_use_multiclass", gt_use_multiclass);
    node->get_parameter<std::string>("inferred_labels_key", inferred_labels_key_val);
    node->get_parameter<std::string>("gt_labels_key", gt_labels_key_val);

    if (inferred_use_multiclass) {
        mcd_data.set_inferred_multiclass_mode(true, dir + "/" + input_label_prefix);
        if (!mcd_data.load_label_config(resolve_label_config_path(inferred_labels_key_val))) {
            RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load inferred label config.");
        }
        bool use_uncertainty_filter = false;
        std::string confusion_matrix_file, uncertainty_filter_mode;
        double uncertainty_drop_percent, uncertainty_min_weight;
        node->get_parameter<bool>("use_uncertainty_filter", use_uncertainty_filter);
        node->get_parameter<std::string>("confusion_matrix_file", confusion_matrix_file);
        node->get_parameter<std::string>("uncertainty_filter_mode", uncertainty_filter_mode);
        node->get_parameter<double>("uncertainty_drop_percent", uncertainty_drop_percent);
        node->get_parameter<double>("uncertainty_min_weight", uncertainty_min_weight);
        mcd_data.set_uncertainty_filter(use_uncertainty_filter, inferred_labels_key_val, uncertainty_filter_mode,
                                        static_cast<float>(uncertainty_drop_percent), static_cast<float>(uncertainty_min_weight));
        if (use_uncertainty_filter && !confusion_matrix_file.empty()) {
            std::string cm_path = pkg_config_datasets + confusion_matrix_file;
            if (!mcd_data.load_confusion_matrix(cm_path)) {
                RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load confusion matrix.");
            }
        }
    }

    if (gt_use_multiclass) {
        mcd_data.set_gt_multiclass_mode(true);
        if (!mcd_data.load_gt_label_config(resolve_label_config_path(gt_labels_key_val))) {
            RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load GT label config.");
        }
    }

    if (!colors_file.empty()) {
        std::string colors_file_path = pkg_config_datasets + colors_file;
        if (!mcd_data.load_colors_from_yaml(colors_file_path)) {
            RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load colors from " << colors_file_path);
        }
    }

    mcd_data.set_color_mode(osm_bki::MapColorMode::Semantic);
    mcd_data.set_osm_decay_meters(static_cast<float>(osm_decay_meters));
    mcd_data.set_osm_tree_point_radius(static_cast<float>(osm_tree_point_radius_meters));

    if (!osm_file.empty()) {
        std::string full_osm_path = osm_file;
        if (osm_file[0] != '/' && !dir.empty()) {
            if (!sequence_name.empty())
                full_osm_path = dir + "/" + sequence_name + "/" + osm_file;
            else
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
            RCLCPP_INFO_STREAM(node->get_logger(), "Loaded OSM from " << full_osm_path);
        } else {
            RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load OSM: " << full_osm_path);
        }
    }

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
        bool publish_osm_prior_map = false;
        std::string osm_prior_map_color_mode_str = "osm_blend";
        std::string osm_prior_map_topic = "/semantic_osm_prior_map";
        node->get_parameter<bool>("publish_osm_prior_map", publish_osm_prior_map);
        node->get_parameter<std::string>("osm_prior_map_color_mode", osm_prior_map_color_mode_str);
        node->get_parameter<std::string>("osm_prior_map_topic", osm_prior_map_topic);
        osm_bki::MapColorMode osm_color_mode = osm_bki::MapColorMode::OSMBlend;
        if (osm_prior_map_color_mode_str == "osm_building") osm_color_mode = osm_bki::MapColorMode::OSMBuilding;
        else if (osm_prior_map_color_mode_str == "osm_road") osm_color_mode = osm_bki::MapColorMode::OSMRoad;
        else if (osm_prior_map_color_mode_str == "osm_grassland") osm_color_mode = osm_bki::MapColorMode::OSMGrassland;
        else if (osm_prior_map_color_mode_str == "osm_tree") osm_color_mode = osm_bki::MapColorMode::OSMTree;
        else if (osm_prior_map_color_mode_str == "osm_parking") osm_color_mode = osm_bki::MapColorMode::OSMParking;
        else if (osm_prior_map_color_mode_str == "osm_fence") osm_color_mode = osm_bki::MapColorMode::OSMFence;
        else if (osm_prior_map_color_mode_str == "osm_sidewalk") osm_color_mode = osm_bki::MapColorMode::OSMSidewalk;
        else if (osm_prior_map_color_mode_str == "osm_cycleway") osm_color_mode = osm_bki::MapColorMode::OSMCycleway;
        else if (osm_prior_map_color_mode_str == "osm_forest") osm_color_mode = osm_bki::MapColorMode::OSMForest;
        double osm_prior_map_z = 0.0;
        node->get_parameter<double>("osm_prior_map_z", osm_prior_map_z);
        mcd_data.set_publish_osm_prior_map(publish_osm_prior_map, osm_prior_map_topic, osm_color_mode,
                                           static_cast<float>(osm_prior_map_z));

        bool publish_osm_converted_map = false;
        std::string osm_converted_map_topic = "/semantic_osm_converted_map";
        node->get_parameter<bool>("publish_osm_converted_map", publish_osm_converted_map);
        node->get_parameter<std::string>("osm_converted_map_topic", osm_converted_map_topic);
        mcd_data.set_publish_osm_converted_map(publish_osm_converted_map, osm_converted_map_topic);

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
        mcd_data.set_osm_height_filter_enabled(osm_height_filtering);
        mcd_data.set_publish_height_bins(publish_osm_height_bins_scan, publish_osm_height_bins_map,
                                         static_cast<float>(osm_height_bins_step_meters),
                                         static_cast<float>(osm_height_bins_map_leaf_size),
                                         osm_height_bins_scan_topic, osm_height_bins_map_topic);

        // Height filter mode selection.
        std::string height_filter_type = "discreet";
        node->get_parameter<std::string>("height_filter_type", height_filter_type);
        bool use_gaussian_height = osm_height_filtering && (height_filter_type == "gaussian");
        mcd_data.set_height_filter_mode_gaussian(use_gaussian_height);
        if (use_gaussian_height) {
            double hk_lambda = 0.0, hk_gate = 0.0, sensor_height = 0.0;
            bool hk_redistribute = false;
            std::vector<double> mu_d, tau_d, dz_d;
            node->get_parameter<double>("height_kernel_lambda", hk_lambda);
            node->get_parameter<std::vector<double>>("height_kernel_dead_zone", dz_d);
            node->get_parameter<bool>("height_kernel_redistribute", hk_redistribute);
            node->get_parameter<double>("height_kernel_gate", hk_gate);
            node->get_parameter<double>("sensor_mounting_height", sensor_height);
            node->get_parameter<std::vector<double>>("height_kernel_mu", mu_d);
            node->get_parameter<std::vector<double>>("height_kernel_tau", tau_d);
            std::vector<float> mu_f(mu_d.begin(), mu_d.end());
            std::vector<float> tau_f(tau_d.begin(), tau_d.end());
            std::vector<float> dz_f(dz_d.begin(), dz_d.end());
            mcd_data.set_height_kernel_params(static_cast<float>(hk_lambda), mu_f, tau_f,
                                              dz_f, hk_redistribute,
                                              static_cast<float>(hk_gate),
                                              static_cast<float>(sensor_height));
            RCLCPP_INFO_STREAM(node->get_logger(),
                "Height filter mode: gaussian (lambda=" << hk_lambda
                << ", dead_zone=[" << dz_f.size() << " per-class]"
                << ", gate=" << hk_gate
                << ", sensor_height=" << sensor_height
                << ", " << mu_f.size() << " mu / " << tau_f.size() << " tau)");
        } else if (osm_height_filtering) {
            RCLCPP_INFO_STREAM(node->get_logger(), "Height filter mode: discreet (CM-based)");
        }

        // Widths and decay come from ROS params (osm_geometry_parameters.*) applied before
        // loadFromOSM; do not re-load from the confusion-matrix yaml here.
        if (!osm_cm_file.empty() && osm_prior_str > 0.0) {
            std::string cm_path = pkg_config_datasets + osm_cm_file;
            if (mcd_data.load_osm_confusion_matrix(cm_path)) {
                RCLCPP_INFO_STREAM(node->get_logger(), "Loaded OSM confusion matrix from " << cm_path);
            } else {
                RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load OSM confusion matrix from " << cm_path);
            }
            if (osm_height_filtering && !use_gaussian_height) {
                if (mcd_data.load_osm_height_confusion_matrix(cm_path)) {
                    RCLCPP_INFO_STREAM(node->get_logger(), "Loaded OSM height confusion matrix from " << cm_path);
                } else {
                    RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load OSM height confusion matrix; height filtering disabled");
                }
            }
        }
    }

    mcd_data.set_up_evaluation(dir + "/" + gt_label_prefix, dir + "/" + evaluation_result_prefix);
    mcd_data.process_scans(dir + "/" + input_data_prefix, dir + "/" + input_label_prefix, scan_num, keyframe_dist, query, publish_semantic_occ_map);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
