#include <string>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "bkioctomap.h"
#include "markerarray_pub.h"
#include "semantickitti_util.h"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("semantickitti_node");

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
    
    // SemanticKITTI
    std::string dir;
    std::string input_data_prefix;
    std::string input_label_prefix;
    std::string lidar_pose_file;
    std::string gt_label_prefix;
    std::string evaluation_result_prefix;
    bool query = false;
    bool visualize = false;

    // Declare parameters
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
    node->declare_parameter<std::string>("dir", dir);
    node->declare_parameter<std::string>("input_data_prefix", input_data_prefix);
    node->declare_parameter<std::string>("input_label_prefix", input_label_prefix);
    node->declare_parameter<std::string>("lidar_pose_file", lidar_pose_file);
    node->declare_parameter<std::string>("gt_label_prefix", gt_label_prefix);
    node->declare_parameter<std::string>("evaluation_result_prefix", evaluation_result_prefix);
    node->declare_parameter<bool>("query", query);
    node->declare_parameter<bool>("visualize", visualize);
    node->declare_parameter<std::string>("inferred_labels_key", "semkitti");
    node->declare_parameter<std::string>("gt_labels_key", "semkitti");
    node->declare_parameter<std::string>("colors_file", "");

    // Get parameters
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
    node->get_parameter<std::string>("dir", dir);
    node->get_parameter<std::string>("input_data_prefix", input_data_prefix);
    node->get_parameter<std::string>("input_label_prefix", input_label_prefix);
    node->get_parameter<std::string>("lidar_pose_file", lidar_pose_file);
    node->get_parameter<std::string>("gt_label_prefix", gt_label_prefix);
    node->get_parameter<std::string>("evaluation_result_prefix", evaluation_result_prefix);
    node->get_parameter<bool>("query", query);
    node->get_parameter<bool>("visualize", visualize);

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

      "SemanticKITTI:" << std::endl <<
      "dir: " << dir << std::endl <<
      "input_data_prefix: " << input_data_prefix << std::endl <<
      "input_label_prefix: " << input_label_prefix << std::endl <<
      "lidar_pose_file: " << lidar_pose_file << std::endl <<
      "gt_label_prefix: " << gt_label_prefix << std::endl <<
      "evaluation_result_prefix: " << evaluation_result_prefix << std::endl <<
      "query: " << query << std::endl <<
      "visualize:" << visualize
      );

    
    ///////// Build Map /////////////////////
    SemanticKITTIData semantic_kitti_data(node, resolution, block_depth, sf2, ell, num_class, free_thresh, occupied_thresh, var_thresh, ds_resolution, free_resolution, max_range, map_topic, prior);
    semantic_kitti_data.read_lidar_poses(dir + '/' + lidar_pose_file);
    semantic_kitti_data.set_up_evaluation(dir + '/' + gt_label_prefix, dir + '/' + evaluation_result_prefix);

    // Load common taxonomy label mappings
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
      if (!semantic_kitti_data.load_common_label_config(common_label_path, inferred_labels_key, gt_labels_key)) {
        RCLCPP_FATAL_STREAM(node->get_logger(),
            "Failed to load common label config from " << common_label_path);
        return 1;
      }
    }

    // Load colors
    {
      std::string colors_file;
      node->get_parameter<std::string>("colors_file", colors_file);
      if (!colors_file.empty()) {
        std::string colors_path;
        size_t dp = dir.rfind("/data/");
        if (dp != std::string::npos) {
          colors_path = dir.substr(0, dp) + "/config/datasets/" + colors_file;
        } else {
          colors_path = ament_index_cpp::get_package_share_directory("semantic_bki")
                        + "/config/datasets/" + colors_file;
        }
        if (semantic_kitti_data.load_colors_from_yaml(colors_path)) {
          RCLCPP_INFO_STREAM(node->get_logger(), "Loaded colors from " << colors_path);
        } else {
          RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load colors from " << colors_path << ". Using defaults.");
          semantic_kitti_data.load_colors_from_params();
        }
      } else {
        semantic_kitti_data.load_colors_from_params();
      }
    }

    semantic_kitti_data.process_scans(dir + '/' + input_data_prefix, dir + '/' + input_label_prefix, scan_num, query, visualize);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
