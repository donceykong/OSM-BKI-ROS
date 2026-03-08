/**
 * Standalone ROS2 node for loading and visualizing OSM data in RViz.
 * Loads roads and buildings from a .osm XML file and publishes them as MarkerArray.
 */

#include <rclcpp/rclcpp.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
#include "osm_visualizer.h"
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <Eigen/Dense>
#include <yaml-cpp/yaml.h>

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("osm_visualizer_node");

    node->declare_parameter<std::string>("osm_file", "");
    node->declare_parameter<double>("osm_origin_lat", 59.347671416);
    node->declare_parameter<double>("osm_origin_lon", 18.072069652);
    node->declare_parameter<double>("publish_rate", 2.0);
    node->declare_parameter<std::string>("topic", "/osm_geometries");
    node->declare_parameter<double>("tree_point_radius_meters", 5.0);
    node->declare_parameter<double>("stairs_width_meters", 1.5);
    node->declare_parameter<double>("road_width_meters", 6.0);
    node->declare_parameter<double>("sidewalk_width_meters", 2.0);
    node->declare_parameter<double>("cycleway_width_meters", 2.0);
    node->declare_parameter<double>("fence_width_meters", 0.6);
    node->declare_parameter<double>("wall_width_meters", 0.8);
    node->declare_parameter<double>("pole_point_radius_meters", 2.0);
    node->declare_parameter<std::string>("osm_confusion_matrix_file", "");
    node->declare_parameter<std::string>("config_datasets_dir", "");
    node->declare_parameter<std::string>("lidar_pose_file", "");
    node->declare_parameter<std::string>("sequence_name", "");
    node->declare_parameter<std::string>("lidar_pose_suffix", "");
    node->declare_parameter<std::string>("pose_format", "mcd");  // "mcd" = num,timestamp,x,y,z,qx,qy,qz,qw; "kitti360" = frame_index + 12 or 16 floats (3x4/4x4)
    node->declare_parameter<bool>("osm_in_sequence_dir", false);  // if true, OSM path is data_dir/sequence_name/osm_file
    node->declare_parameter<std::string>("data_dir", "");

    std::string osm_file;
    double osm_origin_lat, osm_origin_lon;
    double publish_rate;
    std::string topic;
    double tree_point_radius_meters;
    double stairs_width_meters;
    double road_width_meters;
    double sidewalk_width_meters;
    double cycleway_width_meters;
    double fence_width_meters;
    double wall_width_meters;
    double pole_point_radius_meters;
    std::string osm_confusion_matrix_file;
    std::string lidar_pose_file;
    std::string sequence_name;
    std::string lidar_pose_suffix;
    std::string data_dir;
    node->get_parameter("osm_file", osm_file);
    node->get_parameter("osm_origin_lat", osm_origin_lat);
    node->get_parameter("osm_origin_lon", osm_origin_lon);
    node->get_parameter("publish_rate", publish_rate);
    node->get_parameter("topic", topic);
    node->get_parameter("tree_point_radius_meters", tree_point_radius_meters);
    node->get_parameter("stairs_width_meters", stairs_width_meters);
    node->get_parameter("road_width_meters", road_width_meters);
    node->get_parameter("sidewalk_width_meters", sidewalk_width_meters);
    node->get_parameter("cycleway_width_meters", cycleway_width_meters);
    node->get_parameter("fence_width_meters", fence_width_meters);
    node->get_parameter("wall_width_meters", wall_width_meters);
    node->get_parameter("pole_point_radius_meters", pole_point_radius_meters);
    node->get_parameter("osm_confusion_matrix_file", osm_confusion_matrix_file);
    std::string config_datasets_dir;
    node->get_parameter("config_datasets_dir", config_datasets_dir);
    std::string pose_format;
    node->get_parameter("lidar_pose_file", lidar_pose_file);
    node->get_parameter("sequence_name", sequence_name);
    node->get_parameter("lidar_pose_suffix", lidar_pose_suffix);
    node->get_parameter("pose_format", pose_format);
    bool osm_in_sequence_dir = false;
    node->get_parameter("osm_in_sequence_dir", osm_in_sequence_dir);
    node->get_parameter("data_dir", data_dir);
    if (!sequence_name.empty() && !lidar_pose_suffix.empty()) {
        lidar_pose_file = sequence_name + "/" + lidar_pose_suffix;
    }

    if (osm_file.empty()) {
        RCLCPP_ERROR(node->get_logger(), "osm_file parameter is required. Set it in config or as launch argument.");
        return 1;
    }

    std::string full_path = osm_file;
    if (osm_file[0] != '/') {
        std::ifstream check(osm_file);
        if (!check.good()) {
            if (!data_dir.empty()) {
                if (osm_in_sequence_dir && !sequence_name.empty()) {
                    full_path = data_dir + "/" + sequence_name + "/" + osm_file;
                } else {
                    full_path = data_dir + "/" + osm_file;
                }
            } else {
                try {
                    std::string pkg_share = ament_index_cpp::get_package_share_directory("osm_bki");
                    full_path = pkg_share + "/data/mcd/" + osm_file;
                } catch (...) {}
            }
        }
    }

    RCLCPP_INFO_STREAM(node->get_logger(), "OSM Visualizer: Loading " << full_path);
    RCLCPP_INFO_STREAM(node->get_logger(), "  Origin: (" << osm_origin_lat << ", " << osm_origin_lon << "), Topic: " << topic);

    osm_bki::OSMVisualizer visualizer(node, topic);
    visualizer.setTreePointRadius(static_cast<float>(tree_point_radius_meters));
    visualizer.setPolePointRadius(static_cast<float>(pole_point_radius_meters));
    visualizer.setRoadWidth(static_cast<float>(road_width_meters));
    visualizer.setSidewalkWidth(static_cast<float>(sidewalk_width_meters));
    visualizer.setCyclewayWidth(static_cast<float>(cycleway_width_meters));
    visualizer.setFenceWidth(static_cast<float>(fence_width_meters));
    visualizer.setWallWidth(static_cast<float>(wall_width_meters));
    visualizer.setStairsWidth(static_cast<float>(stairs_width_meters));

    if (!osm_confusion_matrix_file.empty()) {
        try {
            std::string cm_path = osm_confusion_matrix_file;
            if (osm_confusion_matrix_file[0] != '/') {
                if (!config_datasets_dir.empty()) {
                    cm_path = config_datasets_dir + (config_datasets_dir.back() == '/' ? "" : "/") + osm_confusion_matrix_file;
                } else {
                    std::string pkg_share = ament_index_cpp::get_package_share_directory("osm_bki");
                    cm_path = pkg_share + "/config/datasets/" + osm_confusion_matrix_file;
                }
            }
            YAML::Node root = YAML::LoadFile(cm_path);
            if (root["osm_geometry_parameters"]) {
                auto p = root["osm_geometry_parameters"];
                if (p["tree_point_radius_meters"]) visualizer.setTreePointRadius(p["tree_point_radius_meters"].as<float>());
                if (p["pole_point_radius_meters"]) visualizer.setPolePointRadius(p["pole_point_radius_meters"].as<float>());
                if (p["road_width_meters"]) visualizer.setRoadWidth(p["road_width_meters"].as<float>());
                if (p["sidewalk_width_meters"]) visualizer.setSidewalkWidth(p["sidewalk_width_meters"].as<float>());
                if (p["cycleway_width_meters"]) visualizer.setCyclewayWidth(p["cycleway_width_meters"].as<float>());
                if (p["fence_width_meters"]) visualizer.setFenceWidth(p["fence_width_meters"].as<float>());
                if (p["wall_width_meters"]) visualizer.setWallWidth(p["wall_width_meters"].as<float>());
                if (p["stairs_width_meters"]) visualizer.setStairsWidth(p["stairs_width_meters"].as<float>());
                RCLCPP_INFO_STREAM(node->get_logger(), "Loaded OSM geometry parameters from " << cm_path);
            }
        } catch (const std::exception& e) {
            RCLCPP_WARN_STREAM(node->get_logger(), "Failed to load OSM geometry parameters: " << e.what());
        }
    }
    if (!visualizer.loadFromOSM(full_path, osm_origin_lat, osm_origin_lon)) {
        RCLCPP_ERROR(node->get_logger(), "Failed to load OSM file.");
        return 1;
    }

    // Transform OSM data to first pose origin and load lidar path if pose file is provided
    if (!lidar_pose_file.empty()) {
        std::string full_pose_path = lidar_pose_file;
        if (lidar_pose_file[0] != '/') {
            if (!data_dir.empty()) {
                full_pose_path = data_dir + "/" + lidar_pose_file;
            } else {
                try {
                    std::string pkg_share = ament_index_cpp::get_package_share_directory("osm_bki");
                    full_pose_path = pkg_share + "/data/mcd/" + lidar_pose_file;
                } catch (...) {
                    RCLCPP_WARN(node->get_logger(), "Could not determine package share directory, using pose file as-is");
                }
            }
        }
        
        RCLCPP_INFO_STREAM(node->get_logger(), "Reading poses from: " << full_pose_path << " (format: " << pose_format << ")");
        
        std::ifstream fPoses(full_pose_path);
        if (!fPoses.is_open()) {
            RCLCPP_WARN_STREAM(node->get_logger(), "Warning: Cannot open pose file " << full_pose_path << ". OSM data will not be transformed and no path will be drawn.");
        } else {
            std::vector<Eigen::Matrix4d> poses;
            if (pose_format == "kitti360") {
                // KITTI360: each line = frame_index (int) + 12 or 16 floats (3x4 or 4x4 row-major), same as scripts/kitti360/visualize_sem_map_KITTI360.py
                std::string line;
                while (std::getline(fPoses, line)) {
                    std::istringstream ss(line);
                    std::vector<double> values;
                    int frame_index = -1;
                    if (!(ss >> frame_index)) continue;
                    double v;
                    while (ss >> v) values.push_back(v);
                    if (values.size() == 12u) {
                        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                        T.block<3, 4>(0, 0) = Eigen::Map<Eigen::Matrix<double, 3, 4, Eigen::RowMajor>>(values.data());
                        poses.push_back(T);
                    } else if (values.size() == 16u) {
                        Eigen::Matrix4d T = Eigen::Map<Eigen::Matrix<double, 4, 4, Eigen::RowMajor>>(values.data());
                        poses.push_back(T);
                    }
                }
            } else {
                // MCD format: optional header, then num,timestamp,x,y,z,qx,qy,qz,qw per line
                std::string header_line;
                std::getline(fPoses, header_line);
                bool has_header = (header_line.find("num") != std::string::npos ||
                                 header_line.find("timestamp") != std::string::npos ||
                                 header_line.find("x") != std::string::npos);
                if (!has_header) {
                    fPoses.close();
                    fPoses.open(full_pose_path);
                }
                while (!fPoses.eof()) {
                    std::string s;
                    std::getline(fPoses, s);
                    if (s.empty() || s[0] == '#') continue;
                    std::stringstream ss(s);
                    std::string token;
                    std::vector<double> values;
                    char delimiter = (s.find(',') != std::string::npos) ? ',' : ' ';
                    while (std::getline(ss, token, delimiter)) {
                        try {
                            values.push_back(std::stod(token));
                        } catch (...) {}
                    }
                    if (values.size() >= 8) {
                        double x = values[2], y = values[3], z = values[4];
                        double qx = values[5], qy = values[6], qz = values[7];
                        double qw = values.size() > 8 ? values[8] : 1.0;
                        Eigen::Quaterniond quat(qw, qx, qy, qz);
                        quat.normalize();
                        Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                        T.block<3, 3>(0, 0) = quat.toRotationMatrix();
                        T(0, 3) = x; T(1, 3) = y; T(2, 3) = z;
                        poses.push_back(T);
                    }
                }
            }
            fPoses.close();
            
            if (!poses.empty()) {
                Eigen::Vector3d first_t(poses[0](0,3), poses[0](1,3), poses[0](2,3));
                RCLCPP_INFO_STREAM(node->get_logger(), "First pose - Translation: [" << first_t.x() << ", " << first_t.y() << ", " << first_t.z() << "]");

                // For KITTI-360: translation-only shift (matching Python visualize_map_osm.py).
                // For MCD: full 4x4 inverse.
                Eigen::Matrix4d first_pose_for_osm;
                if (pose_format == "kitti360") {
                    first_pose_for_osm = Eigen::Matrix4d::Identity();
                    first_pose_for_osm.block<3, 1>(0, 3) = first_t;
                } else {
                    first_pose_for_osm = poses[0];
                }
                RCLCPP_INFO(node->get_logger(), "Transforming OSM data to first pose origin (so both start at 0,0,0)...");
                visualizer.transformToFirstPoseOrigin(first_pose_for_osm);
                RCLCPP_INFO(node->get_logger(), "OSM data transformed to first pose origin.");
                
                // Build path in first-pose frame (same as OSM)
                std::vector<std::pair<float, float>> path;
                path.reserve(poses.size());
                for (const auto& T : poses) {
                    float px = static_cast<float>(T(0, 3) - first_t.x());
                    float py = static_cast<float>(T(1, 3) - first_t.y());
                    path.push_back({ px, py });
                }
                visualizer.setPath(path);
                RCLCPP_INFO_STREAM(node->get_logger(), "Lidar path loaded: " << path.size() << " poses (green polyline in RViz).");
            } else {
                RCLCPP_WARN(node->get_logger(), "Warning: No valid poses in file. OSM data will not be transformed and no path will be drawn.");
            }
        }
    } else {
        RCLCPP_INFO(node->get_logger(), "No lidar_pose_file specified. OSM data will use original coordinates and no path will be drawn.");
    }

    visualizer.publish();
    visualizer.startPeriodicPublishing(publish_rate);
    RCLCPP_INFO_STREAM(node->get_logger(), "Publishing OSM markers at " << publish_rate << " Hz");

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
