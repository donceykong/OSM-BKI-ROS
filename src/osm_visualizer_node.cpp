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
    node->declare_parameter<std::string>("lidar_pose_file", "");
    node->declare_parameter<std::string>("data_dir", "");

    std::string osm_file;
    double osm_origin_lat, osm_origin_lon;
    double publish_rate;
    std::string topic;
    double tree_point_radius_meters;
    double stairs_width_meters;
    std::string lidar_pose_file;
    std::string data_dir;
    node->get_parameter("osm_file", osm_file);
    node->get_parameter("osm_origin_lat", osm_origin_lat);
    node->get_parameter("osm_origin_lon", osm_origin_lon);
    node->get_parameter("publish_rate", publish_rate);
    node->get_parameter("topic", topic);
    node->get_parameter("tree_point_radius_meters", tree_point_radius_meters);
    node->get_parameter("stairs_width_meters", stairs_width_meters);
    node->get_parameter("lidar_pose_file", lidar_pose_file);
    node->get_parameter("data_dir", data_dir);

    if (osm_file.empty()) {
        RCLCPP_ERROR(node->get_logger(), "osm_file parameter is required. Set it in config or as launch argument.");
        return 1;
    }

    std::string full_path = osm_file;
    if (osm_file[0] != '/') {
        std::ifstream check(osm_file);
        if (!check.good()) {
            if (!data_dir.empty()) {
                full_path = data_dir + "/" + osm_file;
            } else {
                try {
                    std::string pkg_share = ament_index_cpp::get_package_share_directory("semantic_bki");
                    full_path = pkg_share + "/data/mcd/" + osm_file;
                } catch (...) {}
            }
        }
    }

    RCLCPP_INFO_STREAM(node->get_logger(), "OSM Visualizer: Loading " << full_path);
    RCLCPP_INFO_STREAM(node->get_logger(), "  Origin: (" << osm_origin_lat << ", " << osm_origin_lon << "), Topic: " << topic);

    semantic_bki::OSMVisualizer visualizer(node, topic);
    visualizer.setTreePointRadius(static_cast<float>(tree_point_radius_meters));
    visualizer.setStairsWidth(static_cast<float>(stairs_width_meters));
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
                    std::string pkg_share = ament_index_cpp::get_package_share_directory("semantic_bki");
                    full_pose_path = pkg_share + "/data/mcd/" + lidar_pose_file;
                } catch (...) {
                    RCLCPP_WARN(node->get_logger(), "Could not determine package share directory, using pose file as-is");
                }
            }
        }
        
        RCLCPP_INFO_STREAM(node->get_logger(), "Reading poses from: " << full_pose_path);
        
        std::ifstream fPoses(full_pose_path);
        if (!fPoses.is_open()) {
            RCLCPP_WARN_STREAM(node->get_logger(), "Warning: Cannot open pose file " << full_pose_path << ". OSM data will not be transformed and no path will be drawn.");
        } else {
            // Skip header line if present
            std::string header_line;
            std::getline(fPoses, header_line);
            
            bool has_header = (header_line.find("num") != std::string::npos ||
                             header_line.find("timestamp") != std::string::npos ||
                             header_line.find("x") != std::string::npos);
            
            if (!has_header) {
                fPoses.close();
                fPoses.open(full_pose_path);
            }
            
            // Read all poses
            std::vector<Eigen::Matrix4d> poses;
            while (!fPoses.eof()) {
                std::string s;
                std::getline(fPoses, s);
                
                if (s.empty() || s[0] == '#') {
                    continue;
                }
                
                std::stringstream ss(s);
                std::string token;
                std::vector<double> values;
                
                char delimiter = ',';
                if (s.find(',') == std::string::npos) {
                    delimiter = ' ';
                }
                
                while (std::getline(ss, token, delimiter)) {
                    try {
                        double val = std::stod(token);
                        values.push_back(val);
                    } catch (...) {
                        continue;
                    }
                }
                
                if (values.size() >= 8) {
                    double x = values[2];
                    double y = values[3];
                    double z = values[4];
                    double qx = values[5];
                    double qy = values[6];
                    double qz = values[7];
                    double qw = values.size() > 8 ? values[8] : 1.0;
                    
                    Eigen::Quaterniond quat(qw, qx, qy, qz);
                    quat.normalize();
                    
                    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
                    T.block<3, 3>(0, 0) = quat.toRotationMatrix();
                    T(0, 3) = x;
                    T(1, 3) = y;
                    T(2, 3) = z;
                    poses.push_back(T);
                }
            }
            fPoses.close();
            
            if (!poses.empty()) {
                Eigen::Matrix4d first_pose = poses[0];
                RCLCPP_INFO_STREAM(node->get_logger(), "First pose - Translation: [" << first_pose(0,3) << ", " << first_pose(1,3) << ", " << first_pose(2,3) << "]");
                RCLCPP_INFO(node->get_logger(), "Transforming OSM data to first pose origin (so both start at 0,0,0)...");
                visualizer.transformToFirstPoseOrigin(first_pose);
                RCLCPP_INFO(node->get_logger(), "OSM data transformed to first pose origin.");
                
                // Build path in first-pose frame (same as OSM) for debugging
                Eigen::Matrix4d first_pose_inverse = first_pose.inverse();
                std::vector<std::pair<float, float>> path;
                path.reserve(poses.size());
                for (const auto& T : poses) {
                    Eigen::Vector4d p_world(T(0, 3), T(1, 3), T(2, 3), 1.0);
                    Eigen::Vector4d p_local = first_pose_inverse * p_world;
                    path.push_back({ static_cast<float>(p_local(0)), static_cast<float>(p_local(1)) });
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
