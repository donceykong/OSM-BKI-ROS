#include <string>
#include <iostream>
#include <rclcpp/rclcpp.hpp>
#include <chrono>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include "bkioctomap.h"
#include "markerarray_pub.h"

void load_pcd(std::string filename, semantic_bki::point3f &origin, semantic_bki::PCLPointCloud &cloud) {
    pcl::PCLPointCloud2 cloud2;
    Eigen::Vector4f _origin;
    Eigen::Quaternionf orientaion;
    pcl::io::loadPCDFile(filename, cloud2, _origin, orientaion);
    pcl::fromPCLPointCloud2(cloud2, cloud);
    origin.x() = _origin[0];
    origin.y() = _origin[1];
    origin.z() = _origin[2];
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("toy_example_node");
    
    std::string map_topic_csm("/semantic_csm");
    std::string map_topic("/semantic_bki");
    std::string var_topic_csm("/semantic_csm_variance");
    std::string var_topic("/semantic_bki_variance");
    std::string dir;
    std::string prefix;
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

    // Declare parameters
    node->declare_parameter<std::string>("dir", dir);
    node->declare_parameter<std::string>("prefix", prefix);
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

    // Get parameters
    node->get_parameter<std::string>("dir", dir);
    node->get_parameter<std::string>("prefix", prefix);
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
   
    RCLCPP_INFO_STREAM(node->get_logger(), "Parameters:" << std::endl <<
            "dir: " << dir << std::endl <<
            "prefix: " << prefix << std::endl <<
            "block_depth: " << block_depth << std::endl <<
            "sf2: " << sf2 << std::endl <<
            "ell: " << ell << std::endl <<
            "prior: " << prior << std::endl <<
            "var_thresh: " << var_thresh << std::endl <<
            "free_thresh: " << free_thresh << std::endl <<
            "occupied_thresh: " << occupied_thresh << std::endl <<
            "resolution: " << resolution << std::endl <<
            "num_class: " << num_class << std::endl <<
            "free_resolution: " << free_resolution << std::endl <<
            "ds_resolution: " << ds_resolution << std::endl <<
            "scan_sum: " << scan_num << std::endl <<
            "max_range: " << max_range
            );

    /////////////////////// Semantic CSM //////////////////////
    semantic_bki::SemanticBKIOctoMap map_csm(resolution, 1, num_class, sf2, ell, prior, var_thresh, free_thresh, occupied_thresh);
    auto start = node->now();
    for (int scan_id = 1; scan_id <= scan_num; ++scan_id) {
        semantic_bki::PCLPointCloud cloud;
        semantic_bki::point3f origin;
        std::string filename(dir + "/" + prefix + "_" + std::to_string(scan_id) + ".pcd");
        load_pcd(filename, origin, cloud);
        map_csm.insert_pointcloud_csm(cloud, origin, resolution, free_resolution, max_range);
        RCLCPP_INFO_STREAM(node->get_logger(), "Scan " << scan_id << " done");
    }
    auto end = node->now();
    rclcpp::Duration elapsed = end - start;
    RCLCPP_INFO_STREAM(node->get_logger(), "Semantic CSM finished in " << elapsed.seconds() << "s");

    /////////////////////// Publish Map //////////////////////
    float max_var = std::numeric_limits<float>::min();
    float min_var = std::numeric_limits<float>::max(); 
    semantic_bki::MarkerArrayPub m_pub_csm(node, map_topic_csm, resolution);
    for (auto it = map_csm.begin_leaf(); it != map_csm.end_leaf(); ++it) {
        if (it.get_node().get_state() == semantic_bki::State::OCCUPIED) {
            semantic_bki::point3f p = it.get_loc();
            int semantics = it.get_node().get_semantics();
            m_pub_csm.insert_point3d_semantics(p.x(), p.y(), p.z(), it.get_size(), semantics, 0);
            std::vector<float> vars(num_class);
            it.get_node().get_vars(vars);
            if (vars[semantics] > max_var)
		          max_var = vars[semantics];
		        if (vars[semantics] < min_var)
		          min_var = vars[semantics];
        }
    }
    m_pub_csm.publish();
    std::cout << "max_var: " << max_var << std::endl;
    std::cout << "min_var: " << min_var << std::endl;
    
    /////////////////////// Variance Map //////////////////////
    semantic_bki::MarkerArrayPub v_pub_csm(node, var_topic_csm, resolution);
    for (auto it = map_csm.begin_leaf(); it != map_csm.end_leaf(); ++it) {
        if (it.get_node().get_state() == semantic_bki::State::OCCUPIED) {
            semantic_bki::point3f p = it.get_loc();
            int semantics = it.get_node().get_semantics();
            std::vector<float> vars(num_class);
            it.get_node().get_vars(vars);
            v_pub_csm.insert_point3d_variance(p.x(), p.y(), p.z(), min_var, max_var, it.get_size(), vars[semantics]);
        }
    }
    v_pub_csm.publish();

    
    /////////////////////// Semantic BKI //////////////////////
    semantic_bki::SemanticBKIOctoMap map(resolution, block_depth, num_class, sf2, ell, prior, var_thresh, free_thresh, occupied_thresh);
    start = node->now();
    for (int scan_id = 1; scan_id <= scan_num; ++scan_id) {
        semantic_bki::PCLPointCloud cloud;
        semantic_bki::point3f origin;
        std::string filename(dir + "/" + prefix + "_" + std::to_string(scan_id) + ".pcd");
        load_pcd(filename, origin, cloud);
        map.insert_pointcloud(cloud, origin, resolution, free_resolution, max_range);
        RCLCPP_INFO_STREAM(node->get_logger(), "Scan " << scan_id << " done");
    }
    end = node->now();
    elapsed = end - start;
    RCLCPP_INFO_STREAM(node->get_logger(), "Semantic BKI finished in " << elapsed.seconds() << "s");
 
    
    /////////////////////// Publish Map //////////////////////
    max_var = std::numeric_limits<float>::min();
    min_var = std::numeric_limits<float>::max(); 
    semantic_bki::MarkerArrayPub m_pub(node, map_topic, resolution);
    for (auto it = map.begin_leaf(); it != map.end_leaf(); ++it) {
        if (it.get_node().get_state() == semantic_bki::State::OCCUPIED) {
            semantic_bki::point3f p = it.get_loc();
            int semantics = it.get_node().get_semantics();
            m_pub.insert_point3d_semantics(p.x(), p.y(), p.z(), it.get_size(), semantics, 0);
            std::vector<float> vars(num_class);
            it.get_node().get_vars(vars);
            if (vars[semantics] > max_var)
		          max_var = vars[semantics];
		        if (vars[semantics] < min_var)
		          min_var = vars[semantics];
        }
    }
    m_pub.publish();
    std::cout << "max_var: " << max_var << std::endl;
    std::cout << "min_var: " << min_var << std::endl;

    /////////////////////// Variance Map //////////////////////
    semantic_bki::MarkerArrayPub v_pub(node, var_topic, resolution);
    for (auto it = map.begin_leaf(); it != map.end_leaf(); ++it) {
        if (it.get_node().get_state() == semantic_bki::State::OCCUPIED) {
            semantic_bki::point3f p = it.get_loc();
            int semantics = it.get_node().get_semantics();
            std::vector<float> vars(num_class);
            it.get_node().get_vars(vars);
            v_pub.insert_point3d_variance(p.x(), p.y(), p.z(), min_var, max_var, it.get_size(), vars[semantics]);
        }
    }
    v_pub.publish();

    rclcpp::spin(node);
    rclcpp::shutdown();

    return 0;
}
