#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <geometry_msgs/msg/point.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <rclcpp/rclcpp.hpp>
#include <yaml-cpp/yaml.h>

#include <cmath>
#include <algorithm>
#include <string>
#include <map>
#include <fstream>

namespace osm_bki {

    double interpolate( double val, double y0, double x0, double y1, double x1 ) {
        return (val-x0)*(y1-y0)/(x1-x0) + y0;
    }

    double base( double val ) {
        if ( val <= -0.75 ) return 0;
        else if ( val <= -0.25 ) return interpolate( val, 0.0, -0.75, 1.0, -0.25 );
        else if ( val <= 0.25 ) return 1.0;
        else if ( val <= 0.75 ) return interpolate( val, 1.0, 0.25, 0.0, 0.75 );
        else return 0.0;
    }

    double red( double gray ) {
        return base( gray - 0.5 );
    }
    double green( double gray ) {
        return base( gray );
    }
    double blue( double gray ) {
        return base( gray + 0.5 );
    }
    
    std_msgs::msg::ColorRGBA JetMapColor(float gray) {
      std_msgs::msg::ColorRGBA color;
      color.a = 1.0;

      color.r = red(gray);
      color.g = green(gray);
      color.b = blue(gray);
      return color;
    }

    std_msgs::msg::ColorRGBA SemanticMapColor(int c) {
      std_msgs::msg::ColorRGBA color;
      color.a = 1.0;

      switch (c) {
        case 1:
          color.r = 1;
          color.g = 0;
          color.b = 0;
          break;
        case 2:
          color.r = 70.0/255;
          color.g = 130.0/255;
          color.b = 180.0/255;
          break;
        case 3:
          color.r = 218.0/255;
          color.g = 112.0/255;
          color.b = 214.0/255;
          break;
        default:
          color.r = 1;
          color.g = 1;
          color.b = 1;
          break;
      }

      return color;
    }

    std_msgs::msg::ColorRGBA SemanticKITTISemanticMapColor(int c) {
      std_msgs::msg::ColorRGBA color;
      color.a = 1.0;

      switch (c) {
        case 1:  // car
          color.r = 245.0 / 255;
          color.g = 150.0 / 255;
          color.b = 100.0 / 255;
          break;
        case 2:  // bicycle
          color.r = 245.0 / 255;
          color.g = 230.0 / 255;
          color.b = 100.0 / 255;
          break;
        case 3:  // motorcycle
          color.r = 150.0 / 255;
          color.g = 60.0 / 255;
          color.b = 30.0 / 255;
          break;
        case 4:  // truck
          color.r = 180.0 / 255;
          color.g = 30.0 / 255;
          color.b = 80.0 / 255;
          break;
        case 5:  // other-vehicle
          color.r = 255.0 / 255;
          color.g = 80.0 / 255;
          color.b = 100.0 / 255;
          break;
        case 6:  // person
          color.r = 30.0 / 255;
          color.g = 30.0 / 255;
          color.b = 1;
          break;
        case 7:  // bicyclist
          color.r = 200.0 / 255;
          color.g = 40.0 / 255;
          color.b = 1;
          break;
        case 8:  // motorcyclist
          color.r = 90.0 / 255;
          color.g = 30.0 / 255;
          color.b = 150.0 / 255;
          break;
        case 9:  // road
          color.r = 1;
          color.g = 0;
          color.b = 1;
          break;
        case 10: // parking
          color.r = 1;
          color.g = 150.0 / 255;
          color.b = 1;
          break;
        case 11: // sidewalk
          color.r = 75.0 / 255;
          color.g = 0;
          color.b = 75.0 / 255;
          break;
        case 12: // other-ground
          color.r = 75.0 / 255;
          color.g = 0;
          color.b = 175.0 / 255;
          break;
        case 13: // building
          color.r = 0;
          color.g = 200.0 / 255;
          color.b = 1;
          break;
        case 14: // fence
          color.r = 50.0 / 255;
          color.g = 120.0 / 255;
          color.b = 1;
          break;
        case 15: // vegetation
          color.r = 0;
          color.g = 175.0 / 255;
          color.b = 0;
          break;
        case 16: // trunk
          color.r = 0;
          color.g = 60.0 / 255;
          color.b = 135.0 / 255;
          break;
        case 17: // terrain
          color.r = 80.0 / 255;
          color.g = 240.0 / 255;
          color.b = 150.0 / 255;
          break;
        case 18: // pole
          color.r = 150.0 / 255;
          color.g = 240.0 / 255;
          color.b = 1;
          break;
        case 19: // traffic-sign
          color.r = 0;
          color.g = 0;
          color.b = 1;
          break;
        default:
          color.r = 1;
          color.g = 1;
          color.b = 1;
          break;
      }
      return color;
    }

    std_msgs::msg::ColorRGBA NCLTSemanticMapColor(int c) {
      std_msgs::msg::ColorRGBA color;
      color.a = 1.0;

      switch (c) {
        case 1:  // water
          color.r = 30.0 / 255;
          color.g = 144.0 / 255;
          color.b = 250.0 / 255;
          break;
        case 2:  // road
          color.r = 250.0 / 255;
          color.g = 250.0 / 255;
          color.b = 250.0 / 255;
          break;
        case 3:  // sidewalk
          color.r = 128.0 / 255;
          color.g = 64.0 / 255;
          color.b = 128.0 / 255;
          break;
        case 4:  // terrain
          color.r = 128.0 / 255;
          color.g = 128.0 / 255;
          color.b = 0;
          break;
        case 5:  // building
          color.r = 250.0 / 255;
          color.g = 128.0 / 255;
          color.b = 0;
          break;
        case 6:  // vegetation
          color.r = 107.0 / 255;
          color.g = 142.0/ 255;
          color.b = 35.0 / 255;
          break;
        case 7:  // car
          color.r = 0;
          color.g = 0;
          color.b = 142.0 / 255;
          break;
        case 8:  // person
          color.r = 220.0 / 255;
          color.g = 20.0 / 255;
          color.b = 60.0 / 255;
          break;
        case 9:  // bike
          color.r = 119.0 / 255;
          color.g = 11.0 / 255;
          color.b = 32.0/ 255;
          break;
        case 10:  // pole
          color.r = 192.0 / 255;
          color.g = 192.0 / 255;
          color.b = 192.0 / 255;
          break;
        case 11:  // stair
          color.r = 123.0 / 255;
          color.g = 104.0 / 255;
          color.b = 238.0 / 255;
          break;
        case 12:  // traffic sign
          color.r = 250.0 / 255;
          color.g = 250.0 / 255;
          color.b = 0;
          break;
        case 13:  // sky
          color.r = 135.0 / 255;
          color.g = 206.0 / 255;
          color.b = 235.0 / 255;
          break;
        default:
          color.r = 1;
          color.g = 1;
          color.b = 1;
          break;
      }
      return color;
    }

    std_msgs::msg::ColorRGBA KITTISemanticMapColor(int c) {
      std_msgs::msg::ColorRGBA color;
      color.a = 1.0;
      
      switch (c) {
        case 1:  // building
          color.r = 128.0 / 255;
          color.g = 0;
          color.b = 0;
          break;
        case 2:  // sky
          color.r = 128.0 / 255;
          color.g = 128.0 / 255;
          color.b = 128.0 / 255;
          break;
        case 3:  // road
          color.r = 128.0 / 255;
          color.g = 64.0  / 255;
          color.b = 128.0 / 255;
          break;
        case 4:  // vegetation
          color.r = 128.0 / 255;
          color.g = 128.0 / 255;
          color.b = 0;
          break;
        case 5:  // sidewalk
          color.r = 0;
          color.g = 0;
          color.b = 192.0 / 255;
          break;
        case 6:  // car
          color.r = 64.0 / 255;
          color.g = 0;
          color.b = 128.0 / 255;
          break;
        case 7:  // pedestrain
          color.r = 64.0 / 255;
          color.g = 64.0 / 255;
          color.b = 0;
          break;
        case 8:  // cyclist
          color.r = 0;
          color.g = 128.0 / 255;
          color.b = 192.0 / 255;
          break;
        case 9:  // signate
          color.r = 192.0 / 255;
          color.g = 128.0 / 255;
          color.b = 128.0 / 255;
          break;
        case 10: // fense
          color.r = 64.0  / 255;
          color.g = 64.0  / 255;
          color.b = 128.0 / 255;
          break;
        case 11: // pole
          color.r = 192.0 / 255;
          color.g = 192.0 / 255;
          color.b = 128.0 / 255;
          break;
        default:
          color.r = 1;
          color.g = 1;
          color.b = 1;
          break;
      }

      return color;
    }


    std_msgs::msg::ColorRGBA heightMapColor(double h) {

        std_msgs::msg::ColorRGBA color;
        color.a = 1.0;
        // blend over HSV-values (more colors)

        double s = 1.0;
        double v = 1.0;

        h -= floor(h);
        h *= 6;
        int i;
        double m, n, f;

        i = floor(h);
        f = h - i;
        if (!(i & 1))
            f = 1 - f; // if i is even
        m = v * (1 - s);
        n = v * (1 - s * f);

        switch (i) {
            case 6:
            case 0:
                color.r = v;
                color.g = n;
                color.b = m;
                break;
            case 1:
                color.r = n;
                color.g = v;
                color.b = m;
                break;
            case 2:
                color.r = m;
                color.g = v;
                color.b = n;
                break;
            case 3:
                color.r = m;
                color.g = n;
                color.b = v;
                break;
            case 4:
                color.r = n;
                color.g = m;
                color.b = v;
                break;
            case 5:
                color.r = v;
                color.g = m;
                color.b = n;
                break;
            default:
                color.r = 1;
                color.g = 0.5;
                color.b = 0.5;
                break;
        }

        return color;
    }

    /// Visualization mode: semantic class (from inference) or OSM prior (single or blended).
    enum class MapColorMode {
        Semantic,
        OSMBuilding,
        OSMRoad,
        OSMGrassland,
        OSMTree,
        OSMParking,
        OSMFence,
        OSMBlend   /// All six OSM priors at once; overlapping weights blend colors
    };

    /// Color for OSM prior value in [0,1]: blend from light gray (0) to class color (1).
    /// prior_type: 0=building, 1=road, 2=grassland, 3=tree, 4=parking, 5=fence.
    inline std_msgs::msg::ColorRGBA OSMPriorMapColor(int prior_type, float value) {
        std_msgs::msg::ColorRGBA color;
        color.a = 1.0;
        float g = 0.85f;  // light gray at 0
        float r = g, b = g;
        float cr = 0, cg = 0, cb = 0;
        switch (prior_type) {
            case 0: cr = 0.0f;  cg = 0.0f;  cb = 1.0f; break;   // building blue
            case 1: cr = 1.0f;  cg = 0.0f;  cb = 0.0f; break;   // road red
            case 2: cr = 0.4f;  cg = 0.85f; cb = 0.35f; break;  // grassland green
            case 3: cr = 0.1f;  cg = 0.5f;  cb = 0.2f; break;   // tree dark green
            case 4: cr = 1.0f;  cg = 0.65f; cb = 0.0f; break;   // parking orange
            case 5: cr = 0.5f;  cg = 0.45f; cb = 0.4f; break;   // fence gray/brown
            default: cr = cg = cb = 0.5f; break;
        }
        value = std::max(0.f, std::min(1.f, value));
        color.r = (1.f - value) * r + value * cr;
        color.g = (1.f - value) * g + value * cg;
        color.b = (1.f - value) * b + value * cb;
        return color;
    }

    /// Blend the six OSM prior colors by weight. Weights in [0,1]. If all zero, returns light gray.
    inline std_msgs::msg::ColorRGBA OSMPriorBlendColor(float w_building, float w_road, float w_grassland, float w_tree, float w_parking, float w_fence) {
        std_msgs::msg::ColorRGBA color;
        color.a = 1.0;
        const float gray = 0.85f;
        w_building  = std::max(0.f, std::min(1.f, w_building));
        w_road      = std::max(0.f, std::min(1.f, w_road));
        w_grassland = std::max(0.f, std::min(1.f, w_grassland));
        w_tree      = std::max(0.f, std::min(1.f, w_tree));
        w_parking   = std::max(0.f, std::min(1.f, w_parking));
        w_fence     = std::max(0.f, std::min(1.f, w_fence));
        float sum = w_building + w_road + w_grassland + w_tree + w_parking + w_fence;
        if (sum <= 0.f) {
            color.r = color.g = color.b = gray;
            return color;
        }
        // Weighted blend: building=blue, road=red, grassland=light green, tree=dark green, parking=orange, fence=gray
        color.r = (w_building * 0.f   + w_road * 1.f + w_grassland * 0.4f + w_tree * 0.1f + w_parking * 1.0f + w_fence * 0.5f) / sum;
        color.g = (w_building * 0.f   + w_road * 0.f + w_grassland * 0.85f + w_tree * 0.5f + w_parking * 0.65f + w_fence * 0.45f) / sum;
        color.b = (w_building * 1.f  + w_road * 0.f + w_grassland * 0.35f + w_tree * 0.2f + w_parking * 0.f + w_fence * 0.4f) / sum;
        return color;
    }

    class MarkerArrayPub {
        typedef pcl::PointXYZ PointType;
        typedef pcl::PointCloud<PointType> PointCloud;
    public:
        MarkerArrayPub(rclcpp::Node::SharedPtr node, std::string topic, float resolution) 
            : node_(node),
              markerarray_frame_id_("map"),
              topic_(topic),
              resolution_(resolution) {
            pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(topic_, rclcpp::QoS(1).transient_local());

            msg_ = std::make_shared<visualization_msgs::msg::MarkerArray>();
            msg_->markers.resize(2);
            for (int i = 0; i < 2; ++i) {
                msg_->markers[i].header.frame_id = markerarray_frame_id_;
                msg_->markers[i].ns = "map";
                msg_->markers[i].id = i;
                msg_->markers[i].type = visualization_msgs::msg::Marker::CUBE_LIST;
                msg_->markers[i].scale.x = resolution * pow(2, i);
                msg_->markers[i].scale.y = resolution * pow(2, i);
                msg_->markers[i].scale.z = resolution * pow(2, i);
                std_msgs::msg::ColorRGBA color;
                color.r = 0.0;
                color.g = 0.0;
                color.b = 1.0;
                color.a = 1.0;
                msg_->markers[i].color = color;
            }
        }

        void insert_point3d(float x, float y, float z, float min_z, float max_z, float size) {
            geometry_msgs::msg::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;
            if (size > 0)
                depth = (int) log2(size / 0.1);
            
            // Clamp depth to valid array bounds (0 to num_markers-1)
            depth = std::max(0, std::min(depth, (int)msg_->markers.size() - 1));

            msg_->markers[depth].points.push_back(center);
            if (min_z < max_z) {
                double h = (1.0 - std::min(std::max((z - min_z) / (max_z - min_z), 0.0f), 1.0f)) * 0.8;
                msg_->markers[depth].colors.push_back(heightMapColor(h));
            }
        }

        void clear_map(float size) {
          int depth = 0;
          if (size > 0)
            depth = (int) log2(size / 0.1);
          
          // Clamp depth to valid array bounds (0 to num_markers-1)
          depth = std::max(0, std::min(depth, (int)msg_->markers.size() - 1));

          msg_->markers[depth].points.clear();
          msg_->markers[depth].colors.clear();
        }

        void insert_point3d_semantics(float x, float y, float z, float size, int c, int dataset) {
            geometry_msgs::msg::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;
            if (size > 0)
                depth = (int) log2(size / 0.1);
            
            // Clamp depth to valid array bounds (0 to num_markers-1)
            depth = std::max(0, std::min(depth, (int)msg_->markers.size() - 1));

            msg_->markers[depth].points.push_back(center);
            
            // Use color map if loaded, otherwise fall back to hardcoded colors
            std_msgs::msg::ColorRGBA color;
            if (!color_map_.empty() && color_map_.find(c) != color_map_.end()) {
                color = color_map_[c];
            } else {
                // Fall back to hardcoded colors if color map not loaded
                switch (dataset) {
                  case 1:
                    color = KITTISemanticMapColor(c);
                    break;
                  case 2:
                    color = SemanticKITTISemanticMapColor(c);
                    break;
                  default:
                    color = SemanticMapColor(c);
                    break;
                }
            }
            msg_->markers[depth].colors.push_back(color);
        }

        /// Insert voxel colored by OSM prior value (0–1). prior_type: 0=building, 1=road, 2=grassland, 3=tree, 4=parking, 5=fence.
        void insert_point3d_osm_prior(float x, float y, float z, float size, float value, int prior_type) {
            geometry_msgs::msg::Point center;
            center.x = x;
            center.y = y;
            center.z = z;
            int depth = 0;
            if (size > 0)
                depth = (int) log2(size / 0.1);
            depth = std::max(0, std::min(depth, (int)msg_->markers.size() - 1));
            msg_->markers[depth].points.push_back(center);
            msg_->markers[depth].colors.push_back(OSMPriorMapColor(prior_type, value));
        }

        /// Insert voxel colored by blending all six OSM prior weights (overlapping weights blend colors).
        void insert_point3d_osm_blend(float x, float y, float z, float size, float w_building, float w_road, float w_grassland, float w_tree, float w_parking, float w_fence) {
            geometry_msgs::msg::Point center;
            center.x = x;
            center.y = y;
            center.z = z;
            int depth = 0;
            if (size > 0)
                depth = (int) log2(size / 0.1);
            depth = std::max(0, std::min(depth, (int)msg_->markers.size() - 1));
            msg_->markers[depth].points.push_back(center);
            msg_->markers[depth].colors.push_back(OSMPriorBlendColor(w_building, w_road, w_grassland, w_tree, w_parking, w_fence));
        }

        /// Set color mode for map visualization (semantic class vs OSM prior layer).
        void set_color_mode(MapColorMode mode) { color_mode_ = mode; }
        MapColorMode get_color_mode() const { return color_mode_; }

        /// Return the color for a semantic class ID, using the loaded YAML
        /// color_map_ if available, otherwise falling back to hardcoded palettes.
        std_msgs::msg::ColorRGBA get_color_for_class(int c, int dataset = 2) const {
            if (!color_map_.empty()) {
                auto it = color_map_.find(c);
                if (it != color_map_.end())
                    return it->second;
            }
            switch (dataset) {
              case 1:  return KITTISemanticMapColor(c);
              case 2:  return SemanticKITTISemanticMapColor(c);
              default: return SemanticMapColor(c);
            }
        }

        // Load colors from YAML file
        bool load_colors_from_params(rclcpp::Node::SharedPtr node) {
            try {
                // Try to get colors parameter from node
                if (!node->has_parameter("colors")) {
                    RCLCPP_WARN_STREAM(node->get_logger(), "No 'colors' parameter found. Using default hardcoded colors.");
                    return false;
                }
                
                rclcpp::Parameter colors_param = node->get_parameter("colors");
                if (colors_param.get_type() != rclcpp::ParameterType::PARAMETER_NOT_SET) {
                    // Parameter exists but may not be a dictionary
                    // For YAML files, we'll read them directly
                    RCLCPP_WARN_STREAM(node->get_logger(), "Colors parameter found but YAML reading not fully implemented. Using default colors.");
                    return false;
                }
            } catch (const std::exception& e) {
                RCLCPP_WARN_STREAM(node->get_logger(), "Error reading colors parameter: " << e.what() << ". Using default hardcoded colors.");
            }
            
            return false;
        }
        
        // Load colors from YAML file path
        bool load_colors_from_yaml(const std::string& yaml_file_path) {
            RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: MarkerArrayPub::load_colors_from_yaml: Starting, file=" << yaml_file_path);
            if (!node_) {
                RCLCPP_WARN_STREAM(rclcpp::get_logger("markerarray_pub"), "WARNING: node_ is null in load_colors_from_yaml!");
                return false;  // Safety check: node_ must be valid
            }
            RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: node_ is valid, loading YAML file");
            
            try {
                YAML::Node yaml_node = YAML::LoadFile(yaml_file_path);
                RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: YAML file loaded successfully");
                
                // Navigate through ROS2 parameter format: /**: ros__parameters: colors:
                YAML::Node colors_node;
                if (yaml_node["/**"] && yaml_node["/**"]["ros__parameters"] && yaml_node["/**"]["ros__parameters"]["colors"]) {
                    colors_node = yaml_node["/**"]["ros__parameters"]["colors"];
                } else if (yaml_node["colors"]) {
                    colors_node = yaml_node["colors"];
                } else if (yaml_node["color_map"]) {
                    colors_node = yaml_node["color_map"];
                } else {
                    RCLCPP_WARN_STREAM(node_->get_logger(), "No 'colors' or 'color_map' key found in YAML file: " << yaml_file_path);
                    return false;
                }
                
                if (!colors_node.IsMap()) {
                    RCLCPP_WARN_STREAM(node_->get_logger(), "WARNING: Colors node is not a map in YAML file: " << yaml_file_path);
                    return false;
                }
                RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Colors node is a map, starting to parse entries");
                
                color_map_.clear();
                int entry_count = 0;
                for (const auto& entry : colors_node) {
                    entry_count++;
                    try {
                        std::string key_str = entry.first.as<std::string>();
                        int class_id = std::stoi(key_str);
                        
                        if (entry.second.IsSequence() && entry.second.size() == 3) {
                            std_msgs::msg::ColorRGBA color;
                            color.a = 1.0;
                            // Convert from [0, 255] range to [0, 1] range
                            color.r = entry.second[0].as<double>() / 255.0;
                            color.g = entry.second[1].as<double>() / 255.0;
                            color.b = entry.second[2].as<double>() / 255.0;
                            color_map_[class_id] = color;
                        }
                    } catch (const std::exception& e) {
                        RCLCPP_WARN_STREAM(node_->get_logger(), "Error parsing color entry: " << e.what() << ", skipping");
                        continue;
                    }
                }
                
                bool success = !color_map_.empty();
                return success;
            } catch (const YAML::Exception& e) {
                RCLCPP_WARN_STREAM(node_->get_logger(), "YAML error loading colors from file " << yaml_file_path << ": " << e.what());
                return false;
            } catch (const std::exception& e) {
                RCLCPP_WARN_STREAM(node_->get_logger(), "Error loading colors from YAML file " << yaml_file_path << ": " << e.what());
                return false;
            } catch (...) {
                RCLCPP_WARN_STREAM(node_->get_logger(), "Unknown error loading colors from YAML file " << yaml_file_path);
                return false;
            }
        }

        void insert_point3d_variance(float x, float y, float z, float min_v, float max_v, float size, float var) {
            geometry_msgs::msg::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;
            if (size > 0)
                    depth = (int) log2(size / 0.1);
            
            // Clamp depth to valid array bounds (0 to num_markers-1)
            depth = std::max(0, std::min(depth, (int)msg_->markers.size() - 1));

            float middle = (max_v + min_v) / 2;
            var = (var - middle) / (middle - min_v);
            msg_->markers[depth].points.push_back(center);
            msg_->markers[depth].colors.push_back(JetMapColor(var));
        }

        void insert_point3d(float x, float y, float z, float min_z, float max_z) {
            insert_point3d(x, y, z, min_z, max_z, -1.0f);
        }

        void insert_point3d(float x, float y, float z) {
            insert_point3d(x, y, z, 1.0f, 0.0f, -1.0f);
        }

        void insert_color_point3d(float x, float y, float z, double min_v, double max_v, double v) {
            geometry_msgs::msg::Point center;
            center.x = x;
            center.y = y;
            center.z = z;

            int depth = 0;
            msg_->markers[depth].points.push_back(center);

            double h = (1.0 - std::min(std::max((v - min_v) / (max_v - min_v), 0.0), 1.0)) * 0.8;
            msg_->markers[depth].colors.push_back(heightMapColor(h));
        }

        void clear() {
            for (size_t i = 0; i < msg_->markers.size(); ++i) {
                msg_->markers[i].points.clear();
                msg_->markers[i].colors.clear();
            }
        }

        void publish() const {
            if (!msg_ || !pub_) {
                return;  // Safety check
            }
            if (msg_->markers.empty()) {
                return;  // No markers to publish
            }
            try {
                msg_->markers[0].header.stamp = node_->now();
                pub_->publish(*msg_);  // ROS2 publish accepts const reference
            } catch (const std::exception& e) {
                RCLCPP_ERROR_STREAM(node_->get_logger(), "Exception in MarkerArrayPub::publish(): " << e.what());
            }
        }

    private:
        rclcpp::Node::SharedPtr node_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
        visualization_msgs::msg::MarkerArray::SharedPtr msg_;
        std::string markerarray_frame_id_;
        std::string topic_;
        float resolution_;
        std::map<int, std_msgs::msg::ColorRGBA> color_map_;  // Class ID to color mapping
        MapColorMode color_mode_{MapColorMode::Semantic};
    };

}
