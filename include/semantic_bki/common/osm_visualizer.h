#pragma once

#include <rclcpp/rclcpp.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <std_msgs/msg/color_rgba.hpp>
#include <vector>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include <osmium/io/any_input.hpp>
#include <osmium/handler.hpp>
#include <osmium/visitor.hpp>
#include <osmium/osm/way.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/index/map/sparse_mem_array.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/osm/location.hpp>
#include "osm_geometry.h"

namespace semantic_bki {

    class OSMVisualizer {
    public:
        OSMVisualizer(rclcpp::Node::SharedPtr node, const std::string& topic);
        ~OSMVisualizer() = default;

        /**
         * Load OSM geometries from .osm XML file.
         * Extracts buildings, roads, and sidewalks using libosmium for proper parsing.
         * @param osm_file Path to .osm XML file
         * @param origin_lat Latitude of local coordinate origin (degrees)
         * @param origin_lon Longitude of local coordinate origin (degrees)
         * @return true if loaded successfully
         */
        bool loadFromOSM(const std::string& osm_file, double origin_lat, double origin_lon);

        /**
         * Publish OSM geometries as MarkerArray messages to RViz.
         */
        void publish();

        /**
         * Start periodic publishing at the specified rate.
         * @param rate Publishing rate in Hz (default: 1.0 Hz)
         */
        void startPeriodicPublishing(double rate = 1.0);

        /**
         * Transform all OSM coordinates by the inverse of the first pose.
         * This aligns OSM data with the same coordinate frame as scans/map (relative to first pose).
         * @param first_pose 4x4 transformation matrix of the original first pose (before alignment)
         */
        void transformToFirstPoseOrigin(const Eigen::Matrix4d& first_pose);

        /**
         * Set the lidar trajectory path (sequence of x,y points in map frame) for debugging.
         * Drawn as a polyline. Call after transformToFirstPoseOrigin if using same frame.
         */
        void setPath(const std::vector<std::pair<float, float>>& path);

        /**
         * Set radius (meters) for tree points (natural=tree nodes). Drawn as 2D circles. Same value used for prior projection.
         * @param radius_meters Radius in meters (default 5.0)
         */
        void setTreePointRadius(float radius_meters) { tree_point_radius_meters_ = std::max(0.1f, radius_meters); }

        float getTreePointRadius() const { return tree_point_radius_meters_; }

        /**
         * Set width (meters) for stairs (highway=steps). Each polyline segment is drawn as a rectangle
         * with this width perpendicular to the segment.
         * @param width_meters Width in meters (default 1.5)
         */
        void setStairsWidth(float width_meters) { stairs_width_meters_ = std::max(0.1f, width_meters); }

        float getStairsWidth() const { return stairs_width_meters_; }

        /// Return OSM geometries (after transform if applied). Used to set voxel OSM priors.
        const std::vector<Geometry2D>& getBuildings() const { return buildings_; }
        const std::vector<Geometry2D>& getRoads() const { return roads_; }
        const std::vector<Geometry2D>& getSidewalks() const { return sidewalks_; }
        const std::vector<Geometry2D>& getParking() const { return parking_; }
        const std::vector<Geometry2D>& getFences() const { return fences_; }
        const std::vector<Geometry2D>& getStairs() const { return stairs_; }
        const std::vector<Geometry2D>& getGrasslands() const { return grasslands_; }
        const std::vector<Geometry2D>& getTrees() const { return trees_; }
        const std::vector<Geometry2D>& getForests() const { return forests_; }
        const std::vector<std::pair<float, float>>& getTreePoints() const { return tree_points_; }
        const std::vector<Geometry2D>& getCycleways() const { return cycleways_; }
        const std::vector<Geometry2D>& getWalls() const { return walls_; }
        const std::vector<Geometry2D>& getWater() const { return water_; }
        const std::vector<std::pair<float, float>>& getPolePoints() const { return pole_points_; }

    private:
        /**
         * Timer callback for periodic publishing.
         */
        void timerCallback();

        /**
         * Create Marker message for buildings (line outlines).
         */
        visualization_msgs::msg::Marker createBuildingMarker(const std::vector<Geometry2D>& buildings);

        /**
         * Create Marker message for roads (red polylines).
         */
        visualization_msgs::msg::Marker createRoadMarker(const std::vector<Geometry2D>& roads);

        /**
         * Create Marker message for sidewalks (cyan polylines).
         */
        visualization_msgs::msg::Marker createSidewalkMarker(const std::vector<Geometry2D>& sidewalks);

        /**
         * Create Marker message for parking (yellow/orange polylines).
         */
        visualization_msgs::msg::Marker createParkingMarker(const std::vector<Geometry2D>& parking);

        /**
         * Create Marker message for fences (barrier=fence polylines).
         */
        visualization_msgs::msg::Marker createFenceMarker(const std::vector<Geometry2D>& fences);

        /**
         * Create Marker message for stairs (highway=steps). Each segment drawn as rectangle with stairs_width_meters_.
         */
        visualization_msgs::msg::Marker createStairsMarker(const std::vector<Geometry2D>& stairs);

        /**
         * Create Marker message for lidar path (green polyline).
         */
        visualization_msgs::msg::Marker createPathMarker() const;

        /**
         * Create Marker message for grasslands (greenish polygon outlines).
         */
        visualization_msgs::msg::Marker createGrasslandMarker(const std::vector<Geometry2D>& grasslands);

        /**
         * Create Marker message for trees (landcover=trees, orchard, vineyard).
         */
        visualization_msgs::msg::Marker createTreeMarker(const std::vector<Geometry2D>& trees);

        /**
         * Create Marker message for forests (landuse=forest, natural=forest/wood).
         */
        visualization_msgs::msg::Marker createForestMarker(const std::vector<Geometry2D>& forests);

        /**
         * Create Marker message for tree points (single-node trees as 2D circle outlines).
         */
        visualization_msgs::msg::Marker createTreePointsMarker() const;

        visualization_msgs::msg::Marker createCyclewayMarker(const std::vector<Geometry2D>& cycleways);
        visualization_msgs::msg::Marker createWallMarker(const std::vector<Geometry2D>& walls);
        visualization_msgs::msg::Marker createWaterMarker(const std::vector<Geometry2D>& water);
        visualization_msgs::msg::Marker createPolePointsMarker() const;

        rclcpp::Node::SharedPtr node_;
        rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr pub_;
        rclcpp::TimerBase::SharedPtr publish_timer_;
        std::string topic_;
        std::string frame_id_;

        std::vector<Geometry2D> buildings_;
        std::vector<Geometry2D> roads_;
        std::vector<Geometry2D> sidewalks_;
        std::vector<Geometry2D> parking_;
        std::vector<Geometry2D> fences_;
        std::vector<Geometry2D> stairs_;
        std::vector<Geometry2D> grasslands_;
        std::vector<Geometry2D> trees_;           // landcover=trees, orchard, vineyard
        std::vector<Geometry2D> forests_;         // landuse=forest, natural=forest/wood
        std::vector<std::pair<float, float>> tree_points_;  // Single-point trees (natural=tree nodes)
        std::vector<Geometry2D> cycleways_;
        std::vector<Geometry2D> walls_;
        std::vector<Geometry2D> water_;
        std::vector<std::pair<float, float>> pole_points_;  // traffic_signals, power poles, etc.
        std::vector<std::pair<float, float>> path_;  // Lidar trajectory for debugging
        float tree_point_radius_meters_{5.0f};  // Radius for tree point circles (visualization and prior)
        float stairs_width_meters_{1.5f};       // Width of rectangle enclosing each stairs segment

        bool transformed_; // Flag to track if data has already been transformed
    };

} // namespace semantic_bki
