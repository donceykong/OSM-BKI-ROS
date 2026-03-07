#include "osm_visualizer.h"
#include <sstream>
#include <cmath>
#include <vector>
#include <string>
#include <memory>
#include <algorithm>
#include <limits>
#include <Eigen/Dense>
#include <osmium/io/any_input.hpp>
#include <osmium/handler.hpp>
#include <osmium/visitor.hpp>
#include <osmium/osm/way.hpp>
#include <osmium/osm/node.hpp>
#include <osmium/osm/relation.hpp>
#include <osmium/osm/area.hpp>
#include <osmium/index/map/sparse_mem_array.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/osm/location.hpp>
#include <osmium/area/assembler.hpp>
#include <osmium/area/multipolygon_manager.hpp>
#include <osmium/tags/tags_filter.hpp>

namespace {

    /// Add polygon outline (outer ring + holes) as line segments to marker.
    void addPolygonOutlineToMarker(const semantic_bki::Geometry2D& poly,
                                   visualization_msgs::msg::Marker& marker) {
        auto addRing = [&](const std::vector<std::pair<float, float>>& coords) {
            if (coords.size() < 2) return;
            for (size_t i = 0; i < coords.size(); ++i) {
                size_t next = (i + 1) % coords.size();
                geometry_msgs::msg::Point p1, p2;
                p1.x = coords[i].first;
                p1.y = coords[i].second;
                p1.z = 0.0;
                p2.x = coords[next].first;
                p2.y = coords[next].second;
                p2.z = 0.0;
                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
        };
        addRing(poly.coords);
        for (const auto& hole : poly.holes) {
            addRing(hole.coords);
        }
    }

    /// Add a rectangle outline for each segment of a polyline (width band rendering).
    void addPolylineBandOutlineToMarker(const std::vector<std::pair<float, float>>& coords,
                                        float width_m,
                                        visualization_msgs::msg::Marker& marker) {
        if (coords.size() < 2) return;
        const float hw = std::max(0.1f, width_m) * 0.5f;
        const float eps = 1e-6f;

        auto addLine = [&](float ax, float ay, float bx, float by) {
            geometry_msgs::msg::Point p1, p2;
            p1.x = ax; p1.y = ay; p1.z = 0.0;
            p2.x = bx; p2.y = by; p2.z = 0.0;
            marker.points.push_back(p1);
            marker.points.push_back(p2);
        };

        for (size_t i = 0; i + 1 < coords.size(); ++i) {
            float x1 = coords[i].first;
            float y1 = coords[i].second;
            float x2 = coords[i + 1].first;
            float y2 = coords[i + 1].second;
            float dx = x2 - x1;
            float dy = y2 - y1;
            float L = std::sqrt(dx * dx + dy * dy);
            if (L < eps) continue;
            float nx = -dy / L;
            float ny = dx / L;

            float c1x = x1 + hw * nx, c1y = y1 + hw * ny;
            float c2x = x1 - hw * nx, c2y = y1 - hw * ny;
            float c3x = x2 - hw * nx, c3y = y2 - hw * ny;
            float c4x = x2 + hw * nx, c4y = y2 + hw * ny;

            addLine(c1x, c1y, c2x, c2y);
            addLine(c2x, c2y, c3x, c3y);
            addLine(c3x, c3y, c4x, c4y);
            addLine(c4x, c4y, c1x, c1y);
        }
    }

    // Handler class for extracting buildings, roads, sidewalks, cycleways, walls, water, pole points, etc. from OSM
    class OSMGeometryHandler : public osmium::handler::Handler {
    public:
        static constexpr double EARTH_RADIUS_M = 6378137.0;

        OSMGeometryHandler(double origin_lat, double origin_lon,
                          std::vector<semantic_bki::Geometry2D>& buildings,
                          std::vector<semantic_bki::Geometry2D>& roads,
                          std::vector<semantic_bki::Geometry2D>& sidewalks,
                          std::vector<semantic_bki::Geometry2D>& cycleways,
                          std::vector<semantic_bki::Geometry2D>& walls,
                          std::vector<semantic_bki::Geometry2D>& water,
                          std::vector<std::pair<float, float>>& pole_points,
                          std::vector<semantic_bki::Geometry2D>& parking,
                          std::vector<semantic_bki::Geometry2D>& fences,
                          std::vector<semantic_bki::Geometry2D>& stairs,
                          std::vector<semantic_bki::Geometry2D>& grasslands,
                          std::vector<semantic_bki::Geometry2D>& trees,
                          std::vector<semantic_bki::Geometry2D>& forests,
                          std::vector<std::pair<float, float>>& tree_points)
            : origin_lat_(origin_lat), origin_lon_(origin_lon),
              buildings_(buildings), roads_(roads), sidewalks_(sidewalks), cycleways_(cycleways),
              walls_(walls), water_(water), pole_points_(pole_points),
              parking_(parking), fences_(fences), stairs_(stairs),
              grasslands_(grasslands), trees_(trees), forests_(forests), tree_points_(tree_points) {
            scale_ = std::cos(origin_lat * M_PI / 180.0);
            double origin_lon_rad = origin_lon * M_PI / 180.0;
            double origin_lat_rad = origin_lat * M_PI / 180.0;
            origin_mx_ = scale_ * origin_lon_rad * EARTH_RADIUS_M;
            origin_my_ = scale_ * EARTH_RADIUS_M * std::log(std::tan(M_PI / 4.0 + origin_lat_rad / 2.0));
        }

        std::pair<float, float> latlon_to_xy(double lat, double lon) const {
            double lon_rad = lon * M_PI / 180.0;
            double lat_rad = lat * M_PI / 180.0;
            double mx = scale_ * lon_rad * EARTH_RADIUS_M;
            double my = scale_ * EARTH_RADIUS_M * std::log(std::tan(M_PI / 4.0 + lat_rad / 2.0));
            return {static_cast<float>(mx - origin_mx_), static_cast<float>(my - origin_my_)};
        }

        void node(const osmium::Node& node) {
            const osmium::Location& loc = node.location();
            if (!loc.valid()) return;
            auto xy = latlon_to_xy(loc.lat(), loc.lon());

            // Single-point trees: natural=tree
            const char* natural_tag = node.tags()["natural"];
            if (natural_tag && std::string(natural_tag) == "tree") {
                tree_points_.push_back(xy);
                return;
            }

            // Pole-like points: traffic signals, power poles, street cabinets, masts
            const char* highway_tag = node.tags()["highway"];
            if (highway_tag && std::string(highway_tag) == "traffic_signals") {
                pole_points_.push_back(xy);
                return;
            }
            const char* power_tag = node.tags()["power"];
            if (power_tag && (std::string(power_tag) == "pole" || std::string(power_tag) == "tower")) {
                pole_points_.push_back(xy);
                return;
            }
            const char* man_made_tag = node.tags()["man_made"];
            if (man_made_tag && (std::string(man_made_tag) == "street_cabinet" || std::string(man_made_tag) == "mast")) {
                pole_points_.push_back(xy);
            }
        }

        void way(const osmium::Way& way) {
            // Extract node coordinates
            semantic_bki::Geometry2D geom;
            for (const auto& node_ref : way.nodes()) {
                const osmium::Location& location = node_ref.location();
                if (location.valid()) {
                    geom.coords.push_back(latlon_to_xy(location.lat(), location.lon()));
                }
            }

            if (geom.coords.size() < 2) {
                return;
            }

            // Check if this way is a building
            const char* building_tag = way.tags()["building"];
            if (building_tag) {
                if (geom.coords.size() >= 3) {
                    buildings_.push_back(geom);
                }
                return;
            }

            // --- PARKING (ways) ---
            const char* amenity_tag = way.tags()["amenity"];
            if (amenity_tag && std::string(amenity_tag) == "parking") {
                parking_.push_back(geom);
                return;
            }
            if (amenity_tag && std::string(amenity_tag) == "parking_space") {
                parking_.push_back(geom);
                return;
            }

            // --- FENCES (barrier=fence) and WALLS (barrier=wall, man_made=wall) ---
            const char* barrier_tag = way.tags()["barrier"];
            if (barrier_tag && std::string(barrier_tag) == "fence") {
                fences_.push_back(geom);
                return;
            }
            if (barrier_tag && std::string(barrier_tag) == "wall") {
                walls_.push_back(geom);
                return;
            }
            const char* man_made_way_tag = way.tags()["man_made"];
            if (man_made_way_tag && std::string(man_made_way_tag) == "wall") {
                walls_.push_back(geom);
                return;
            }

            // --- STAIRS (highway=steps) ---
            const char* highway_tag = way.tags()["highway"];
            if (highway_tag && std::string(highway_tag) == "steps") {
                stairs_.push_back(geom);
                return;
            }

            // --- CYCLEWAYS (highway=cycleway or highway=path + bicycle=yes) ---
            if (highway_tag && std::string(highway_tag) == "cycleway") {
                cycleways_.push_back(geom);
                return;
            }
            const char* bicycle_tag = way.tags()["bicycle"];
            if (highway_tag && std::string(highway_tag) == "path" && bicycle_tag && std::string(bicycle_tag) == "yes") {
                cycleways_.push_back(geom);
                return;
            }

            // --- SIDEWALKS: highway=footway + footway=sidewalk, or highway=path/footway/pedestrian ---
            const char* footway_tag = way.tags()["footway"];
            if (highway_tag) {
                std::string highway(highway_tag);
                if (highway == "footway") {
                    if (footway_tag && std::string(footway_tag) == "sidewalk") {
                        sidewalks_.push_back(geom);
                    } else {
                        sidewalks_.push_back(geom);  // footway without sidewalk tag -> sidewalk
                    }
                    return;
                }
                if (highway == "path" || highway == "pedestrian" || highway == "foot") {
                    sidewalks_.push_back(geom);
                    return;
                }
            }

            // --- WATER (natural=water ways) ---
            const char* natural_way_tag = way.tags()["natural"];
            if (natural_way_tag && std::string(natural_way_tag) == "water") {
                water_.push_back(geom);
                return;
            }

            // --- ROADS (exclude footway, path, pedestrian, cycleway - handled above) ---
            if (highway_tag) {
                std::string highway(highway_tag);
                if (highway == "motorway" || highway == "trunk" || highway == "primary" ||
                    highway == "secondary" || highway == "tertiary" ||
                    highway == "unclassified" || highway == "residential" ||
                    highway == "motorway_link" || highway == "trunk_link" ||
                    highway == "primary_link" || highway == "secondary_link" ||
                    highway == "tertiary_link" || highway == "living_street" ||
                    highway == "service" || highway == "road") {
                    roads_.push_back(geom);
                    const char* sidewalk_tag = way.tags()["sidewalk"];
                    if (sidewalk_tag) {
                        std::string sw(sidewalk_tag);
                        if (sw == "both" || sw == "left" || sw == "right" || sw == "yes") {
                            sidewalks_.push_back(geom);
                        }
                    }
                }
                return;
            }

            // Grassland: landuse=grass, natural=grassland, landuse=meadow, landuse=greenfield
            const char* landuse_tag = way.tags()["landuse"];
            if (landuse_tag) {
                std::string landuse(landuse_tag);
                if (landuse == "grass" || landuse == "meadow" || landuse == "greenfield") {
                    if (geom.coords.size() >= 3) {
                        grasslands_.push_back(geom);
                    }
                    return;
                }
            }
            const char* natural_tag = way.tags()["natural"];
            if (natural_tag) {
                std::string natural(natural_tag);
                if (natural == "grassland" || natural == "heath" || natural == "scrub") {
                    if (geom.coords.size() >= 3) {
                        grasslands_.push_back(geom);
                    }
                    return;
                }
                // Forest: natural=wood, natural=forest (larger wooded areas)
                if (natural == "wood" || natural == "forest") {
                    if (geom.coords.size() >= 3) {
                        forests_.push_back(geom);
                    }
                    return;
                }
            }
            if (landuse_tag && std::string(landuse_tag) == "forest") {
                if (geom.coords.size() >= 3) {
                    forests_.push_back(geom);
                }
            }
            // landcover=trees (tree-covered areas)
            const char* landcover_tag = way.tags()["landcover"];
            if (landcover_tag && std::string(landcover_tag) == "trees") {
                if (geom.coords.size() >= 3) {
                    trees_.push_back(geom);
                }
            }
        }

        /// Helper: extract outer + inner rings from area into polygons with holes, push to container.
        void push_area_with_holes(const osmium::Area& area, std::vector<semantic_bki::Geometry2D>& container) {
            for (const auto& outer_ring : area.outer_rings()) {
                semantic_bki::Geometry2D geom;
                for (const auto& node_ref : outer_ring) {
                    const osmium::Location& location = node_ref.location();
                    if (location.valid()) {
                        geom.coords.push_back(latlon_to_xy(location.lat(), location.lon()));
                    }
                }
                for (const auto& inner_ring : area.inner_rings(outer_ring)) {
                    semantic_bki::Geometry2D hole;
                    for (const auto& node_ref : inner_ring) {
                        const osmium::Location& location = node_ref.location();
                        if (location.valid()) {
                            hole.coords.push_back(latlon_to_xy(location.lat(), location.lon()));
                        }
                    }
                    if (hole.coords.size() >= 3) {
                        geom.holes.push_back(std::move(hole));
                    }
                }
                if (geom.coords.size() >= 3) {
                    container.push_back(std::move(geom));
                }
            }
        }

        // Handle areas (from multipolygons or closed ways) - called by MultipolygonManager
        void area(const osmium::Area& area) {
            // Check for parking areas (amenity=parking as multipolygon)
            const char* amenity_tag = area.tags()["amenity"];
            if (amenity_tag && std::string(amenity_tag) == "parking") {
                push_area_with_holes(area, parking_);
                return;
            }

            // Check tags to determine type
            const char* building_tag = area.tags()["building"];
            if (building_tag) {
                push_area_with_holes(area, buildings_);
                return;
            }

            // Check for grassland/meadow/recreation areas (multipolygons)
            const char* landuse_tag = area.tags()["landuse"];
            if (landuse_tag) {
                std::string landuse(landuse_tag);
                if (landuse == "grass" || landuse == "meadow" || landuse == "greenfield" ||
                    landuse == "recreation_ground") {
                    push_area_with_holes(area, grasslands_);
                    return;
                }
                if (landuse == "orchard" || landuse == "vineyard") {
                    push_area_with_holes(area, trees_);
                    return;
                }
                if (landuse == "forest") {
                    push_area_with_holes(area, forests_);
                    return;
                }
            }

            // // Check for leisure=park, leisure=garden (multipolygons) -> grasslands
            // const char* leisure_tag = area.tags()["leisure"];
            // if (leisure_tag) {
            //     std::string leisure(leisure_tag);
            //     if (leisure == "park" || leisure == "garden") {
            //         push_area_with_holes(area, grasslands_);
            //         return;
            //     }
            // }

            // Check for water (natural=water multipolygons)
            const char* natural_tag = area.tags()["natural"];
            if (natural_tag && std::string(natural_tag) == "water") {
                push_area_with_holes(area, water_);
                return;
            }
            if (natural_tag) {
                std::string natural(natural_tag);
                if (natural == "wood" || natural == "forest") {
                    push_area_with_holes(area, forests_);
                    return;
                }
            }
            if (landuse_tag && std::string(landuse_tag) == "forest") {
                push_area_with_holes(area, forests_);
                return;
            }
            // landcover=trees (tree-covered areas, multipolygons) -> trees_
            const char* landcover_tag = area.tags()["landcover"];
            if (landcover_tag && std::string(landcover_tag) == "trees") {
                push_area_with_holes(area, trees_);
            }
        }

    private:
        double origin_lat_, origin_lon_;
        double scale_, origin_mx_, origin_my_;
        std::vector<semantic_bki::Geometry2D>& buildings_;
        std::vector<semantic_bki::Geometry2D>& roads_;
        std::vector<semantic_bki::Geometry2D>& sidewalks_;
        std::vector<semantic_bki::Geometry2D>& cycleways_;
        std::vector<semantic_bki::Geometry2D>& walls_;
        std::vector<semantic_bki::Geometry2D>& water_;
        std::vector<std::pair<float, float>>& pole_points_;
        std::vector<semantic_bki::Geometry2D>& parking_;
        std::vector<semantic_bki::Geometry2D>& fences_;
        std::vector<semantic_bki::Geometry2D>& stairs_;
        std::vector<semantic_bki::Geometry2D>& grasslands_;
        std::vector<semantic_bki::Geometry2D>& trees_;
        std::vector<semantic_bki::Geometry2D>& forests_;
        std::vector<std::pair<float, float>>& tree_points_;
    };
}

namespace semantic_bki {

    OSMVisualizer::OSMVisualizer(rclcpp::Node::SharedPtr node, const std::string& topic) 
        : node_(node), topic_(topic), frame_id_("map"), transformed_(false) {
        if (!node_) {
            RCLCPP_ERROR_STREAM(rclcpp::get_logger("osm_visualizer"), "ERROR: OSMVisualizer constructor: node_ is null!");
            return;
        }
        // Only create publisher if topic is not empty (allows using OSMVisualizer just for loading/transforming OSM data)
        if (!topic_.empty()) {
            // RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: OSMVisualizer constructor: Creating publisher for topic: " << topic_);
            // Use default QoS for ROS2 (compatible with ros2 topic hz and RViz)
            // transient_local requires matching QoS on subscriber side, which ros2 topic hz doesn't use
            // Use default reliable QoS instead for better compatibility
            pub_ = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
                topic_, 
                rclcpp::QoS(10).reliable());
            if (!pub_) {
                RCLCPP_ERROR_STREAM(node_->get_logger(), "ERROR: Failed to create OSM publisher!");
            } else {
                // RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: OSM publisher created successfully");
                // RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Publisher topic name: " << pub_->get_topic_name());
                // RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Publisher subscription count: " << pub_->get_subscription_count());
                // RCLCPP_INFO_STREAM(node_->get_logger(), "OSMVisualizer: Publisher created for topic: " << topic_ << " with reliable QoS");
            }
        } else {
            pub_ = nullptr;  // No publisher needed if topic is empty
        }
    }

    bool OSMVisualizer::loadFromOSM(const std::string& osm_file, double origin_lat, double origin_lon) {
        buildings_.clear();
        roads_.clear();
        sidewalks_.clear();
        cycleways_.clear();
        walls_.clear();
        water_.clear();
        pole_points_.clear();
        parking_.clear();
        fences_.clear();
        stairs_.clear();
        grasslands_.clear();
        trees_.clear();
        forests_.clear();
        tree_points_.clear();
        
        // RCLCPP_INFO_STREAM(node_->get_logger(), "OSMVisualizer::loadFromOSM called with file: " << osm_file);
        // RCLCPP_INFO_STREAM(node_->get_logger(), "  Origin: (" << origin_lat << ", " << origin_lon << ")");
        
        try {
            osmium::io::File input_file(osm_file);
            
            // Create index to store node locations
            osmium::index::map::SparseMemArray<osmium::unsigned_object_id_type, osmium::Location> index;
            
            // Handler to store node locations
            osmium::handler::NodeLocationsForWays<osmium::index::map::SparseMemArray<osmium::unsigned_object_id_type, osmium::Location>> 
                location_handler(index);
            
            OSMGeometryHandler handler(origin_lat, origin_lon, buildings_, roads_, sidewalks_, cycleways_, walls_, water_, pole_points_, parking_, fences_, stairs_, grasslands_, trees_, forests_, tree_points_);
            
            // MultipolygonManager to convert multipolygon relations (type=multipolygon) to areas
            // Filter: which relations to assemble. Each rule accepts relations with matching tags.
            osmium::area::AssemblerConfig assembler_config;
            osmium::TagsFilter filter(false);  // Start with false (reject all)
            filter.add_rule(true, "building");  // Buildings (simple and multipolygon)
            filter.add_rule(true, "landuse", "grass");
            filter.add_rule(true, "landuse", "meadow");
            filter.add_rule(true, "landuse", "greenfield");
            filter.add_rule(true, "landuse", "forest");
            filter.add_rule(true, "landuse", "recreation_ground");
            filter.add_rule(true, "landuse", "orchard");
            filter.add_rule(true, "landuse", "vineyard");
            filter.add_rule(true, "natural", "grassland");
            filter.add_rule(true, "natural", "heath");
            filter.add_rule(true, "natural", "scrub");
            filter.add_rule(true, "natural", "wood");
            filter.add_rule(true, "natural", "forest");
            filter.add_rule(true, "landcover", "trees");  // Tree-covered areas (OSM tag)
            filter.add_rule(true, "leisure", "park");
            filter.add_rule(true, "leisure", "garden");
            filter.add_rule(true, "amenity", "parking");
            filter.add_rule(true, "natural", "water");
            using MultipolygonManager = osmium::area::MultipolygonManager<osmium::area::Assembler>;
            MultipolygonManager mp_manager(assembler_config, filter);
            
            // First pass: read all objects
            // - location_handler stores node locations
            // - handler processes ways and nodes (buildings, roads, etc. as simple ways)
            // - mp_manager collects multipolygon relations and their member ways
            osmium::io::Reader reader1(input_file);
            osmium::apply(reader1, location_handler, handler, mp_manager);
            reader1.close();
            
            // Prepare MultipolygonManager for second pass (required before lookup)
            mp_manager.prepare_for_lookup();
            
            // Second pass: read again, MultipolygonManager outputs completed areas (multipolygons)
            osmium::io::Reader reader2(input_file);
            osmium::apply(reader2, location_handler, mp_manager.handler([&handler](osmium::memory::Buffer&& buffer) {
                // This callback receives buffers containing completed areas (multipolygons)
                // Process only areas - handler.area() will be called for each area
                for (const auto& item : buffer) {
                    if (item.type() == osmium::item_type::area) {
                        handler.area(static_cast<const osmium::Area&>(item));
                    }
                }
            }));
            reader2.close();
            
            // RCLCPP_INFO_STREAM(node_->get_logger(), "Loaded " << buildings_.size() << " buildings, " << roads_.size() << " roads/sidewalks, " << grasslands_.size() << " grasslands, " << trees_.size() << " tree/forest polygons, " << tree_points_.size() << " tree points from OSM file using libosmium");
            
            if (buildings_.empty() && roads_.empty() && sidewalks_.empty() && cycleways_.empty() && walls_.empty() && water_.empty() && pole_points_.empty() && parking_.empty() && fences_.empty() && stairs_.empty() && grasslands_.empty() && trees_.empty() && forests_.empty() && tree_points_.empty()) {
                // RCLCPP_WARN(node_->get_logger(), "WARNING: No buildings, roads, grasslands, or trees found in OSM file.");
            }
            
            size_t total_building_points = 0, total_road_points = 0, total_grass_points = 0, total_tree_points = 0;
            for (const auto& b : buildings_) total_building_points += b.coords.size();
            for (const auto& r : roads_) total_road_points += r.coords.size();
            for (const auto& g : grasslands_) total_grass_points += g.coords.size();
            for (const auto& t : trees_) total_tree_points += t.coords.size();
            // RCLCPP_INFO_STREAM(node_->get_logger(), "Total points - buildings: " << total_building_points << ", roads: " << total_road_points << ", grasslands: " << total_grass_points << ", tree polygons: " << total_tree_points << ", tree points: " << tree_points_.size());
            
            return true;
        } catch (const std::exception& e) {
            RCLCPP_ERROR_STREAM(node_->get_logger(), "Error parsing OSM file with libosmium: " << e.what());
            return false;
        }
    }

    visualization_msgs::msg::Marker OSMVisualizer::createBuildingMarker(const std::vector<Geometry2D>& buildings) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_buildings";
        marker.id = 0;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST; // Use LINE_LIST for building outlines
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.5; // Line width in meters
        marker.color.r = 0.0;
        marker.color.g = 0.0;
        marker.color.b = 1.0;
        marker.color.a = 1.0; // Fully opaque blue lines

        for (const auto& building : buildings) {
            if (building.coords.size() < 2) continue;
            bool has_invalid = false;
            for (const auto& coord : building.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) ||
                    std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            for (const auto& hole : building.holes) {
                for (const auto& coord : hole.coords) {
                    if (std::isnan(coord.first) || std::isnan(coord.second) ||
                        std::isinf(coord.first) || std::isinf(coord.second)) {
                        has_invalid = true;
                        break;
                    }
                }
                if (has_invalid) break;
            }
            if (has_invalid) continue;
            addPolygonOutlineToMarker(building, marker);
        }

        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createRoadMarker(const std::vector<Geometry2D>& roads) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_roads";
        marker.id = 1;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST; // Use LINE_LIST for road polylines
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.15; // Outline stroke width (polygon band width handled geometrically)
        marker.color.r = 1.0;
        marker.color.g = 0.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0; // Fully opaque red lines

        for (const auto& road : roads) {
            if (road.coords.size() < 2) continue; // Need at least 2 points for a line
            
            // Check for NaN or invalid coordinates
            bool has_invalid = false;
            for (const auto& coord : road.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) ||
                    std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) {
                // RCLCPP_WARN_STREAM(node_->get_logger(), "Skipping road polyline with invalid (NaN/Inf) coordinates");
                continue;
            }
            
            addPolylineBandOutlineToMarker(road.coords, road_width_meters_, marker);
        }

        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createSidewalkMarker(const std::vector<Geometry2D>& sidewalks) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_sidewalks";
        marker.id = 6;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.12;
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 1.0f;
        marker.color.a = 1.0f;  // Cyan

        for (const auto& sidewalk : sidewalks) {
            if (sidewalk.coords.size() < 2) continue;
            bool has_invalid = false;
            for (const auto& coord : sidewalk.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) ||
                    std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) continue;
            addPolylineBandOutlineToMarker(sidewalk.coords, sidewalk_width_meters_, marker);
        }
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createParkingMarker(const std::vector<Geometry2D>& parking) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_parking";
        marker.id = 7;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.3;
        marker.color.r = 1.0f;
        marker.color.g = 0.65f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0f;  // Orange

        for (const auto& park : parking) {
            if (park.coords.size() < 2) continue;
            bool has_invalid = false;
            for (const auto& coord : park.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) ||
                    std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            for (const auto& hole : park.holes) {
                for (const auto& coord : hole.coords) {
                    if (std::isnan(coord.first) || std::isnan(coord.second) ||
                        std::isinf(coord.first) || std::isinf(coord.second)) {
                        has_invalid = true;
                        break;
                    }
                }
                if (has_invalid) break;
            }
            if (has_invalid) continue;
            addPolygonOutlineToMarker(park, marker);
        }
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createFenceMarker(const std::vector<Geometry2D>& fences) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_fences";
        marker.id = 8;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.12;
        marker.color.r = 0.5f;
        marker.color.g = 0.45f;
        marker.color.b = 0.4f;
        marker.color.a = 1.0f;  // Gray/brown

        for (const auto& fence : fences) {
            if (fence.coords.size() < 2) continue;
            bool has_invalid = false;
            for (const auto& coord : fence.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) ||
                    std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) continue;
            addPolylineBandOutlineToMarker(fence.coords, fence_width_meters_, marker);
        }
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createStairsMarker(const std::vector<Geometry2D>& stairs) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_stairs";
        marker.id = 9;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.12;
        marker.color.r = 0.6f;
        marker.color.g = 0.4f;
        marker.color.b = 0.2f;
        marker.color.a = 1.0f;  // Brown/tan for stairs

        for (const auto& stair : stairs) {
            if (stair.coords.size() < 2) continue;
            bool has_invalid = false;
            for (const auto& coord : stair.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) ||
                    std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            if (has_invalid) continue;
            addPolylineBandOutlineToMarker(stair.coords, stairs_width_meters_, marker);
        }
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createPathMarker() const {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "lidar_path";
        marker.id = 2;
        marker.type = visualization_msgs::msg::Marker::LINE_STRIP;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.4;  // Line width
        marker.color.r = 0.0;
        marker.color.g = 1.0;
        marker.color.b = 0.0;
        marker.color.a = 1.0;  // Green, opaque

        for (const auto& pt : path_) {
            geometry_msgs::msg::Point p;
            p.x = pt.first;
            p.y = pt.second;
            p.z = 0.0;
            marker.points.push_back(p);
        }
        return marker;
    }

    void OSMVisualizer::setPath(const std::vector<std::pair<float, float>>& path) {
        path_ = path;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createGrasslandMarker(const std::vector<Geometry2D>& grasslands) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_grasslands";
        marker.id = 3;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.25;
        marker.color.r = 0.4f;
        marker.color.g = 0.85f;
        marker.color.b = 0.35f;
        marker.color.a = 0.7f;

        for (const auto& grassland : grasslands) {
            if (grassland.coords.size() < 3) continue;
            bool has_invalid = false;
            for (const auto& coord : grassland.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) || std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            for (const auto& hole : grassland.holes) {
                for (const auto& coord : hole.coords) {
                    if (std::isnan(coord.first) || std::isnan(coord.second) || std::isinf(coord.first) || std::isinf(coord.second)) {
                        has_invalid = true;
                        break;
                    }
                }
                if (has_invalid) break;
            }
            if (has_invalid) continue;
            addPolygonOutlineToMarker(grassland, marker);
        }
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createForestMarker(const std::vector<Geometry2D>& forests) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_forests";
        marker.id = 10;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.35;
        marker.color.r = 0.0f;
        marker.color.g = 0.4f;
        marker.color.b = 0.1f;
        marker.color.a = 0.95f;

        for (const auto& forest : forests) {
            if (forest.coords.size() < 3) continue;
            bool has_invalid = false;
            for (const auto& coord : forest.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) || std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            for (const auto& hole : forest.holes) {
                for (const auto& coord : hole.coords) {
                    if (std::isnan(coord.first) || std::isnan(coord.second) || std::isinf(coord.first) || std::isinf(coord.second)) {
                        has_invalid = true;
                        break;
                    }
                }
                if (has_invalid) break;
            }
            if (has_invalid) continue;
            addPolygonOutlineToMarker(forest, marker);
        }
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createTreeMarker(const std::vector<Geometry2D>& trees) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_trees";
        marker.id = 4;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.3;
        marker.color.r = 0.1f;
        marker.color.g = 0.5f;
        marker.color.b = 0.2f;
        marker.color.a = 0.9f;

        for (const auto& tree : trees) {
            if (tree.coords.size() < 3) continue;
            bool has_invalid = false;
            for (const auto& coord : tree.coords) {
                if (std::isnan(coord.first) || std::isnan(coord.second) || std::isinf(coord.first) || std::isinf(coord.second)) {
                    has_invalid = true;
                    break;
                }
            }
            for (const auto& hole : tree.holes) {
                for (const auto& coord : hole.coords) {
                    if (std::isnan(coord.first) || std::isnan(coord.second) || std::isinf(coord.first) || std::isinf(coord.second)) {
                        has_invalid = true;
                        break;
                    }
                }
                if (has_invalid) break;
            }
            if (has_invalid) continue;
            addPolygonOutlineToMarker(tree, marker);
        }
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createTreePointsMarker() const {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_tree_points";
        marker.id = 5;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.25;
        marker.color.r = 0.1f;
        marker.color.g = 0.5f;
        marker.color.b = 0.2f;
        marker.color.a = 0.9f;
        const int circle_segments = 24;
        const float r = tree_point_radius_meters_;
        for (const auto& pt : tree_points_) {
            if (std::isnan(pt.first) || std::isnan(pt.second) || std::isinf(pt.first) || std::isinf(pt.second)) continue;
            for (int i = 0; i < circle_segments; ++i) {
                float a1 = 2.0f * static_cast<float>(M_PI) * i / circle_segments;
                float a2 = 2.0f * static_cast<float>(M_PI) * (i + 1) / circle_segments;
                geometry_msgs::msg::Point p1, p2;
                p1.x = pt.first + r * std::cos(a1);
                p1.y = pt.second + r * std::sin(a1);
                p1.z = 0.0;
                p2.x = pt.first + r * std::cos(a2);
                p2.y = pt.second + r * std::sin(a2);
                p2.z = 0.0;
                marker.points.push_back(p1);
                marker.points.push_back(p2);
            }
        }
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createCyclewayMarker(const std::vector<Geometry2D>& cycleways) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_cycleways";
        marker.id = 11;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.12;
        marker.color.r = 0.0f; marker.color.g = 0.5f; marker.color.b = 1.0f; marker.color.a = 1.0f;
        for (const auto& cw : cycleways) {
            if (cw.coords.size() < 2) continue;
            addPolylineBandOutlineToMarker(cw.coords, cycleway_width_meters_, marker);
        }
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createWallMarker(const std::vector<Geometry2D>& walls) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_walls";
        marker.id = 12;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.12;
        marker.color.r = 0.5f; marker.color.g = 0.5f; marker.color.b = 0.5f; marker.color.a = 1.0f;
        for (const auto& w : walls) {
            if (w.coords.size() < 2) continue;
            addPolylineBandOutlineToMarker(w.coords, wall_width_meters_, marker);
        }
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createWaterMarker(const std::vector<Geometry2D>& water) {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_water";
        marker.id = 13;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.25;
        marker.color.r = 0.0f; marker.color.g = 0.0f; marker.color.b = 1.0f; marker.color.a = 0.8f;
        for (const auto& wa : water) {
            addPolygonOutlineToMarker(wa, marker);
        }
        return marker;
    }

    visualization_msgs::msg::Marker OSMVisualizer::createPolePointsMarker() const {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = frame_id_;
        marker.header.stamp = node_->now();
        marker.ns = "osm_pole_points";
        marker.id = 14;
        marker.type = visualization_msgs::msg::Marker::LINE_LIST;
        marker.action = visualization_msgs::msg::Marker::ADD;
        marker.pose.orientation.w = 1.0;
        marker.scale.x = 0.2;
        marker.color.r = 0.8f; marker.color.g = 0.4f; marker.color.b = 0.0f; marker.color.a = 1.0f;
        const float r = pole_point_radius_meters_;
        const int circle_segments = 16;
        for (const auto& pt : pole_points_) {
            if (std::isnan(pt.first) || std::isnan(pt.second) || std::isinf(pt.first) || std::isinf(pt.second)) continue;
            for (int i = 0; i < circle_segments; ++i) {
                float a1 = 2.0f * static_cast<float>(M_PI) * i / circle_segments;
                float a2 = 2.0f * static_cast<float>(M_PI) * (i + 1) / circle_segments;
                geometry_msgs::msg::Point p1, p2;
                p1.x = pt.first + r * std::cos(a1); p1.y = pt.second + r * std::sin(a1); p1.z = 0.0;
                p2.x = pt.first + r * std::cos(a2); p2.y = pt.second + r * std::sin(a2); p2.z = 0.0;
                marker.points.push_back(p1); marker.points.push_back(p2);
            }
        }
        return marker;
    }

    void OSMVisualizer::publish() {
        // RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: publish() called, buildings_.size()=" << buildings_.size());
        
        if (!pub_) {
            RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Cannot publish - publisher is null!");
            return;
        }
        
        if (!node_) {
            RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Cannot publish - node is null!");
            return;
        }
        
        // RCLCPP_WARN_STREAM(node_->get_logger(), "CHECKPOINT: Publisher and node are valid, creating marker array");
        visualization_msgs::msg::MarkerArray marker_array;
        
        if (!buildings_.empty()) {
            auto marker = createBuildingMarker(buildings_);
            marker_array.markers.push_back(marker);
            // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << buildings_.size() << " buildings with " << marker.points.size() << " line points");
        } else {
            // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: No buildings loaded! buildings_.size()=" << buildings_.size());
        }
        
        if (!roads_.empty()) {
            auto marker = createRoadMarker(roads_);
            marker_array.markers.push_back(marker);
            // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << roads_.size() << " roads with " << marker.points.size() << " line points");
        } else {
            // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: No roads loaded! roads_.size()=" << roads_.size());
        }

        if (!sidewalks_.empty()) {
            auto marker = createSidewalkMarker(sidewalks_);
            marker_array.markers.push_back(marker);
        }

        if (!parking_.empty()) {
            auto marker = createParkingMarker(parking_);
            marker_array.markers.push_back(marker);
        }

        if (!fences_.empty()) {
            auto marker = createFenceMarker(fences_);
            marker_array.markers.push_back(marker);
        }

        if (!stairs_.empty()) {
            auto marker = createStairsMarker(stairs_);
            marker_array.markers.push_back(marker);
        }
        
        if (!grasslands_.empty()) {
            auto marker = createGrasslandMarker(grasslands_);
            marker_array.markers.push_back(marker);
            // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM: Added " << grasslands_.size() << " grasslands with " << marker.points.size() << " line points");
        }
        
        if (!trees_.empty()) {
            auto marker = createTreeMarker(trees_);
            marker_array.markers.push_back(marker);
        }
        if (!forests_.empty()) {
            auto marker = createForestMarker(forests_);
            marker_array.markers.push_back(marker);
        }
        if (!tree_points_.empty()) {
            auto marker = createTreePointsMarker();
            marker_array.markers.push_back(marker);
        }
        if (!cycleways_.empty()) {
            marker_array.markers.push_back(createCyclewayMarker(cycleways_));
        }
        if (!walls_.empty()) {
            marker_array.markers.push_back(createWallMarker(walls_));
        }
        if (!water_.empty()) {
            marker_array.markers.push_back(createWaterMarker(water_));
        }
        if (!pole_points_.empty()) {
            marker_array.markers.push_back(createPolePointsMarker());
        }
        
        if (marker_array.markers.empty()) {
            // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: No markers to publish! (buildings=" << buildings_.size() << ", roads=" << roads_.size() << ", ... trees=" << trees_.size() << ", forests=" << forests_.size() << ", tree_points=" << tree_points_.size() << ")");
            return;
        }
        
        // Set lifetime for all markers - use zero duration (never expire) like ROS1
        // In ROS2, zero duration means never expire
        rclcpp::Duration zero_lifetime(0, 0); // Never expire (same as ROS1 ros::Duration())
        for (auto& marker : marker_array.markers) {
            marker.lifetime = zero_lifetime;
            // Ensure frame_id is set correctly
            marker.header.frame_id = frame_id_;
            marker.header.stamp = rclcpp::Time(0);
        }
        
        int total_points = 0;
        for (const auto& m : marker_array.markers) {
            total_points += m.points.size();
        }
        // RCLCPP_INFO_STREAM(node_->get_logger(), "OSMVisualizer: Publishing " << marker_array.markers.size() << " marker(s) (buildings + roads) with " << total_points << " total line points to topic " << topic_ << " in frame " << frame_id_);
        
        // Log marker details and bounding box for debugging
        if (!marker_array.markers.empty()) {
            const auto& first_marker = marker_array.markers[0];
            // RCLCPP_INFO_STREAM(node_->get_logger(), "First marker: ns=" << first_marker.ns << ", id=" << first_marker.id << ", type=" << first_marker.type << ", points=" << first_marker.points.size() << ", frame=" << first_marker.header.frame_id);
            
            // Calculate bounding box of marker points
            if (!first_marker.points.empty()) {
                float min_x = std::numeric_limits<float>::max();
                float max_x = std::numeric_limits<float>::lowest();
                float min_y = std::numeric_limits<float>::max();
                float max_y = std::numeric_limits<float>::lowest();
                for (const auto& pt : first_marker.points) {
                    min_x = std::min(min_x, static_cast<float>(pt.x));
                    max_x = std::max(max_x, static_cast<float>(pt.x));
                    min_y = std::min(min_y, static_cast<float>(pt.y));
                    max_y = std::max(max_y, static_cast<float>(pt.y));
                }
                // RCLCPP_INFO_STREAM(node_->get_logger(), "Marker bounds: X=[" << min_x << ", " << max_x << "], Y=[" << min_y << ", " << max_y << "]");
                // RCLCPP_INFO_STREAM(node_->get_logger(), "First point: [" << first_marker.points[0].x << ", " << first_marker.points[0].y << ", " << first_marker.points[0].z << "]");
            }
        }
        
        // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: About to publish MarkerArray with " << marker_array.markers.size() << " markers");
        // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Publisher valid: " << (pub_ ? "YES" : "NO"));
        // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Topic: " << topic_);
        
        try {
            pub_->publish(marker_array);
            // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Successfully published MarkerArray to " << topic_ << " with " << marker_array.markers.size() << " markers");
        } catch (const std::exception& e) {
            RCLCPP_ERROR_STREAM(node_->get_logger(), "OSMVisualizer: Exception while publishing: " << e.what());
        }
    }

    void OSMVisualizer::startPeriodicPublishing(double rate) {
        if (!pub_) {
            RCLCPP_WARN(node_->get_logger(), "OSMVisualizer: Cannot start periodic publishing - no publisher (topic was empty)");
            return;
        }
        if (rate <= 0.0) {
            // RCLCPP_WARN(node_->get_logger(), "OSMVisualizer: Invalid publishing rate, using default 1.0 Hz");
            rate = 1.0;
        }
        // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Creating timer for periodic publishing at " << rate << " Hz");
        // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Node pointer: " << node_.get() << ", this pointer: " << this);
        // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Publisher pointer: " << pub_.get() << ", Publisher count: " << pub_.use_count());
        
        publish_timer_ = node_->create_wall_timer(
            std::chrono::milliseconds(static_cast<int>(1000.0 / rate)),
            [this]() {
                this->timerCallback();
            });
        if (!publish_timer_) {
            // RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Failed to create publishing timer!");
        } else {
            // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Timer created successfully, periodic publishing started at " << rate << " Hz");
            // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Timer will fire every " << (1000.0 / rate) << " ms");
            // // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Timer pointer: " << publish_timer_.get() << ", Timer count: " << publish_timer_.use_count());
        }
    }

    void OSMVisualizer::timerCallback() {
        try {
            // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Timer callback triggered, republishing markers (buildings=" << buildings_.size() << ", roads=" << roads_.size() << ", sidewalks=" << sidewalks_.size() << ", parking=" << parking_.size() << ", grasslands=" << grasslands_.size() << ", trees=" << trees_.size() << ", tree_points=" << tree_points_.size() << ")");
            if (!node_) {
                RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Timer callback - node_ is null!");
                return;
            }
            if (!pub_) {
                RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Timer callback - pub_ is null!");
                return;
            }
            if (!publish_timer_) {
                RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Timer callback - publish_timer_ is null!");
                return;
            }
            // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: About to call publish(), pub_ count: " << pub_.use_count());
            publish();
            // RCLCPP_WARN_STREAM(node_->get_logger(), "OSMVisualizer: Timer callback completed successfully, markers published");
        } catch (const std::exception& e) {
            RCLCPP_ERROR_STREAM(node_->get_logger(), "OSMVisualizer: Exception in timer callback: " << e.what());
        } catch (...) {
            RCLCPP_ERROR(node_->get_logger(), "OSMVisualizer: Unknown exception in timer callback!");
        }
    }

    void OSMVisualizer::transformToFirstPoseOrigin(const Eigen::Matrix4d& first_pose) {
        // Prevent multiple transformations
        if (transformed_) {
            // RCLCPP_WARN(node_->get_logger(), "OSM data has already been transformed. Skipping additional transformation to prevent double transformation.");
            return;
        }
        
        // OSM origin = world-frame origin GPS (matching Python MCD_ORIGIN_LATLON).
        // OSM coordinates are already in the world frame. Apply first_pose_inverse
        // to convert to the normalized frame (first pose at origin).
        Eigen::Matrix4d first_pose_inverse = first_pose.inverse();
        
        auto transformPoint = [&first_pose_inverse](float& x, float& y) {
            Eigen::Vector4d point(static_cast<double>(x), static_cast<double>(y), 0.0, 1.0);
            Eigen::Vector4d transformed = first_pose_inverse * point;
            x = static_cast<float>(transformed(0));
            y = static_cast<float>(transformed(1));
        };
        
        for (auto& building : buildings_) {
            for (auto& coord : building.coords) {
                transformPoint(coord.first, coord.second);
            }
            for (auto& hole : building.holes) {
                for (auto& coord : hole.coords) {
                    transformPoint(coord.first, coord.second);
                }
            }
        }
        
        // Transform roads
        for (auto& road : roads_) {
            for (auto& coord : road.coords) {
                transformPoint(coord.first, coord.second);
            }
        }

        // Transform sidewalks and parking
        for (auto& sidewalk : sidewalks_) {
            for (auto& coord : sidewalk.coords) {
                transformPoint(coord.first, coord.second);
            }
        }
        for (auto& park : parking_) {
            for (auto& coord : park.coords) {
                transformPoint(coord.first, coord.second);
            }
            for (auto& hole : park.holes) {
                for (auto& coord : hole.coords) {
                    transformPoint(coord.first, coord.second);
                }
            }
        }
        for (auto& fence : fences_) {
            for (auto& coord : fence.coords) {
                transformPoint(coord.first, coord.second);
            }
        }
        for (auto& stair : stairs_) {
            for (auto& coord : stair.coords) {
                transformPoint(coord.first, coord.second);
            }
        }
        
        // Transform grasslands, trees, forests
        for (auto& grassland : grasslands_) {
            for (auto& coord : grassland.coords) {
                transformPoint(coord.first, coord.second);
            }
            for (auto& hole : grassland.holes) {
                for (auto& coord : hole.coords) {
                    transformPoint(coord.first, coord.second);
                }
            }
        }
        for (auto& tree : trees_) {
            for (auto& coord : tree.coords) {
                transformPoint(coord.first, coord.second);
            }
            for (auto& hole : tree.holes) {
                for (auto& coord : hole.coords) {
                    transformPoint(coord.first, coord.second);
                }
            }
        }
        for (auto& forest : forests_) {
            for (auto& coord : forest.coords) {
                transformPoint(coord.first, coord.second);
            }
            for (auto& hole : forest.holes) {
                for (auto& coord : hole.coords) {
                    transformPoint(coord.first, coord.second);
                }
            }
        }
        for (auto& pt : tree_points_) {
            transformPoint(pt.first, pt.second);
        }
        for (auto& cw : cycleways_) {
            for (auto& coord : cw.coords) transformPoint(coord.first, coord.second);
        }
        for (auto& w : walls_) {
            for (auto& coord : w.coords) transformPoint(coord.first, coord.second);
        }
        for (auto& wa : water_) {
            for (auto& coord : wa.coords) transformPoint(coord.first, coord.second);
            for (auto& hole : wa.holes)
                for (auto& coord : hole.coords) transformPoint(coord.first, coord.second);
        }
        for (auto& pt : pole_points_) {
            transformPoint(pt.first, pt.second);
        }
        
        // Log bounding box after transformation
        if (!buildings_.empty() || !roads_.empty()) {
            float min_x_after = std::numeric_limits<float>::max();
            float max_x_after = std::numeric_limits<float>::lowest();
            float min_y_after = std::numeric_limits<float>::max();
            float max_y_after = std::numeric_limits<float>::lowest();
            for (const auto& building : buildings_) {
                for (const auto& coord : building.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
                for (const auto& hole : building.holes) {
                    for (const auto& coord : hole.coords) {
                        min_x_after = std::min(min_x_after, coord.first);
                        max_x_after = std::max(max_x_after, coord.first);
                        min_y_after = std::min(min_y_after, coord.second);
                        max_y_after = std::max(max_y_after, coord.second);
                    }
                }
            }
            for (const auto& road : roads_) {
                for (const auto& coord : road.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
            }
            for (const auto& sw : sidewalks_) {
                for (const auto& coord : sw.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
            }
            for (const auto& p : parking_) {
                for (const auto& coord : p.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
                for (const auto& hole : p.holes) {
                    for (const auto& coord : hole.coords) {
                        min_x_after = std::min(min_x_after, coord.first);
                        max_x_after = std::max(max_x_after, coord.first);
                        min_y_after = std::min(min_y_after, coord.second);
                        max_y_after = std::max(max_y_after, coord.second);
                    }
                }
            }
            for (const auto& f : fences_) {
                for (const auto& coord : f.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
            }
            for (const auto& st : stairs_) {
                for (const auto& coord : st.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
            }
            for (const auto& g : grasslands_) {
                for (const auto& coord : g.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
                for (const auto& hole : g.holes) {
                    for (const auto& coord : hole.coords) {
                        min_x_after = std::min(min_x_after, coord.first);
                        max_x_after = std::max(max_x_after, coord.first);
                        min_y_after = std::min(min_y_after, coord.second);
                        max_y_after = std::max(max_y_after, coord.second);
                    }
                }
            }
            for (const auto& t : trees_) {
                for (const auto& coord : t.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
                for (const auto& hole : t.holes) {
                    for (const auto& coord : hole.coords) {
                        min_x_after = std::min(min_x_after, coord.first);
                        max_x_after = std::max(max_x_after, coord.first);
                        min_y_after = std::min(min_y_after, coord.second);
                        max_y_after = std::max(max_y_after, coord.second);
                    }
                }
            }
            for (const auto& f : forests_) {
                for (const auto& coord : f.coords) {
                    min_x_after = std::min(min_x_after, coord.first);
                    max_x_after = std::max(max_x_after, coord.first);
                    min_y_after = std::min(min_y_after, coord.second);
                    max_y_after = std::max(max_y_after, coord.second);
                }
                for (const auto& hole : f.holes) {
                    for (const auto& coord : hole.coords) {
                        min_x_after = std::min(min_x_after, coord.first);
                        max_x_after = std::max(max_x_after, coord.first);
                        min_y_after = std::min(min_y_after, coord.second);
                        max_y_after = std::max(max_y_after, coord.second);
                    }
                }
            }
            // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM geometries AFTER transform - Bounds: [" << min_x_after << ", " << min_y_after << "] to [" << max_x_after << ", " << max_y_after << "]");
        }
        
        transformed_ = true;
        
        // RCLCPP_INFO_STREAM(node_->get_logger(), "OSM geometries (buildings, roads, grasslands, trees) transformed to first pose origin frame.");
    }

} // namespace semantic_bki
