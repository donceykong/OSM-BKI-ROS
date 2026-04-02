#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

namespace osm_bki {

    /// 2D polygon (list of (x,y) vertices; same convention as OSMVisualizer).
    /// Optional holes (inner rings): points inside a hole are considered outside the filled polygon.
    struct Geometry2D {
        std::vector<std::pair<float, float>> coords;
        std::vector<Geometry2D> holes;  // Inner rings (cutouts); empty = no holes
    };

    /// Ray-casting point-in-polygon test (returns true if (px,py) is inside poly).
    inline bool point_in_polygon(float px, float py, const Geometry2D& poly) {
        const auto& c = poly.coords;
        if (c.size() < 3) return false;
        int n = static_cast<int>(c.size());
        bool inside = false;
        for (int i = 0, j = n - 1; i < n; j = i++) {
            float xi = c[i].first, yi = c[i].second;
            float xj = c[j].first, yj = c[j].second;
            if (((yi > py) != (yj > py)) &&
                (px < (xj - xi) * (py - yi) / (yj - yi) + xi))
                inside = !inside;
        }
        return inside;
    }

    /// Squared distance from point (px,py) to segment (ax,ay)-(bx,by).
    inline float segment_distance_sq(float px, float py, float ax, float ay, float bx, float by) {
        float dx = bx - ax, dy = by - ay;
        float len_sq = dx * dx + dy * dy;
        if (len_sq < 1e-12f) {
            float dx2 = px - ax, dy2 = py - ay;
            return dx2 * dx2 + dy2 * dy2;
        }
        float t = ((px - ax) * dx + (py - ay) * dy) / len_sq;
        t = std::max(0.f, std::min(1.f, t));
        float qx = ax + t * dx, qy = ay + t * dy;
        float ddx = px - qx, ddy = py - qy;
        return ddx * ddx + ddy * ddy;
    }

    /// Point-in-polygon with holes: inside filled region iff inside outer AND outside all holes.
    inline bool point_in_polygon_with_holes(float px, float py, const Geometry2D& poly) {
        if (!point_in_polygon(px, py, poly)) return false;
        for (const auto& hole : poly.holes) {
            if (point_in_polygon(px, py, hole)) return false;  // Inside a hole = outside filled
        }
        return true;
    }

    /// Minimum distance from (px,py) to a single ring boundary (used for outer and holes).
    inline float distance_to_ring_boundary_sq(float px, float py, const Geometry2D& ring, bool* out_inside) {
        const auto& c = ring.coords;
        if (c.size() < 2) {
            if (out_inside) *out_inside = false;
            return std::numeric_limits<float>::max();
        }
        bool inside = point_in_polygon(px, py, ring);
        if (out_inside) *out_inside = inside;
        float min_d_sq = std::numeric_limits<float>::max();
        int n = static_cast<int>(c.size());
        for (int i = 0, j = n - 1; i < n; j = i++) {
            float d_sq = segment_distance_sq(px, py,
                c[j].first, c[j].second,
                c[i].first, c[i].second);
            if (d_sq < min_d_sq) min_d_sq = d_sq;
        }
        return min_d_sq;
    }

    /// Minimum distance from (px,py) to polygon boundary. Returns signed distance:
    /// negative = inside filled region, positive = outside. Supports polygons with holes.
    inline float distance_to_polygon_boundary(float px, float py, const Geometry2D& poly) {
        const auto& c = poly.coords;
        if (c.size() < 2) return std::numeric_limits<float>::max();

        if (poly.holes.empty()) {
            bool inside = point_in_polygon(px, py, poly);
            bool dummy;
            float min_d_sq = distance_to_ring_boundary_sq(px, py, poly, &dummy);
            float d = std::sqrt(min_d_sq);
            return inside ? -d : d;
        }

        // Polygon with holes: boundary = outer ring + all hole rings
        bool inside_outer = point_in_polygon(px, py, poly);
        bool inside_any_hole = false;
        for (const auto& hole : poly.holes) {
            if (point_in_polygon(px, py, hole)) {
                inside_any_hole = true;
                break;
            }
        }
        bool inside_filled = inside_outer && !inside_any_hole;

        float min_d_sq = distance_to_ring_boundary_sq(px, py, poly, nullptr);
        for (const auto& hole : poly.holes) {
            bool in_hole;
            float d_sq = distance_to_ring_boundary_sq(px, py, hole, &in_hole);
            if (d_sq < min_d_sq) min_d_sq = d_sq;
        }
        float d = std::sqrt(min_d_sq);
        return inside_filled ? -d : d;
    }

    /// Minimum distance from (px,py) to polyline (open path, not closed polygon).
    /// Returns distance to nearest segment.
    inline float distance_to_polyline(float px, float py, const Geometry2D& polyline) {
        const auto& c = polyline.coords;
        if (c.size() < 2) return std::numeric_limits<float>::max();
        float min_d_sq = std::numeric_limits<float>::max();
        for (size_t i = 0; i < c.size() - 1; ++i) {
            float d_sq = segment_distance_sq(px, py,
                c[i].first, c[i].second,
                c[i+1].first, c[i+1].second);
            if (d_sq < min_d_sq) min_d_sq = d_sq;
        }
        return std::sqrt(min_d_sq);
    }

    /// Signed distance from (px,py) to a polyline "band" of given width:
    /// negative = inside the width band, positive = outside.
    inline float distance_to_polyline_band_signed(float px, float py, const Geometry2D& polyline, float width) {
        float d = distance_to_polyline(px, py, polyline);
        float half_w = std::max(0.f, width) * 0.5f;
        return d - half_w;
    }

    /// Signed distance from (px,py) to circle: negative = inside, positive = outside.
    /// Returns d - radius where d = distance from (px,py) to circle center.
    inline float distance_to_circle_signed(float px, float py, float cx, float cy, float radius) {
        float dx = px - cx, dy = py - cy;
        float d = std::sqrt(dx * dx + dy * dy);
        return d - radius;
    }

    /// Minimum distance from (px,py) to nearest point in point list.
    inline float distance_to_points(float px, float py, const std::vector<std::pair<float, float>>& points) {
        if (points.empty()) return std::numeric_limits<float>::max();
        float min_d_sq = std::numeric_limits<float>::max();
        for (const auto& pt : points) {
            float dx = px - pt.first, dy = py - pt.second;
            float d_sq = dx * dx + dy * dy;
            if (d_sq < min_d_sq) min_d_sq = d_sq;
        }
        return std::sqrt(min_d_sq);
    }

    /// OSM prior value: inside -> 1.0, outside -> decay by distance (linear dropoff).
    /// signed_d: result of distance_to_polygon_boundary (negative inside, positive outside).
    /// decay_m: distance in meters over which prior drops from 1 to 0.
    inline float osm_prior_from_signed_distance(float signed_d, float decay_m) {
        if (decay_m <= 0.f) return signed_d <= 0.f ? 1.f : 0.f;
        if (signed_d <= 0.f) return 1.f;
        return std::max(0.f, 1.f - signed_d / decay_m);
    }

    /// OSM prior value for polylines/points: decay by distance (always positive distance).
    /// d: distance to nearest polyline segment or point.
    /// decay_m: distance in meters over which prior drops from 1 to 0.
    inline float osm_prior_from_distance(float d, float decay_m) {
        if (decay_m <= 0.f) return 0.f;
        return std::max(0.f, 1.f - d / decay_m);
    }

    /// Check if a Geometry2D's bounding box overlaps a circle (cx, cy, radius).
    inline bool geometry_overlaps_circle(const Geometry2D &geom, float cx, float cy, float radius) {
        if (geom.coords.empty()) return false;
        // Compute AABB of geometry
        float min_x = std::numeric_limits<float>::max(), max_x = std::numeric_limits<float>::lowest();
        float min_y = std::numeric_limits<float>::max(), max_y = std::numeric_limits<float>::lowest();
        for (const auto &p : geom.coords) {
            if (p.first < min_x) min_x = p.first;
            if (p.first > max_x) max_x = p.first;
            if (p.second < min_y) min_y = p.second;
            if (p.second > max_y) max_y = p.second;
        }
        // Closest point on AABB to circle center
        float nearest_x = std::max(min_x, std::min(cx, max_x));
        float nearest_y = std::max(min_y, std::min(cy, max_y));
        float dx = cx - nearest_x, dy = cy - nearest_y;
        return (dx * dx + dy * dy) <= (radius * radius);
    }

    /// Check if a point (px, py) is within radius of circle (cx, cy, radius).
    inline bool point_overlaps_circle(float px, float py, float cx, float cy, float radius) {
        float dx = px - cx, dy = py - cy;
        return (dx * dx + dy * dy) <= (radius * radius);
    }
}
