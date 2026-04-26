#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>
#include <map>
#include <unordered_map>
#include <cstdint>

namespace osm_bki {

    /// 2D polygon (list of (x,y) vertices; same convention as OSMVisualizer).
    /// Optional holes (inner rings): points inside a hole are considered outside the filled polygon.
    struct Geometry2D {
        std::vector<std::pair<float, float>> coords;
        std::vector<Geometry2D> holes;  // Inner rings (cutouts); empty = no holes
        /// Cached AABB over `coords`. Default values cause bbox_distance_sq to return
        /// FLT_MAX, so an unpopulated bbox always fails a prefilter — call compute_bbox
        /// after constructing the geometry to enable the prefilter.
        float bbox_min_x = std::numeric_limits<float>::max();
        float bbox_max_x = std::numeric_limits<float>::lowest();
        float bbox_min_y = std::numeric_limits<float>::max();
        float bbox_max_y = std::numeric_limits<float>::lowest();
    };

    /// Compute and cache the AABB on a Geometry2D (and any holes). Call once after
    /// the geometry's `coords` are finalized.
    inline void compute_bbox(Geometry2D& g) {
        g.bbox_min_x = std::numeric_limits<float>::max();
        g.bbox_max_x = std::numeric_limits<float>::lowest();
        g.bbox_min_y = std::numeric_limits<float>::max();
        g.bbox_max_y = std::numeric_limits<float>::lowest();
        for (const auto& p : g.coords) {
            if (p.first  < g.bbox_min_x) g.bbox_min_x = p.first;
            if (p.first  > g.bbox_max_x) g.bbox_max_x = p.first;
            if (p.second < g.bbox_min_y) g.bbox_min_y = p.second;
            if (p.second > g.bbox_max_y) g.bbox_max_y = p.second;
        }
        for (auto& hole : g.holes) compute_bbox(hole);
    }

    /// Squared distance from (px,py) to the geometry's cached AABB. Returns 0 when
    /// (px,py) is inside the bbox, so a `> decay_sq` test never skips a polygon
    /// that contains the query point.
    inline float bbox_distance_sq(float px, float py, const Geometry2D& g) {
        float dx = 0.f, dy = 0.f;
        if      (px < g.bbox_min_x) dx = g.bbox_min_x - px;
        else if (px > g.bbox_max_x) dx = px - g.bbox_max_x;
        if      (py < g.bbox_min_y) dy = g.bbox_min_y - py;
        else if (py > g.bbox_max_y) dy = py - g.bbox_max_y;
        return dx * dx + dy * dy;
    }

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

    // ═══════════════════════════════════════════════════════════════════════
    // Network polygon building (polylines → explicit band polygons with
    // OSM2World-style junction cuts: orthogonal / angle-bisector / edge-intersection)
    // ═══════════════════════════════════════════════════════════════════════

    /// Per-highway-type default widths (OSM2World RoadModule, metres). Returns fallback for unmapped types.
    inline float highway_default_width(const std::string& hw_type, float fallback = 4.0f) {
        if (hw_type == "motorway"  || hw_type == "motorway_link")  return 8.75f;
        if (hw_type == "trunk"     || hw_type == "trunk_link"  ||
            hw_type == "primary"   || hw_type == "primary_link" ||
            hw_type == "secondary" || hw_type == "secondary_link") return 7.0f;
        if (hw_type == "service") return 3.5f;
        if (hw_type == "track")   return 2.5f;
        return fallback;
    }

    /// Raw polyline with effective width and endpoint node IDs. Input to build_network_polygons.
    struct RawNetLine {
        std::vector<std::pair<float,float>> coords;
        float   width{4.0f};
        int64_t start_nid{0};
        int64_t end_nid{0};
    };

    /// Convert raw centerline polylines to explicit band polygons with junction geometry.
    /// Returns {segment_polygons, junction_fill_polygons}.
    inline std::pair<std::vector<Geometry2D>, std::vector<Geometry2D>>
    build_network_polygons(
            const std::vector<RawNetLine>& raw_lines,
            const std::unordered_map<int64_t, std::pair<float,float>>& node_xy)
    {
        using XY  = std::pair<float,float>;
        using Cut = std::pair<XY,XY>;         // {left_contact, right_contact}
        using SK  = std::pair<int,bool>;      // {seg_idx, is_start}

        // ── geometry lambdas ────────────────────────────────────────────────
        auto rn2d = [](XY p0, XY p1) -> XY {
            float dx = p1.first-p0.first, dy = p1.second-p0.second;
            float L  = std::sqrt(dx*dx+dy*dy);
            return L < 1e-9f ? XY{1.f,0.f} : XY{dy/L, -dx/L};
        };

        auto li2d = [](float px, float py, float dx, float dy,
                       float qx, float qy, float ex, float ey,
                       float& ox, float& oy) -> bool {
            float cross = dx*ey - dy*ex;
            if (std::abs(cross) < 1e-9f) return false;
            float t = ((qx-px)*ey - (qy-py)*ex) / cross;
            ox = px+t*dx; oy = py+t*dy;
            return true;
        };

        // Away-from-junction direction for a segment endpoint
        auto away_dir = [](const std::vector<XY>& coords, bool is_start) -> XY {
            float dx, dy;
            if (is_start) {
                dx = coords[1].first  - coords[0].first;
                dy = coords[1].second - coords[0].second;
            } else {
                dx = coords[coords.size()-2].first  - coords.back().first;
                dy = coords[coords.size()-2].second - coords.back().second;
            }
            float L = std::sqrt(dx*dx+dy*dy);
            return L > 1e-9f ? XY{dx/L,dy/L} : XY{1.f,0.f};
        };

        // Band polygon from polyline with optional endpoint cuts
        auto poly_coords = [&rn2d](const std::vector<XY>& coords, float hw,
                                    const Cut* sc, const Cut* ec) -> std::vector<XY> {
            if (coords.size() < 2) return {};
            std::vector<XY> lo, ro;

            if (sc) { lo.push_back(sc->first); ro.push_back(sc->second); }
            else {
                XY n = rn2d(coords[0], coords[1]);
                lo.push_back({coords[0].first-n.first*hw, coords[0].second-n.second*hw});
                ro.push_back({coords[0].first+n.first*hw, coords[0].second+n.second*hw});
            }
            for (size_t i = 1; i+1 < coords.size(); ++i) {
                XY n0 = rn2d(coords[i-1], coords[i]);
                XY n1 = rn2d(coords[i],   coords[i+1]);
                float mx = n0.first+n1.first, my = n0.second+n1.second;
                float ml = std::sqrt(mx*mx+my*my);
                if (ml < 1e-9f) { mx = n0.first; my = n0.second; }
                else { mx /= ml; my /= ml; }
                float dot = n0.first*mx + n0.second*my;
                float s   = (dot > 1e-6f) ? std::min(5.f, 1.f/dot) : 5.f;
                lo.push_back({coords[i].first-mx*hw*s, coords[i].second-my*hw*s});
                ro.push_back({coords[i].first+mx*hw*s, coords[i].second+my*hw*s});
            }
            if (ec) { lo.push_back(ec->first); ro.push_back(ec->second); }
            else {
                XY n = rn2d(coords[coords.size()-2], coords.back());
                lo.push_back({coords.back().first-n.first*hw, coords.back().second-n.second*hw});
                ro.push_back({coords.back().first+n.first*hw, coords.back().second+n.second*hw});
            }
            std::vector<XY> poly(ro.begin(), ro.end());
            poly.insert(poly.end(), lo.rbegin(), lo.rend());
            poly.push_back(poly[0]);
            return poly;
        };

        // ── node → segment adjacency ─────────────────────────────────────
        std::unordered_map<int64_t, std::vector<SK>> node_segs;
        for (int i = 0; i < (int)raw_lines.size(); ++i) {
            if (raw_lines[i].coords.size() < 2) continue;
            node_segs[raw_lines[i].start_nid].push_back({i, true});
            node_segs[raw_lines[i].end_nid  ].push_back({i, false});
        }

        std::map<SK, Cut> cuts;
        std::vector<Geometry2D> junction_polys;

        static const float PAR_THRESH = 3.14159265f / 18.f;  // ~10 deg

        for (auto& kv : node_segs) {
            int64_t nid = kv.first;
            std::vector<SK>& conns = kv.second;

            // Filter valid (≥2 coords)
            std::vector<SK> valid;
            for (size_t ci = 0; ci < conns.size(); ++ci)
                if (raw_lines[conns[ci].first].coords.size() >= 2)
                    valid.push_back(conns[ci]);
            int n = (int)valid.size();
            if (n == 0) continue;

            // ── Dead end: orthogonal cut ───────────────────────────────────
            if (n == 1) {
                int si = valid[0].first; bool is_start = valid[0].second;
                float hw = raw_lines[si].width * 0.5f;
                XY nm = is_start ? rn2d(raw_lines[si].coords[0], raw_lines[si].coords[1])
                                 : rn2d(raw_lines[si].coords[raw_lines[si].coords.size()-2],
                                        raw_lines[si].coords.back());
                const XY& c = is_start ? raw_lines[si].coords.front() : raw_lines[si].coords.back();
                cuts[{si, is_start}] = {
                    {c.first-nm.first*hw, c.second-nm.second*hw},
                    {c.first+nm.first*hw, c.second+nm.second*hw}
                };
                continue;
            }

            // ── Connector: angle-bisector cut ─────────────────────────────
            if (n == 2) {
                int si1 = valid[0].first; bool s1 = valid[0].second;
                int si2 = valid[1].first; bool s2 = valid[1].second;
                float hw1 = raw_lines[si1].width * 0.5f;
                float hw2 = raw_lines[si2].width * 0.5f;

                // Toward junction from seg1, away from junction for seg2
                auto toward = [&away_dir](const std::vector<XY>& co, bool is_start) -> XY {
                    XY a = away_dir(co, is_start);
                    return {-a.first, -a.second};
                };
                XY in_v  = toward(raw_lines[si1].coords, s1);
                XY out_v = away_dir(raw_lines[si2].coords, s2);

                float cvx = out_v.first-in_v.first, cvy = out_v.second-in_v.second;
                float cl  = std::sqrt(cvx*cvx+cvy*cvy);
                if (cl < 1e-6f) { cvx = -in_v.second; cvy = in_v.first; }
                else            { cvx /= cl; cvy /= cl; }
                if (in_v.first*cvy - in_v.second*cvx <= 0.f) { cvx=-cvx; cvy=-cvy; }

                const XY& p1 = s1 ? raw_lines[si1].coords.front() : raw_lines[si1].coords.back();
                const XY& p2 = s2 ? raw_lines[si2].coords.front() : raw_lines[si2].coords.back();

                // Store as {LEFT-of-forward, RIGHT-of-forward}. Use the forward tangent
                // AT THE JUNCTION END of the polyline so multi-node (curved) polylines get
                // the correct local direction.
                auto store_cut_2seg = [&](int si, bool is_start, const XY& p, float hw) {
                    const auto& co = raw_lines[si].coords;
                    XY f;
                    if (is_start) {
                        f = {co[1].first - co[0].first, co[1].second - co[0].second};
                    } else {
                        f = {co.back().first  - co[co.size()-2].first,
                             co.back().second - co[co.size()-2].second};
                    }
                    float L = std::sqrt(f.first*f.first + f.second*f.second);
                    if (L > 1e-9f) { f.first/=L; f.second/=L; }
                    XY rn = {f.second, -f.first};           // right-of-forward
                    float dot = cvx*rn.first + cvy*rn.second;
                    XY a = {p.first - cvx*hw, p.second - cvy*hw};
                    XY b = {p.first + cvx*hw, p.second + cvy*hw};
                    cuts[{si, is_start}] = (dot >= 0.f) ? Cut{a, b} : Cut{b, a};
                };
                store_cut_2seg(si1, s1, p1, hw1);
                store_cut_2seg(si2, s2, p2, hw2);
                continue;
            }

            // ── Junction: edge-intersection cut ───────────────────────────
            // Uses away_dir(coords, is_start) which returns the outward-from-junction tangent
            // at the junction end of the polyline. For multi-node (curved) polylines this is
            // the correct local direction; a shared coords[0]→coords[1] would be wrong for
            // segments whose end (not start) meets the junction.

            // Sort segments by toward-junction angle (= negated outward).
            std::sort(valid.begin(), valid.end(), [&](const SK& a, const SK& b) {
                XY da = away_dir(raw_lines[a.first].coords, a.second);
                XY db = away_dir(raw_lines[b.first].coords, b.second);
                return std::atan2(-da.second, -da.first) < std::atan2(-db.second, -db.first);
            });

            // Step 1: intersect left-edge[i] with right-edge[i+1]
            std::vector<bool> has_ix(n, false);
            std::vector<XY>   ixpts(n);
            for (int idx = 0; idx < n; ++idx) {
                int si  = valid[idx].first;         bool iss  = valid[idx].second;
                int ti  = valid[(idx+1)%n].first;   bool tss  = valid[(idx+1)%n].second;
                float s_hw = raw_lines[si].width * 0.5f;
                float t_hw = raw_lines[ti].width * 0.5f;

                // Direction toward the junction along the polyline end that meets it.
                // outward = away_dir(...); toward = -outward.
                XY s_out = away_dir(raw_lines[si].coords, iss);
                XY t_out = away_dir(raw_lines[ti].coords, tss);
                XY sd = {-s_out.first, -s_out.second};
                XY td = {-t_out.first, -t_out.second};

                // Right normal to direction
                XY srn = {sd.second, -sd.first};
                XY trn = {td.second, -td.first};

                const XY& sc2 = iss ? raw_lines[si].coords.front() : raw_lines[si].coords.back();
                const XY& tc2 = tss ? raw_lines[ti].coords.front() : raw_lines[ti].coords.back();

                // Left edge of s: sc - srn*hw; direction toward junction = -sd
                XY sep = {sc2.first-srn.first*s_hw, sc2.second-srn.second*s_hw};
                XY tep = {tc2.first+trn.first*t_hw, tc2.second+trn.second*t_hw};
                float sedx=-sd.first, sedy=-sd.second;
                float tedx=-td.first, tedy=-td.second;
                float dot = sedx*(-tedx) + sedy*(-tedy);
                if (std::acos(std::max(-1.f,std::min(1.f,dot))) < PAR_THRESH) continue;
                float ox,oy;
                if (li2d(sep.first,sep.second,sedx,sedy, tep.first,tep.second,tedx,tedy,ox,oy)) {
                    has_ix[idx]=true; ixpts[idx]={ox,oy};
                }
            }

            // Step 2: project candidates onto each segment; pick farthest-back
            XY jp;
            auto jit = node_xy.find(nid);
            if (jit != node_xy.end()) jp = jit->second;
            else jp = raw_lines[valid[0].first].coords.front();

            std::vector<XY> cut_pts(n);
            for (int idx = 0; idx < n; ++idx) {
                int si = valid[idx].first; bool iss = valid[idx].second;
                const XY& p1 = iss ? raw_lines[si].coords.front() : raw_lines[si].coords.back();
                const XY& p2 = iss ? raw_lines[si].coords[1]
                                   : raw_lines[si].coords[raw_lines[si].coords.size()-2];
                float sdx=p2.first-p1.first, sdy=p2.second-p1.second;
                float sL=std::sqrt(sdx*sdx+sdy*sdy);
                if (sL < 1e-9f) { cut_pts[idx]=p1; continue; }
                float snx=sdx/sL, sny=sdy/sL;

                std::vector<XY> cands = {jp};
                if (has_ix[idx])          cands.push_back(ixpts[idx]);
                if (has_ix[(idx-1+n)%n])  cands.push_back(ixpts[(idx-1+n)%n]);

                // Project each candidate onto the segment line, keep farthest from junction-201m
                XY ref = {jp.first-snx*201.f, jp.second-sny*201.f};
                XY best = cands[0];
                float best_d2 = std::numeric_limits<float>::lowest();
                for (const XY& cand : cands) {
                    float t = (cand.first-p1.first)*snx + (cand.second-p1.second)*sny;
                    XY pr = {p1.first+t*snx, p1.second+t*sny};
                    float d2 = (pr.first-ref.first)*(pr.first-ref.first)
                              +(pr.second-ref.second)*(pr.second-ref.second);
                    if (d2 > best_d2) { best_d2=d2; best=pr; }
                }
                cut_pts[idx] = best;
            }

            // Step 3: build cut vectors and junction polygon
            const float SNAP = 0.01f;
            std::vector<std::pair<XY,XY>> seg_ifaces(n);
            for (int idx = 0; idx < n; ++idx) {
                int si = valid[idx].first; bool iss = valid[idx].second;
                float hw = raw_lines[si].width * 0.5f;

                // Outward-from-junction tangent at the polyline end that meets the junction.
                XY d  = away_dir(raw_lines[si].coords, iss);
                XY rn = {d.second, -d.first};    // right-of-outward
                XY scv = {rn.first*hw, rn.second*hw};

                XY cp = cut_pts[idx];
                XY lc = {cp.first-scv.first, cp.second-scv.second};   // LEFT-of-outward
                XY rc = {cp.first+scv.first, cp.second+scv.second};   // RIGHT-of-outward
                seg_ifaces[idx] = {lc, rc};
                // seg_ifaces holds {LEFT-of-outward, RIGHT-of-outward} for the junction polygon.
                // cuts must hold {LEFT-of-forward, RIGHT-of-forward} for poly_coords; for end
                // cuts (iss=false) forward is opposite outward, so swap the pair.
                cuts[{si, iss}] = iss ? Cut{lc, rc} : Cut{rc, lc};
            }

            // Step 3b: Snap adjacent segment interfaces (merge if too close)
            // This prevents crossing edges at junctions
            for (int idx = 0; idx < n; ++idx) {
                auto& curr = seg_ifaces[idx];
                auto& next = seg_ifaces[(idx+1)%n];
                float dx = next.first.first - curr.second.first;
                float dy = next.first.second - curr.second.second;
                float d = std::sqrt(dx*dx + dy*dy);
                if (d < SNAP) {
                    // Merge: make next's left contact match curr's right contact
                    next.first = curr.second;
                    // Update cuts map to reflect the snapped interface.
                    // next is in outward-frame; cuts must be in forward-frame (swap on end cuts).
                    int next_si = valid[(idx+1)%n].first;
                    bool next_iss = valid[(idx+1)%n].second;
                    cuts[{next_si, next_iss}] = next_iss ? next : Cut{next.second, next.first};
                }
            }

            std::vector<XY> jvec;
            for (int idx = 0; idx < n; ++idx) {
                const XY& lc = seg_ifaces[idx].first;
                const XY& rc = seg_ifaces[idx].second;
                if (jvec.empty() ||
                    std::hypot(lc.first-jvec.back().first, lc.second-jvec.back().second) > SNAP)
                    jvec.push_back(lc);
                jvec.push_back(rc);
                if (has_ix[idx]) {
                    const XY& pb = ixpts[idx];
                    bool dupe = false;
                    for (const XY& v : jvec)
                        if (std::hypot(pb.first-v.first, pb.second-v.second) < SNAP)
                            { dupe=true; break; }
                    if (!dupe) jvec.push_back(pb);
                }
            }
            if ((int)jvec.size() >= 3) {
                if (std::hypot(jvec.back().first-jvec.front().first,
                               jvec.back().second-jvec.front().second) > SNAP)
                    jvec.push_back(jvec.front());
                Geometry2D gj; gj.coords = std::move(jvec);
                junction_polys.push_back(std::move(gj));
            }
        }  // end node loop

        // ── Build segment polygons ────────────────────────────────────────
        std::vector<Geometry2D> seg_polys;
        for (int i = 0; i < (int)raw_lines.size(); ++i) {
            float hw = raw_lines[i].width * 0.5f;
            const Cut* sc = nullptr;
            const Cut* ec = nullptr;
            auto it_s = cuts.find({i, true});
            auto it_e = cuts.find({i, false});
            if (it_s != cuts.end()) sc = &it_s->second;
            if (it_e != cuts.end()) ec = &it_e->second;
            std::vector<XY> pts = poly_coords(raw_lines[i].coords, hw, sc, ec);
            if (pts.size() >= 4) {
                Geometry2D gd; gd.coords = std::move(pts);
                seg_polys.push_back(std::move(gd));
            }
        }

        return std::make_pair(std::move(seg_polys), std::move(junction_polys));
    }

}
