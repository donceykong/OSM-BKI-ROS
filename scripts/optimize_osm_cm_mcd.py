#!/usr/bin/env python3
"""
Optimize the OSM confusion matrix for semantic-OSM prior fusion.

Loads GT-labeled lidar scans, transforms them to map frame, queries OSM
geometry for each voxel, then builds a co-occurrence matrix between GT
semantic classes and OSM categories.  The result is a new
osm_confusion_matrix YAML that maximizes agreement between OSM priors
and ground-truth labels.

Usage:
    python3 optimize_osm_cm.py [--config CONFIG_YAML] [--output OUTPUT_YAML]
                               [--data-dir DATA_DIR] [--max-scans N] [--keyframe-dist D]
                               [--grid-res G] [--visualize] [--visualize-points]
                               [--no-show] [--vis-output VIS_PNG] [--vis-points-output VIS_PNG]
                               [--height-step-meters S] [--height-max-meters M] [--no-height-matrix]

    --visualize           Plot the optimized matrix as a heatmap (blue=suppress, red=boost).
    --visualize-points    Plot GT points + OSM geometry + OSM prior heatmap (spatial view).
    --no-show             Save PNG only, do not block to show figure.
    --height-step-meters  Per-bin height step (m) for osm_height_confusion_matrix (default: 1.0).
    --height-max-meters   Max |z_local| extent (m); num_bins = 2*ceil(max/step) (default: 30.0).
    --no-height-matrix    Skip computing and writing osm_height_confusion_matrix.
    --vis-output          Path for matrix PNG (default: <output>.png).
    --vis-points-output   Path for points+OSM PNG (default: <output>_points_osm.png).
    --use-inferred-row    Use inferred (model) labels as matrix rows. Optimizes for OSM to correct inferred toward GT.

    --data-dir DIR        Root directory for dataset (lidar, poses, labels, OSM). If not set,
                          uses data_root from config; if that is empty, uses <script_dir>/data/<dataset_name>.

Defaults are read from config/methods/mcd.yaml relative to this script.
"""

import argparse
import csv
import math
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import yaml
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from shapely.geometry import Point, LineString, Polygon, box
    from shapely.strtree import STRtree
    from shapely import make_valid
    from shapely.errors import GEOSException
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    box = None
    make_valid = None
    GEOSException = Exception  # fallback if shapely.errors not available

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════════
# 1. Geometry helpers
# ═══════════════════════════════════════════════════════════════════

def _is_poly_with_holes(p):
    """True if p is (outer, holes) tuple. outer=list of (x,y), holes=list of rings."""
    return (isinstance(p, tuple) and len(p) == 2 and
            isinstance(p[0], list) and isinstance(p[1], list) and
            (not p[1] or isinstance(p[1][0], (list, tuple))))


def _poly_outer(p):
    """Extract outer ring from polygon. Supports simple list or (outer, holes) tuple."""
    if _is_poly_with_holes(p):
        return p[0]
    return p


def _poly_holes(p):
    """Extract hole rings from polygon. Returns [] for simple polygons."""
    if _is_poly_with_holes(p):
        return p[1]
    return []


def point_in_polygon(px, py, poly):
    """Ray-casting test. poly is list of (x,y) for a single ring."""
    n = len(poly)
    if n < 3:
        return False
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if ((yi > py) != (yj > py)) and (px < (xj - xi) * (py - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside


def point_in_polygon_with_holes(px, py, poly):
    """Inside filled region iff inside outer AND outside all holes. poly = (outer, holes) or simple list."""
    outer = _poly_outer(poly)
    holes = _poly_holes(poly)
    if not point_in_polygon(px, py, outer):
        return False
    for hole in holes:
        if point_in_polygon(px, py, hole):
            return False  # inside hole = outside filled
    return True


def segment_distance_sq(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        ddx, ddy = px - ax, py - ay
        return ddx * ddx + ddy * ddy
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len_sq))
    qx, qy = ax + t * dx, ay + t * dy
    ddx, ddy = px - qx, py - qy
    return ddx * ddx + ddy * ddy


def distance_to_polyline(px, py, coords):
    if len(coords) < 2:
        return float("inf")
    min_dsq = float("inf")
    for i in range(len(coords) - 1):
        dsq = segment_distance_sq(px, py, *coords[i], *coords[i + 1])
        if dsq < min_dsq:
            min_dsq = dsq
    return math.sqrt(min_dsq)


def _distance_to_ring_boundary(px, py, ring):
    """Signed distance to a single ring: negative inside, positive outside."""
    if len(ring) < 2:
        return float("inf")
    inside = point_in_polygon(px, py, ring)
    n = len(ring)
    min_dsq = float("inf")
    j = n - 1
    for i in range(n):
        dsq = segment_distance_sq(px, py, *ring[j], *ring[i])
        if dsq < min_dsq:
            min_dsq = dsq
        j = i
    d = math.sqrt(min_dsq)
    return -d if inside else d


def distance_to_polygon_boundary(px, py, poly):
    """Signed distance: negative = inside filled region, positive = outside. Supports polygons with holes."""
    outer = _poly_outer(poly)
    holes = _poly_holes(poly)
    if len(outer) < 2:
        return float("inf")
    if not holes:
        return _distance_to_ring_boundary(px, py, outer)
    # Polygon with holes: inside_filled = inside_outer AND not inside any hole
    inside_outer = point_in_polygon(px, py, outer)
    inside_any_hole = any(point_in_polygon(px, py, h) for h in holes)
    inside_filled = inside_outer and not inside_any_hole
    min_d = _distance_to_ring_boundary(px, py, outer)
    for h in holes:
        d = _distance_to_ring_boundary(px, py, h)
        if abs(d) < abs(min_d):
            min_d = d
    # Use sign: inside filled -> negative (closest boundary is "inside" the filled region)
    d_abs = abs(min_d)
    return -d_abs if inside_filled else d_abs


def osm_prior_from_signed_distance(signed_d, decay_m):
    if decay_m <= 0:
        return 1.0 if signed_d <= 0 else 0.0
    if signed_d <= 0:
        return 1.0
    return max(0.0, 1.0 - signed_d / decay_m)


def osm_prior_from_distance(d, decay_m):
    if decay_m <= 0:
        return 0.0
    return max(0.0, 1.0 - d / decay_m)


# ═══════════════════════════════════════════════════════════════════
# 2. OSM XML parser
# ═══════════════════════════════════════════════════════════════════

ROAD_HIGHWAY_TYPES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "motorway_link", "trunk_link",
    "primary_link", "secondary_link", "tertiary_link", "living_street",
    "service", "road",
}
# Per-highway-type default widths (OSM2World RoadModule defaults, metres).
# road_width_meters in GEOM_PARAMS is used as the "else" fallback.
_HIGHWAY_WIDTH_M = {
    "motorway":       8.75,
    "motorway_link":  8.75,
    "trunk":          7.0,
    "trunk_link":     7.0,
    "primary":        7.0,
    "primary_link":   7.0,
    "secondary":      7.0,
    "secondary_link": 7.0,
    "service":        3.5,
    "track":          2.5,
}


def _road_width_for_highway(hw_type):
    """Return half-width lookup width (m) for a given OSM highway tag value."""
    return _HIGHWAY_WIDTH_M.get(hw_type, GEOM_PARAMS.get("road_width_meters", 4.0))


SIDEWALK_HIGHWAY_TYPES = {"footway", "path", "pedestrian", "foot"}
CYCLEWAY_HIGHWAY = {"cycleway"}
GRASSLAND_LANDUSE = {"grass", "meadow", "greenfield", "recreation_ground"}
GRASSLAND_NATURAL = {"grassland", "heath", "scrub"}
FOREST_NATURAL = {"wood", "forest"}
TREE_LANDUSE = {"orchard", "vineyard"}


def _latlon_to_xy(lat, lon, origin_lat, origin_lon):
    scale = math.cos(math.radians(origin_lat))
    x = (lon - origin_lon) * 111319.0 * scale
    y = (lat - origin_lat) * 111319.0
    return (x, y)


def _right_normal_2d(p0, p1):
    """Unit right normal of segment p0→p1 (rightward when walking p0→p1)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    L = math.sqrt(dx * dx + dy * dy)
    if L < 1e-9:
        return (1.0, 0.0)
    return (dy / L, -dx / L)


def _line_intersection_2d(px, py, dx, dy, qx, qy, ex, ey):
    """Intersection of lines (p+t*d) and (q+s*e). Returns (x,y) or None if parallel."""
    cross = dx * ey - dy * ex
    if abs(cross) < 1e-9:
        return None
    t = ((qx - px) * ey - (qy - py) * ex) / cross
    return (px + t * dx, py + t * dy)


def _polyline_to_polygon_coords(coords, half_w, start_left=None, start_right=None,
                                end_left=None, end_right=None):
    """Build explicit band polygon for a polyline.

    Interior vertices use miter (angle-bisector) joins clamped at 5× half_w.
    Endpoints use provided cuts, or fall back to orthogonal.
    Returns a closed list of (x,y) or None if degenerate.
    """
    if len(coords) < 2:
        return None
    left_out, right_out = [], []

    if start_left is not None:
        left_out.append(start_left)
        right_out.append(start_right)
    else:
        n = _right_normal_2d(coords[0], coords[1])
        c = coords[0]
        left_out.append((c[0] - n[0]*half_w, c[1] - n[1]*half_w))
        right_out.append((c[0] + n[0]*half_w, c[1] + n[1]*half_w))

    for i in range(1, len(coords) - 1):
        n0 = _right_normal_2d(coords[i-1], coords[i])
        n1 = _right_normal_2d(coords[i], coords[i+1])
        mx, my = n0[0]+n1[0], n0[1]+n1[1]
        mlen = math.sqrt(mx*mx + my*my)
        if mlen < 1e-9:
            mx, my = n0
        else:
            mx, my = mx/mlen, my/mlen
        dot = n0[0]*mx + n0[1]*my
        scale = min(5.0, 1.0/dot) if dot > 1e-6 else 5.0
        c = coords[i]
        left_out.append((c[0] - mx*half_w*scale, c[1] - my*half_w*scale))
        right_out.append((c[0] + mx*half_w*scale, c[1] + my*half_w*scale))

    if end_left is not None:
        left_out.append(end_left)
        right_out.append(end_right)
    else:
        n = _right_normal_2d(coords[-2], coords[-1])
        c = coords[-1]
        left_out.append((c[0] - n[0]*half_w, c[1] - n[1]*half_w))
        right_out.append((c[0] + n[0]*half_w, c[1] + n[1]*half_w))

    poly = right_out + list(reversed(left_out))
    poly.append(poly[0])
    return poly


def _circle_polygon_coords(cx, cy, radius, n_seg=24):
    """Approximate a circle as a closed n-gon polygon."""
    pts = [(cx + radius * math.cos(2*math.pi*i/n_seg),
            cy + radius * math.sin(2*math.pi*i/n_seg))
           for i in range(n_seg)]
    pts.append(pts[0])
    return pts


_PARALLEL_THRESHOLD_RAD = math.pi / 18  # ~10°, same as OSM2World


def _build_network_polygons(raw_lines, node_xy, width_fn):
    """Convert raw polylines to explicit band polygons with OSM2World-style junction cuts.

    Args:
        raw_lines:  list of (coords, key, start_nid, end_nid)
        node_xy:    dict of node_id -> (x, y)
        width_fn:   callable(key) -> float  (full road width in metres)

    Returns:
        (segment_polys, junction_polys), each a list of (outer_coords, []) tuples.
    """
    node_segs = defaultdict(list)
    for i, (coords, key, snid, enid) in enumerate(raw_lines):
        if len(coords) >= 2:
            node_segs[snid].append((i, True))
            node_segs[enid].append((i, False))

    cuts = {}           # (seg_idx, 'start'|'end') -> (left_pt, right_pt)
    junction_polys = []

    for node_id, conns in node_segs.items():
        valid = [(si, istart) for si, istart in conns if len(raw_lines[si][0]) >= 2]
        n = len(valid)
        if n == 0:
            continue
        elif n == 1:
            _net_cut_orthogonal(raw_lines, width_fn, valid[0][0], valid[0][1], cuts)
        elif n == 2:
            _net_cut_connector(raw_lines, width_fn, valid[0], valid[1], cuts)
        else:
            jp = _net_cut_junction(raw_lines, width_fn, valid,
                                   node_xy.get(node_id), cuts)
            if jp is not None:
                junction_polys.append((jp, []))

    seg_polys = []
    for i, (coords, key, snid, enid) in enumerate(raw_lines):
        half_w = width_fn(key) * 0.5
        sl, sr = cuts.get((i, 'start'), (None, None))
        el, er = cuts.get((i, 'end'), (None, None))
        pts = _polyline_to_polygon_coords(coords, half_w,
                                          start_left=sl, start_right=sr,
                                          end_left=el, end_right=er)
        if pts and len(pts) >= 4:
            seg_polys.append((pts, []))

    return seg_polys, junction_polys


def _net_cut_orthogonal(raw_lines, width_fn, si, is_start, cuts):
    coords, key = raw_lines[si][0], raw_lines[si][1]
    half_w = width_fn(key) * 0.5
    n = _right_normal_2d(coords[0], coords[1]) if is_start \
        else _right_normal_2d(coords[-2], coords[-1])
    c = coords[0] if is_start else coords[-1]
    cuts[(si, 'start' if is_start else 'end')] = (
        (c[0] - n[0]*half_w, c[1] - n[1]*half_w),
        (c[0] + n[0]*half_w, c[1] + n[1]*half_w),
    )


def _net_cut_connector(raw_lines, width_fn, conn1, conn2, cuts):
    """Angle-bisector cut for two connecting segments (OSM2World connector logic)."""
    si1, is_start1 = conn1
    si2, is_start2 = conn2
    coords1, key1 = raw_lines[si1][0], raw_lines[si1][1]
    coords2, key2 = raw_lines[si2][0], raw_lines[si2][1]
    hw1 = width_fn(key1) * 0.5
    hw2 = width_fn(key2) * 0.5

    def toward_junction(coords, is_start):
        """Direction vector pointing TOWARD the junction from the segment."""
        if not is_start:
            dx, dy = coords[-1][0]-coords[-2][0], coords[-1][1]-coords[-2][1]
        else:
            dx, dy = -(coords[1][0]-coords[0][0]), -(coords[1][1]-coords[0][1])
        L = math.sqrt(dx*dx + dy*dy)
        return (dx/L, dy/L) if L > 1e-9 else (1.0, 0.0)

    def away_from_junction(coords, is_start):
        """Direction vector pointing AWAY from the junction."""
        if is_start:
            dx, dy = coords[1][0]-coords[0][0], coords[1][1]-coords[0][1]
        else:
            dx, dy = coords[-2][0]-coords[-1][0], coords[-2][1]-coords[-1][1]
        L = math.sqrt(dx*dx + dy*dy)
        return (dx/L, dy/L) if L > 1e-9 else (1.0, 0.0)

    in_vec = toward_junction(coords1, is_start1)
    out_vec = away_from_junction(coords2, is_start2)

    cvx, cvy = out_vec[0]-in_vec[0], out_vec[1]-in_vec[1]
    clen = math.sqrt(cvx*cvx + cvy*cvy)
    if clen < 1e-6:
        cvx, cvy = -in_vec[1], in_vec[0]
    else:
        cvx, cvy = cvx/clen, cvy/clen

    # Ensure cut vector points to the right (2D cross product check)
    if in_vec[0]*cvy - in_vec[1]*cvx <= 0:
        cvx, cvy = -cvx, -cvy

    c1 = coords1[0] if is_start1 else coords1[-1]
    c2 = coords2[0] if is_start2 else coords2[-1]

    # Store as (LEFT-of-forward, RIGHT-of-forward) using the segment's forward tangent
    # AT THE JUNCTION END so multi-node (curved) polylines get the correct local frame.
    def _store_cut_2seg(si, is_start, p, hw, coords):
        if is_start:
            fx, fy = coords[1][0]-coords[0][0], coords[1][1]-coords[0][1]
        else:
            fx, fy = coords[-1][0]-coords[-2][0], coords[-1][1]-coords[-2][1]
        L = math.sqrt(fx*fx + fy*fy)
        if L > 1e-9:
            fx, fy = fx/L, fy/L
        rn = (fy, -fx)                        # right-of-forward
        dot = cvx*rn[0] + cvy*rn[1]
        a = (p[0] - cvx*hw, p[1] - cvy*hw)
        b = (p[0] + cvx*hw, p[1] + cvy*hw)
        cuts[(si, 'start' if is_start else 'end')] = (a, b) if dot >= 0 else (b, a)

    _store_cut_2seg(si1, is_start1, c1, hw1, coords1)
    _store_cut_2seg(si2, is_start2, c2, hw2, coords2)


def _net_cut_junction(raw_lines, width_fn, valid, junction_pos, cuts):
    """≥3-segment junction cut (OSM2World NetworkCalculator edge-intersection approach).

    Returns junction polygon coords list, or None if degenerate.
    """
    n = len(valid)
    if junction_pos is None:
        si, is_start = valid[0]
        junction_pos = raw_lines[si][0][0] if is_start else raw_lines[si][0][-1]

    def away_dir(coords, is_start):
        if is_start:
            dx, dy = coords[1][0]-coords[0][0], coords[1][1]-coords[0][1]
        else:
            dx, dy = coords[-2][0]-coords[-1][0], coords[-2][1]-coords[-1][1]
        L = math.sqrt(dx*dx+dy*dy)
        return (dx/L, dy/L) if L > 1e-9 else (1.0, 0.0)

    # Sort segments by angle of their away-direction around the junction
    valid_sorted = sorted(valid,
                          key=lambda c: math.atan2(*away_dir(raw_lines[c[0]][0], c[1])[::-1]))

    # Step 1: intersection of left edge of segment[i] with right edge of segment[i+1]
    intersections = []
    for idx in range(n):
        si, is_start = valid_sorted[idx]
        ti, tstart  = valid_sorted[(idx+1) % n]
        sc = raw_lines[si][0][0] if is_start  else raw_lines[si][0][-1]
        tc = raw_lines[ti][0][0] if tstart    else raw_lines[ti][0][-1]
        s_hw = width_fn(raw_lines[si][1]) * 0.5
        t_hw = width_fn(raw_lines[ti][1]) * 0.5
        s_dir = away_dir(raw_lines[si][0], is_start)
        t_dir = away_dir(raw_lines[ti][0], tstart)
        s_rn = (s_dir[1], -s_dir[0])   # right normal of away direction
        t_rn = (t_dir[1], -t_dir[0])
        # Left edge of s (toward junction direction): sc - s_rn * s_hw
        s_ep = (sc[0] - s_rn[0]*s_hw, sc[1] - s_rn[1]*s_hw)
        # Right edge of t: tc + t_rn * t_hw
        t_ep = (tc[0] + t_rn[0]*t_hw, tc[1] + t_rn[1]*t_hw)
        # Edge directions pointing toward junction
        sed = (-s_dir[0], -s_dir[1])
        ted = (-t_dir[0], -t_dir[1])
        dot = sed[0]*(-ted[0]) + sed[1]*(-ted[1])
        if math.acos(max(-1.0, min(1.0, dot))) < _PARALLEL_THRESHOLD_RAD:
            intersections.append(None)
        else:
            intersections.append(_line_intersection_2d(
                s_ep[0], s_ep[1], sed[0], sed[1],
                t_ep[0], t_ep[1], ted[0], ted[1]))

    # Step 2: project candidates onto each segment line; pick farthest-back as cut point
    cut_points = []
    for idx in range(n):
        si, is_start = valid_sorted[idx]
        coords = raw_lines[si][0]
        cands = [junction_pos]
        if intersections[idx] is not None:
            cands.append(intersections[idx])
        if intersections[(idx-1+n) % n] is not None:
            cands.append(intersections[(idx-1+n) % n])

        p1 = coords[0] if is_start else coords[-1]
        p2 = coords[1] if is_start else coords[-2]
        sdx, sdy = p2[0]-p1[0], p2[1]-p1[1]
        sL = math.sqrt(sdx*sdx+sdy*sdy)
        if sL < 1e-9:
            cut_points.append(p1); continue
        sdn = (sdx/sL, sdy/sL)
        projected = [(p1[0]+(c[0]-p1[0])*sdn[0]+(c[1]-p1[1])*sdn[1]*0,
                      p1[1]+(c[0]-p1[0])*sdn[1]) for c in cands]
        # Correct projection: p1 + dot(cand-p1, sdn) * sdn
        projected = []
        for cand in cands:
            t = (cand[0]-p1[0])*sdn[0] + (cand[1]-p1[1])*sdn[1]
            projected.append((p1[0]+t*sdn[0], p1[1]+t*sdn[1]))
        ref = (junction_pos[0] - sdn[0]*201, junction_pos[1] - sdn[1]*201)
        cut_points.append(max(projected,
                              key=lambda p: (p[0]-ref[0])**2 + (p[1]-ref[1])**2))

    # Step 3: build cut vectors and junction polygon
    _SNAP = 0.01
    seg_interfaces = []
    for idx in range(n):
        si, is_start = valid_sorted[idx]
        coords = raw_lines[si][0]
        hw = width_fn(raw_lines[si][1]) * 0.5
        # Outward-from-junction tangent at the polyline end that meets the junction.
        d = away_dir(coords, is_start)
        rn = (d[1], -d[0])                    # right-of-outward
        scv = (rn[0]*hw, rn[1]*hw)
        cp = cut_points[idx]
        lc = (cp[0]-scv[0], cp[1]-scv[1])     # LEFT-of-outward
        rc = (cp[0]+scv[0], cp[1]+scv[1])     # RIGHT-of-outward
        seg_interfaces.append((lc, rc))
        # seg_interfaces holds {LEFT-of-outward, RIGHT-of-outward} for the junction polygon.
        # cuts must hold {LEFT-of-forward, RIGHT-of-forward} for the band builder; for end
        # cuts (is_start=False) forward is opposite outward, so swap the pair.
        cuts[(si, 'start' if is_start else 'end')] = (lc, rc) if is_start else (rc, lc)

    vectors = []
    for idx in range(n):
        lc, rc = seg_interfaces[idx]
        if not vectors or math.hypot(lc[0]-vectors[-1][0], lc[1]-vectors[-1][1]) > _SNAP:
            vectors.append(lc)
        vectors.append(rc)
        pb = intersections[idx]
        if pb is not None and not any(
                math.hypot(pb[0]-v[0], pb[1]-v[1]) < _SNAP for v in vectors):
            vectors.append(pb)

    if len(vectors) < 3:
        return None
    if math.hypot(vectors[-1][0]-vectors[0][0], vectors[-1][1]-vectors[0][1]) > _SNAP:
        vectors.append(vectors[0])
    return vectors


def _way_to_coords(way_el, nodes, origin_lat, origin_lon):
    """Convert a way element to list of (x,y) coords. Returns None if invalid."""
    nd_refs = [nd.attrib["ref"] for nd in way_el.iter("nd")]
    coords = [_latlon_to_xy(*nodes[r], origin_lat, origin_lon) for r in nd_refs if r in nodes]
    return coords if len(coords) >= 2 else None


def _ring_area(coords):
    """Shoelace formula for signed area. Use abs for comparison."""
    if len(coords) < 3:
        return 0
    n = len(coords)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += coords[i][0] * coords[j][1] - coords[j][0] * coords[i][1]
    return abs(area) * 0.5


def _parse_multipolygon_relation(rel, ways_coords, nodes, origin_lat, origin_lon):
    """Parse a multipolygon relation into list of (outer, holes) polygons.
    Returns [] if invalid. Each inner ring is assigned to the smallest outer that contains it.
    """
    outers = []  # list of (way_id, coords)
    inners = []
    for mem in rel.iter("member"):
        role = mem.attrib.get("role", "")
        ref = mem.attrib.get("ref")
        mem_type = mem.attrib.get("type", "")
        if mem_type != "way" or not ref:
            continue
        ref = str(ref)
        coords = ways_coords.get(ref)
        if coords is None or len(coords) < 3:
            continue
        if role == "outer":
            outers.append((ref, coords))
        elif role == "inner":
            inners.append((ref, coords))
    if not outers:
        return []
    # Assign each inner to the smallest outer that contains it (for nested multipolygons)
    outer_to_holes = {i: [] for i in range(len(outers))}
    for _, inner_coords in inners:
        if len(inner_coords) < 3:
            continue
        cx = sum(p[0] for p in inner_coords) / len(inner_coords)
        cy = sum(p[1] for p in inner_coords) / len(inner_coords)
        best_outer = None
        best_area = float("inf")
        for i, (_, outer_coords) in enumerate(outers):
            if point_in_polygon(cx, cy, outer_coords):
                area = _ring_area(outer_coords)
                if area < best_area:
                    best_area = area
                    best_outer = i
        if best_outer is not None:
            outer_to_holes[best_outer].append(inner_coords)
    result = []
    for i, (_, outer_coords) in enumerate(outers):
        holes = outer_to_holes.get(i, [])
        if len(outer_coords) >= 3:
            result.append((outer_coords, holes) if holes else (outer_coords, []))
    return result


def parse_osm_xml(osm_path, origin_lat, origin_lon):
    tree = ET.parse(osm_path)
    root = tree.getroot()
    nodes = {}
    for nd in root.iter("node"):
        nodes[nd.attrib["id"]] = (float(nd.attrib["lat"]), float(nd.attrib["lon"]))

    # Projected (x,y) for each node — needed by network polygon builder
    node_xy = {nid: _latlon_to_xy(lat, lon, origin_lat, origin_lon)
               for nid, (lat, lon) in nodes.items()}

    # Build way_id -> coords and way_id -> node_refs for relation resolution + network building
    ways_coords = {}
    ways_node_refs = {}
    for way in root.iter("way"):
        nd_refs = [nd.attrib["ref"] for nd in way.iter("nd")]
        valid_refs = [r for r in nd_refs if r in nodes]
        coords = [_latlon_to_xy(*nodes[r], origin_lat, origin_lon) for r in valid_refs]
        if len(coords) >= 2:
            ways_coords[way.attrib["id"]] = coords
            ways_node_refs[way.attrib["id"]] = valid_refs

    buildings, grasslands, trees_poly, forests = [], [], [], []
    parking, tree_points = [], []
    # Network line collections: (coords, width_or_key, start_nid, end_nid)
    raw_roads = []
    raw_sidewalks = []
    raw_cycleways = []
    raw_fences = []

    for nd in root.iter("node"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in nd.iter("tag")}
        if tags.get("natural") == "tree":
            tree_points.append(_latlon_to_xy(float(nd.attrib["lat"]), float(nd.attrib["lon"]), origin_lat, origin_lon))

    for way in root.iter("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.iter("tag")}
        way_id = way.attrib["id"]
        coords = ways_coords.get(way_id)
        if coords is None:
            continue
        nd_refs = ways_node_refs.get(way_id, [])
        start_nid = nd_refs[0] if nd_refs else None
        end_nid = nd_refs[-1] if nd_refs else None

        if "building" in tags:
            if len(coords) >= 3:
                buildings.append((coords, []))
            continue
        amenity = tags.get("amenity", "")
        if amenity in ("parking", "parking_space") and len(coords) >= 3:
            parking.append((coords, []))
            continue
        # Resolve width: OSM width=* tag (if present & parseable) overrides the default.
        def _resolve_width(fallback):
            wt = tags.get("width")
            if wt:
                try:
                    return float(wt)
                except (ValueError, TypeError):
                    pass
            return fallback

        if tags.get("barrier") == "fence":
            raw_fences.append((coords, _resolve_width(GEOM_PARAMS["fence_width_meters"]),
                               start_nid, end_nid))
            continue
        hw = tags.get("highway", "")
        if hw in SIDEWALK_HIGHWAY_TYPES:
            raw_sidewalks.append((coords, _resolve_width(GEOM_PARAMS["sidewalk_width_meters"]),
                                  start_nid, end_nid))
            continue
        if hw in CYCLEWAY_HIGHWAY:
            raw_cycleways.append((coords, _resolve_width(GEOM_PARAMS["cycleway_width_meters"]),
                                  start_nid, end_nid))
            continue
        if hw in ROAD_HIGHWAY_TYPES:
            raw_roads.append((coords, _resolve_width(_road_width_for_highway(hw)),
                              start_nid, end_nid))
            continue
        lu = tags.get("landuse", "")
        if lu in GRASSLAND_LANDUSE and len(coords) >= 3:
            grasslands.append((coords, []))
            continue
        nat = tags.get("natural", "")
        if nat in GRASSLAND_NATURAL and len(coords) >= 3:
            grasslands.append((coords, []))
            continue
        if nat in FOREST_NATURAL and len(coords) >= 3:
            forests.append((coords, []))
            continue
        if lu == "forest" and len(coords) >= 3:
            forests.append((coords, []))
            continue
        if lu in TREE_LANDUSE and len(coords) >= 3:
            trees_poly.append((coords, []))
            continue
        if tags.get("landcover") == "trees" and len(coords) >= 3:
            trees_poly.append((coords, []))
            continue

    # Parse multipolygon relations
    for rel in root.iter("relation"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in rel.iter("tag")}
        if tags.get("type") != "multipolygon":
            continue
        polys = _parse_multipolygon_relation(rel, ways_coords, nodes, origin_lat, origin_lon)
        if not polys:
            continue
        if "building" in tags:
            buildings.extend(polys)
        elif tags.get("amenity") in ("parking", "parking_space"):
            parking.extend(polys)
        elif tags.get("landuse") in GRASSLAND_LANDUSE:
            grasslands.extend(polys)
        elif tags.get("landuse") in ("park",):
            grasslands.extend(polys)
        elif tags.get("leisure") in ("park", "garden"):
            grasslands.extend(polys)
        elif tags.get("landuse") in TREE_LANDUSE:
            trees_poly.extend(polys)
        elif tags.get("landuse") == "forest":
            forests.extend(polys)
        elif tags.get("natural") in FOREST_NATURAL:
            forests.extend(polys)
        elif tags.get("landcover") == "trees":
            trees_poly.extend(polys)

    # Build explicit network polygons with OSM2World-style junction logic.
    # Each raw_* tuple now stores an already-resolved per-way width (OSM width=* tag
    # when present, otherwise category default), so width_fn is the identity.
    road_seg, road_junc = _build_network_polygons(raw_roads, node_xy, lambda w: w)
    roads = road_seg + road_junc

    sw_seg, sw_junc = _build_network_polygons(raw_sidewalks, node_xy, lambda w: w)
    sidewalks = sw_seg + sw_junc

    cy_seg, cy_junc = _build_network_polygons(raw_cycleways, node_xy, lambda w: w)
    cycleways = cy_seg + cy_junc

    fn_seg, fn_junc = _build_network_polygons(raw_fences, node_xy, lambda w: w)
    fences = fn_seg + fn_junc

    return dict(buildings=buildings, roads=roads, grasslands=grasslands,
                trees=trees_poly, forests=forests, parking=parking,
                fences=fences, tree_points=tree_points,
                sidewalks=sidewalks, cycleways=cycleways)


# ═══════════════════════════════════════════════════════════════════
# 3. OSM geometry trimming (keep only geometry near scan trajectory)
# ═══════════════════════════════════════════════════════════════════

def _bbox_intersects_expanded(poly_or_line, xmin, xmax, ymin, ymax, margin):
    """True if polygon/line bbox intersects expanded region. Uses outer ring for polygons with holes."""
    if not poly_or_line:
        return False
    ring = _poly_outer(poly_or_line)  # works for both (outer,holes) and simple list
    xs = [p[0] for p in ring]
    ys = [p[1] for p in ring]
    pmin_x, pmax_x = min(xs), max(xs)
    pmin_y, pmax_y = min(ys), max(ys)
    return not (pmax_x < xmin - margin or pmin_x > xmax + margin or
                pmax_y < ymin - margin or pmin_y > ymax + margin)


def trim_osm_to_bbox(geom, xmin, xmax, ymin, ymax, margin):
    """Return a copy of geom with only elements whose bbox intersects the expanded region.

    margin: meters to expand the bbox (e.g. decay_m + max_range) so we keep
    geometry that can influence points near the trajectory.
    """
    out = {}
    out["roads"] = [g for g in geom.get("roads", [])
                    if _bbox_intersects_expanded(g, xmin, xmax, ymin, ymax, margin)]
    for cat in ("buildings", "grasslands", "trees", "forests",
                "parking", "fences", "sidewalks", "cycleways"):
        out[cat] = [g for g in geom.get(cat, []) if _bbox_intersects_expanded(g, xmin, xmax, ymin, ymax, margin)]
    for pt_key in ("tree_points",):
        pts = geom.get(pt_key, [])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        out[pt_key] = [
            pts[i] for i in range(len(pts))
            if (xmin - margin <= xs[i] <= xmax + margin and ymin - margin <= ys[i] <= ymax + margin)
        ]
    return out


# ═══════════════════════════════════════════════════════════════════
# 4. Grid-based OSM prior computation (fast)
# ═══════════════════════════════════════════════════════════════════

OSM_COLUMNS = [
    "roads", "sidewalks", "cycleways", "parking", "grasslands", "trees", "forest",
    "buildings", "fences",
]
N_OSM = len(OSM_COLUMNS)

_GEOM_FALLBACK = {
    "road_width_meters": 4.0,
    "sidewalk_width_meters": 2.0,
    "cycleway_width_meters": 2.0,
    "fence_width_meters": 0.6,
    "tree_point_radius_meters": 5.0,
}
GEOM_PARAMS = dict(_GEOM_FALLBACK)


def _load_geom_defaults(osm_bki_path):
    """Load default geometry parameters from osm_bki.yaml, falling back to built-in values."""
    if not os.path.isfile(osm_bki_path):
        print(f"[WARN] osm_bki.yaml not found at {osm_bki_path}, using built-in defaults")
        return dict(_GEOM_FALLBACK)
    with open(osm_bki_path) as f:
        raw = yaml.safe_load(f)
    if "/**" in raw:
        raw = raw["/**"].get("ros__parameters", raw)
    elif "ros__parameters" in raw:
        raw = raw["ros__parameters"]
    params = raw.get("osm_geometry_parameters", {})
    result = dict(_GEOM_FALLBACK)
    result.update({k: v for k, v in params.items() if v is not None})
    return result

# Search radius for STRtree: geometry beyond this does not affect prior (decay_m typically 3m)
_QUERY_BUFFER = 50.0  # meters


def _build_shapely_index(geom):
    """Build Shapely geometries + STRtrees for fast spatial queries. Returns dict or None if Shapely unavailable."""
    if not SHAPELY_AVAILABLE:
        return None
    idx = {"geom": geom, "tree_points": geom.get("tree_points", [])}
    # Roads: now stored as polygons (already expanded), treat as regular polygons
    road_geoms = []
    for poly in geom.get("roads", []):
        outer = _poly_outer(poly)
        if len(outer) >= 3:
            try:
                shell = list(outer) + [outer[0]] if outer[0] != outer[-1] else list(outer)
                g = Polygon(shell)
                if not g.is_empty:
                    if not g.is_valid and make_valid is not None:
                        g = make_valid(g)
                        if g is not None and not g.is_empty:
                            if hasattr(g, "geoms"):
                                for part in g.geoms:
                                    if part.geom_type == "Polygon" and not part.is_empty:
                                        road_geoms.append(part)
                            elif g.geom_type == "Polygon":
                                road_geoms.append(g)
                    else:
                        road_geoms.append(g)
            except (GEOSException, ValueError, TypeError):
                pass
    idx["roads"] = (STRtree(road_geoms), road_geoms) if road_geoms else (None, [])
    for cat in ("sidewalks", "cycleways", "fences", "buildings", "grasslands", "trees", "forests", "parking"):
        geoms = []
        for poly in geom.get(cat, []):
            outer = _poly_outer(poly)
            holes = _poly_holes(poly)
            if len(outer) < 3:
                continue
            try:
                shell = list(outer) + [outer[0]] if outer[0] != outer[-1] else list(outer)
                hole_rings = []
                for h in holes:
                    if len(h) >= 3:
                        hr = list(h) + [h[0]] if h[0] != h[-1] else list(h)
                        hole_rings.append(hr)
                g = Polygon(shell, hole_rings if hole_rings else None)
                if g.is_empty:
                    continue
                if not g.is_valid and make_valid is not None:
                    g = make_valid(g)
                    if g is None or g.is_empty:
                        continue
                    # make_valid can return MultiPolygon; use first polygon part
                    if hasattr(g, "geoms"):
                        for part in g.geoms:
                            if part.geom_type == "Polygon" and not part.is_empty:
                                geoms.append(part)
                        continue
                    if g.geom_type != "Polygon":
                        continue
                geoms.append(g)
            except (GEOSException, ValueError, TypeError):
                pass
        idx[cat] = (STRtree(geoms), geoms) if geoms else (None, [])
    return idx


def _shapely_signed_distance(g, pt):
    """Signed distance from pt to polygon: negative inside, positive outside. Returns (sd, ok).
    On GEOS TopologyException (invalid geometry), returns (float('inf'), False)."""
    try:
        boundary = g.boundary
        if boundary is None:
            return float("inf"), False
        d = boundary.distance(pt)
        inside = g.contains(pt)
        return (-d if inside else d, True)
    except GEOSException:
        return float("inf"), False


def _query_candidates(idx, cat, x, y):
    """Return list of Shapely geometries that may be near (x,y)."""
    tree, geoms = idx.get(cat, (None, []))
    if not tree or not geoms:
        return []
    q = box(x - _QUERY_BUFFER, y - _QUERY_BUFFER, x + _QUERY_BUFFER, y + _QUERY_BUFFER)
    inds = tree.query(q)
    try:
        inds = np.atleast_1d(inds).tolist()
    except Exception:
        inds = [inds] if isinstance(inds, int) else []
    return [geoms[i] for i in inds if 0 <= i < len(geoms)]


def _compute_single_prior_shapely(x, y, idx, cat, decay_m, tree_radius):
    """Fast prior using Shapely + STRtree."""
    pt = Point(x, y)

    def _prior_dist(d):
        return osm_prior_from_distance(d, decay_m)
    def _prior_signed(sd):
        return osm_prior_from_signed_distance(sd, decay_m)

    if cat == "roads":
        min_d = float("inf")
        for g in _query_candidates(idx, "roads", x, y):
            sd, ok = _shapely_signed_distance(g, pt)
            if not ok:
                continue
            if sd <= 0:
                return 1.0
            if sd < min_d:
                min_d = sd
        return _prior_signed(min_d) if min_d != float("inf") else 0.0

    elif cat == "sidewalks":
        min_d = float("inf")
        for g in _query_candidates(idx, "sidewalks", x, y):
            sd, ok = _shapely_signed_distance(g, pt)
            if not ok: continue
            if sd <= 0: return 1.0
            if sd < min_d: min_d = sd
        return _prior_signed(min_d) if min_d != float("inf") else 0.0

    elif cat == "cycleways":
        min_d = float("inf")
        for g in _query_candidates(idx, "cycleways", x, y):
            sd, ok = _shapely_signed_distance(g, pt)
            if not ok: continue
            if sd <= 0: return 1.0
            if sd < min_d: min_d = sd
        return _prior_signed(min_d) if min_d != float("inf") else 0.0

    elif cat == "parking":
        max_p = 0.0
        for g in _query_candidates(idx, "parking", x, y):
            sd, ok = _shapely_signed_distance(g, pt)
            if ok:
                if sd <= 0:
                    return 1.0
                p = _prior_signed(sd)
            else:
                try:
                    p = _prior_dist(g.distance(pt))
                except GEOSException:
                    continue
            if p > max_p:
                max_p = p
        return max_p

    elif cat == "grasslands":
        min_d = float("inf")
        for g in _query_candidates(idx, "grasslands", x, y):
            sd, ok = _shapely_signed_distance(g, pt)
            if not ok:
                continue
            if sd <= 0:
                return 1.0
            if sd < min_d:
                min_d = sd
        return _prior_signed(min_d) if min_d != float("inf") else 0.0

    elif cat == "trees":
        max_p = 0.0
        for g in _query_candidates(idx, "trees", x, y):
            sd, ok = _shapely_signed_distance(g, pt)
            if not ok:
                continue
            if sd <= 0:
                return 1.0
            p = _prior_signed(sd)
            if p > max_p:
                max_p = p
        for cx, cy in idx["tree_points"]:
            d_center = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            sd = d_center - tree_radius
            p = _prior_signed(sd)
            if p > max_p:
                max_p = p
        return max_p

    elif cat == "forest":
        min_d = float("inf")
        for g in _query_candidates(idx, "forests", x, y):
            sd, ok = _shapely_signed_distance(g, pt)
            if not ok:
                continue
            if sd <= 0:
                return 1.0
            if sd < min_d:
                min_d = sd
        return _prior_signed(min_d) if min_d != float("inf") else 0.0

    elif cat == "buildings":
        min_d = float("inf")
        for g in _query_candidates(idx, "buildings", x, y):
            sd, ok = _shapely_signed_distance(g, pt)
            if not ok:
                continue
            if sd <= 0:
                return 1.0
            if sd < min_d:
                min_d = sd
        return _prior_signed(min_d) if min_d != float("inf") else 0.0

    elif cat == "fences":
        min_d = float("inf")
        for g in _query_candidates(idx, "fences", x, y):
            sd, ok = _shapely_signed_distance(g, pt)
            if not ok: continue
            if sd <= 0: return 1.0
            if sd < min_d: min_d = sd
        return _prior_signed(min_d) if min_d != float("inf") else 0.0

    return 0.0


def _compute_single_prior(x, y, geom, cat, decay_m, tree_radius):
    """Compute OSM prior for one category at (x,y)."""
    def _poly_prior(poly_key):
        min_d = float("inf")
        for poly in geom.get(poly_key, []):
            sd = distance_to_polygon_boundary(x, y, poly)
            if sd <= 0: return 1.0
            if sd < min_d: min_d = sd
        return osm_prior_from_signed_distance(min_d, decay_m)
    if cat == "roads":
        return _poly_prior("roads")
    elif cat == "sidewalks":
        return _poly_prior("sidewalks")
    elif cat == "cycleways":
        return _poly_prior("cycleways")
    elif cat == "parking":
        max_p = 0.0
        for poly in geom["parking"]:
            outer = _poly_outer(poly)
            if len(outer) >= 3:
                sd = distance_to_polygon_boundary(x, y, poly)
                if sd <= 0: return 1.0
                p = osm_prior_from_signed_distance(sd, decay_m)
            else:
                d = distance_to_polyline(x, y, outer)
                p = osm_prior_from_distance(d, decay_m)
            if p > max_p: max_p = p
        return max_p
    elif cat == "grasslands":
        min_d = float("inf")
        for poly in geom["grasslands"]:
            sd = distance_to_polygon_boundary(x, y, poly)
            if sd <= 0: return 1.0
            if sd < min_d: min_d = sd
        return osm_prior_from_signed_distance(min_d, decay_m)
    elif cat == "trees":
        max_p = 0.0
        for poly in geom["trees"]:
            sd = distance_to_polygon_boundary(x, y, poly)
            if sd <= 0: return 1.0
            p = osm_prior_from_signed_distance(sd, decay_m)
            if p > max_p: max_p = p
        for cx, cy in geom["tree_points"]:
            dx, dy = x - cx, y - cy
            d = math.sqrt(dx*dx + dy*dy) - tree_radius
            p = osm_prior_from_signed_distance(d, decay_m)
            if p > max_p: max_p = p
        return max_p
    elif cat == "forest":
        min_d = float("inf")
        for poly in geom["forests"]:
            sd = distance_to_polygon_boundary(x, y, poly)
            if sd <= 0: return 1.0
            if sd < min_d: min_d = sd
        return osm_prior_from_signed_distance(min_d, decay_m)
    elif cat == "buildings":
        min_d = float("inf")
        for poly in geom["buildings"]:
            sd = distance_to_polygon_boundary(x, y, poly)
            if sd <= 0: return 1.0
            if sd < min_d: min_d = sd
        return osm_prior_from_signed_distance(min_d, decay_m)
    elif cat == "fences":
        return _poly_prior("fences")
    return 0.0


def _compute_cell_prior(cx, cy, geom, idx_shapely, decay_m, tree_radius, cats):
    """Compute OSM prior vector for one grid cell. Uses Shapely if available."""
    v = np.zeros(N_OSM, dtype=np.float32)
    if idx_shapely is not None:
        for ci, cat in enumerate(cats):
            v[ci] = _compute_single_prior_shapely(cx, cy, idx_shapely, cat, decay_m, tree_radius)
    else:
        for ci, cat in enumerate(cats):
            v[ci] = _compute_single_prior(cx, cy, geom, cat, decay_m, tree_radius)
    return v


def precompute_osm_grid(geom, xmin, xmax, ymin, ymax, margin, grid_res, decay_m, tree_radius):
    """Precompute OSM prior for all grid cells in trajectory bbox + margin. Done once before scan loop."""
    gx_min = int(np.floor((xmin - margin) / grid_res))
    gx_max = int(np.floor((xmax + margin) / grid_res))
    gy_min = int(np.floor((ymin - margin) / grid_res))
    gy_max = int(np.floor((ymax + margin) / grid_res))
    idx_shapely = _build_shapely_index(geom)
    cats = list(OSM_COLUMNS)
    grid = {}
    for gx in range(gx_min, gx_max + 1):
        for gy in range(gy_min, gy_max + 1):
            cx = (gx + 0.5) * grid_res
            cy = (gy + 0.5) * grid_res
            grid[(gx, gy)] = _compute_cell_prior(cx, cy, geom, idx_shapely, decay_m, tree_radius, cats)
    return grid, idx_shapely


def build_osm_grid(pts_xy, geom, decay_m, tree_radius, grid_res=2.0, precomputed_grid=None, osm_index=None):
    """Compute OSM prior for each point via grid lookup.

    If precomputed_grid is provided, only lookups are done (very fast).
    Otherwise computes per unique cell (slower, legacy path).
    """
    n = len(pts_xy)
    keys = np.floor(pts_xy / grid_res).astype(np.int64)
    osm_vecs = np.zeros((n, N_OSM), dtype=np.float32)
    cats = list(OSM_COLUMNS)

    if precomputed_grid is not None:
        # Fast path: lookup only (rare fallback for points outside bbox)
        for i in range(n):
            k = (int(keys[i, 0]), int(keys[i, 1]))
            if k in precomputed_grid:
                v = precomputed_grid[k]
            else:
                cx = (k[0] + 0.5) * grid_res
                cy = (k[1] + 0.5) * grid_res
                v = _compute_cell_prior(cx, cy, geom, osm_index, decay_m, tree_radius, cats)
                precomputed_grid[k] = v  # cache for reuse
            # Normalize so the per-point OSM prior vector sums to 1 when any category fires.
            s = float(v.sum())
            if s > 1e-6:
                osm_vecs[i] = v / s
            else:
                osm_vecs[i] = v
        return osm_vecs

    # Legacy path: per-scan cache (when no precomputed grid)
    cache = {}
    idx = osm_index if osm_index is not None else _build_shapely_index(geom)
    for i in range(n):
        k = (int(keys[i, 0]), int(keys[i, 1]))
        if k in cache:
            v = cache[k]
        else:
            cx = (k[0] + 0.5) * grid_res
            cy = (k[1] + 0.5) * grid_res
            v = _compute_cell_prior(cx, cy, geom, idx, decay_m, tree_radius, cats)
            cache[k] = v
        # Normalize so the per-point OSM prior vector sums to 1 when any category fires.
        s = float(v.sum())
        if s > 1e-6:
            osm_vecs[i] = v / s
        else:
            osm_vecs[i] = v
    return osm_vecs


# ═══════════════════════════════════════════════════════════════════
# 4. Data loaders
# ═══════════════════════════════════════════════════════════════════

def load_poses(csv_path):
    from scipy.spatial.transform import Rotation
    poses = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader, None)
        has_header = header and any(k in "".join(header) for k in ("num", "timestamp", "x"))
        if not has_header:
            f.seek(0)
            reader = csv.reader(f)
        for row in reader:
            vals = []
            for tok in row:
                try: vals.append(float(tok))
                except ValueError: continue
            if len(vals) < 8:
                continue
            idx = int(vals[0])
            x, y, z = vals[2], vals[3], vals[4]
            qx, qy, qz = vals[5], vals[6], vals[7]
            qw = vals[8] if len(vals) > 8 else 1.0
            R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [x, y, z]
            poses.append((idx, T))
    return poses


def load_calibration(calib_path):
    with open(calib_path) as f:
        cfg = yaml.safe_load(f)
    return np.array(cfg["body"]["os_sensor"]["T"], dtype=np.float64)


def load_label_mappings(labels_common_path, key):
    with open(labels_common_path) as f:
        cfg = yaml.safe_load(f)
    raw_map = cfg.get(f"{key}_to_common", {})
    return {int(k): int(v) for k, v in raw_map.items()}


def load_learning_map_inv(labels_config_path):
    """Load learning_map_inv from labels_semkitti.yaml or labels_mcd.yaml. Returns dict model_idx -> raw_label."""
    with open(labels_config_path) as f:
        cfg = yaml.safe_load(f)
    inv = cfg.get("learning_map_inv", {})
    return {int(k): int(v) for k, v in inv.items()}


def read_multiclass_labels(path, n_points, learning_map_inv, inferred_to_common):
    """Read multiclass confidence scores (float16 stored as uint16), argmax, map to common. Returns (n_points,) int32."""
    raw = np.fromfile(path, dtype=np.uint16)
    n_vals = len(raw)
    n_classes = n_vals // n_points
    if n_vals != n_points * n_classes:
        return None
    raw = raw.reshape(n_points, n_classes)
    scores = raw.view(np.float16)
    argmax = np.argmax(scores, axis=1)
    common = np.zeros(n_points, dtype=np.int32)
    for i in range(n_points):
        model_idx = int(argmax[i])
        raw_label = learning_map_inv.get(model_idx, model_idx)
        common[i] = inferred_to_common.get(raw_label, 0)
    return common


def read_scan_bin(path):
    return np.fromfile(path, dtype=np.float32).reshape(-1, 4)


def read_label_bin(path):
    return np.fromfile(path, dtype=np.uint32)


# ═══════════════════════════════════════════════════════════════════
# 5. Co-occurrence analysis
# ═══════════════════════════════════════════════════════════════════

N_CLASSES = 9
CLASS_NAMES = [
    "unlabeled", "road", "sidewalk", "parking", "building",
    "fence", "vegetation", "vehicle", "terrain",
]


# OSM column index -> set of common GT classes compatible with that OSM category.
# Used for inferred-row: only count OSM evidence when GT aligns (to learn correction).
# Common taxonomy: 0=unlabeled, 1=road, 2=sidewalk, 3=parking, 4=building,
# 5=fence, 6=vegetation, 7=vehicle, 8=terrain.
OSM_GT_COMPATIBLE = {
    0: {1},              # roads -> road
    1: {2},              # sidewalks -> sidewalk
    2: {1, 2},           # cycleways -> road, sidewalk
    3: {3},              # parking -> parking
    4: {6, 8},           # grasslands -> vegetation, terrain
    5: {6},              # trees -> vegetation
    6: {6},              # forest -> vegetation
    7: {4},              # buildings -> building
    8: {5},              # fences -> fence
}


def build_cooccurrence(gt_labels, osm_vecs):
    """GT-row co-occurrence: counts[gt_class][osm_col]."""
    counts = np.zeros((N_CLASSES, N_OSM), dtype=np.float64)
    class_totals = np.zeros(N_CLASSES, dtype=np.float64)
    for c in range(N_CLASSES):
        mask = gt_labels == c
        class_totals[c] = mask.sum()
        if class_totals[c] > 0:
            counts[c] = osm_vecs[mask].sum(axis=0)
    return counts, class_totals


def build_cooccurrence_inferred(inferred_labels, gt_labels, osm_vecs):
    """Inferred-row co-occurrence with GT compatibility: row=inferred, weight by GT matching OSM.
    counts[inferred][j] += osm_vec[j] only when gt is in OSM_GT_COMPATIBLE[j] (correction signal)."""
    counts = np.zeros((N_CLASSES, N_OSM), dtype=np.float64)
    class_totals = np.zeros(N_CLASSES, dtype=np.float64)
    n = len(inferred_labels)
    for i in range(n):
        inf = int(inferred_labels[i])
        gt = int(gt_labels[i])
        if inf < 0 or inf >= N_CLASSES:
            continue
        class_totals[inf] += 1
        for j in range(N_OSM):
            if gt in OSM_GT_COMPATIBLE.get(j, set()):
                counts[inf][j] += osm_vecs[i, j]
    return counts, class_totals


CLASS_WEIGHT_MODES = ("points", "equal", "sqrt", "median_freq")


def _compute_row_weights(class_totals, mode):
    """Per-class row weight used before column normalization in derive_matrix.

    Modes:
      - "points":      weight[c] = class_totals[c]
                       Rows scale with point count. Favors common classes → higher
                       overall accuracy, often lower mIoU.
      - "equal":       weight[c] = 1
                       All classes contribute equally. Favors mIoU, can hurt accuracy.
      - "sqrt":        weight[c] = sqrt(class_totals[c])
                       Dampens dominant classes without silencing them. Often a good
                       compromise between accuracy and mIoU.
      - "median_freq": weight[c] = median(totals_nonzero) / class_totals[c]
                       Median-frequency balancing (Eigen et al. 2015). Rare classes
                       get large weights — strongest mIoU boost.
    """
    n_rows = N_CLASSES - 1
    totals = np.array([class_totals[cls] for cls in range(1, N_CLASSES)], dtype=np.float64)
    if mode == "points":
        return totals
    if mode == "equal":
        return np.ones(n_rows, dtype=np.float64)
    if mode == "sqrt":
        return np.sqrt(np.maximum(totals, 0.0))
    if mode == "median_freq":
        nonzero = totals[totals > 0]
        med = float(np.median(nonzero)) if nonzero.size > 0 else 0.0
        w = np.zeros(n_rows, dtype=np.float64)
        for i, t in enumerate(totals):
            if t > 1e-10:
                w[i] = med / t
        return w
    raise ValueError(f"Unknown class_weight_mode: {mode!r} (choose from {CLASS_WEIGHT_MODES})")


def _column_selectivity(matrix):
    """Per-column selectivity in [0, 1] based on normalized entropy.

    For each column (viewed as a distribution over the class rows), compute
    1 - H(col) / log(n_rows). A one-hot column scores 1 (perfect indicator of a
    single class); a uniform column scores 0 (OSM tells us nothing). Empty or
    all-zero columns score 0.
    """
    n_rows, n_cols = matrix.shape
    sel = np.zeros(n_cols, dtype=np.float64)
    if n_rows < 2:
        return sel
    h_max = math.log(n_rows)
    for j in range(n_cols):
        col = matrix[:, j]
        s = col.sum()
        if s <= 1e-12:
            continue
        p = col / s
        h = 0.0
        for pi in p:
            if pi > 1e-12:
                h -= pi * math.log(pi)
        sel[j] = max(0.0, min(1.0, 1.0 - h / h_max))
    return sel


def _compute_class_prior(class_totals, class_weight_mode):
    """Prior distribution P(c) over labeled classes (1..N_CLASSES-1), sums to 1.

    Shape controlled by `class_weight_mode` (same vocabulary as _compute_row_weights):
      - "points":      empirical marginal (proportional to GT point count)
      - "equal":       uniform
      - "sqrt":        proportional to sqrt(count) — dampened marginal
      - "median_freq": proportional to median(count) / count — inverse-frequency
    """
    n_rows = N_CLASSES - 1
    totals = np.array([class_totals[cls] for cls in range(1, N_CLASSES)], dtype=np.float64)
    total = totals.sum()
    if total <= 0:
        return np.ones(n_rows) / n_rows
    if class_weight_mode == "points":
        w = totals
    elif class_weight_mode == "equal":
        w = np.ones(n_rows, dtype=np.float64)
    elif class_weight_mode == "sqrt":
        w = np.sqrt(np.maximum(totals, 0.0))
    elif class_weight_mode == "median_freq":
        nonzero = totals[totals > 0]
        med = float(np.median(nonzero)) if nonzero.size > 0 else 1.0
        w = np.array([med / t if t > 1e-10 else 0.0 for t in totals], dtype=np.float64)
    else:
        raise ValueError(f"Unknown class_weight_mode: {class_weight_mode!r}")
    s = w.sum()
    return w / s if s > 1e-12 else np.ones(n_rows) / n_rows


def derive_matrix_shrinkage(counts, class_totals, kappa=10.0,
                            class_weight_mode="points", selectivity_weight=False):
    """Derive the OSM confusion matrix via Bayesian shrinkage toward a class prior.

    For each OSM column j:
        M[:, j] = n_j/(n_j+kappa) · P(c|j) + kappa/(n_j+kappa) · P(c)

    where n_j = labeled-point co-occurrence count for column j,
    P(c|j) = counts[c,j]/n_j (empirical posterior over labeled classes), and
    P(c) is the prior shaped by `class_weight_mode`. Each column sums to 1 by
    construction.

    Why this beats the lift-based `max(0, P(c|j)/P(c) - 1)` derivation:
      - Data-poor columns gracefully revert to the prior rather than producing a
        post-normalized column out of a few noisy samples.
      - Cells where a class is genuinely UNDER-represented are no longer clipped
        to zero; the full signed signal is preserved.
      - κ is one interpretable knob (how many "prior pseudo-samples" equal one
        real sample). Low κ trusts the data; high κ trusts the prior.
    """
    n_rows = N_CLASSES - 1
    # Rows are emitted for classes 1..N_CLASSES-1 (exclude class 0 / unlabeled).
    counts_lbl = counts[1:] if counts.shape[0] == N_CLASSES else counts
    prior = _compute_class_prior(class_totals, class_weight_mode)
    col_totals = counts_lbl.sum(axis=0)
    matrix = np.zeros((n_rows, N_OSM))
    for j in range(N_OSM):
        n_j = float(col_totals[j])
        denom = n_j + kappa
        if denom <= 0.0:
            matrix[:, j] = prior
            continue
        w_emp = n_j / denom
        w_prior = kappa / denom
        if n_j > 1e-10:
            p_emp = counts_lbl[:, j] / n_j
        else:
            p_emp = np.zeros(n_rows)
        matrix[:, j] = w_emp * p_emp + w_prior * prior
    if selectivity_weight:
        sel = _column_selectivity(matrix)
        matrix = matrix * sel[np.newaxis, :]
    return matrix


def derive_matrix(counts, class_totals, class_weight_mode="points",
                  selectivity_weight=False):
    """Derive (N_CLASSES-1) x N_OSM matrix from co-occurrence (rows = classes 1..N_CLASSES-1).

    By default each column sums to 1, so matrix[c-1, j] is P(class=c | osm_col=j)
    restricted to classes 1..N_CLASSES-1 (excluding unlabeled).

    See _compute_row_weights for the `class_weight_mode` options.

    If `selectivity_weight=True`, each column is additionally multiplied by its
    selectivity score in [0, 1] (1 - normalized entropy). Columns with a peaked
    class distribution keep near-unit mass (strong unique association); uniform
    columns (no discriminative power) fade toward zero. This makes the effective
    OSM-prior strength proportional to how informative each OSM class actually is.
    """
    n_rows = N_CLASSES - 1
    total_points = class_totals.sum()
    if total_points == 0:
        return np.zeros((n_rows, N_OSM))
    col_totals = counts.sum(axis=0)
    matrix = np.zeros((n_rows, N_OSM))
    for ri, cls in enumerate(range(1, N_CLASSES)):
        p_cls = class_totals[cls] / total_points
        for j in range(N_OSM):
            if col_totals[j] < 1.0 or p_cls < 1e-8:
                matrix[ri][j] = 0.0
            else:
                # Boost = max(0, P(c|osm)/P(c) - 1): non-negative correction signal.
                p_cls_given_osm = counts[cls][j] / col_totals[j]
                matrix[ri][j] = max(0.0, p_cls_given_osm / p_cls - 1.0)
    row_weights = _compute_row_weights(class_totals, class_weight_mode)
    for ri in range(n_rows):
        matrix[ri, :] *= row_weights[ri]
    # Per-column probability normalization: each column sums to 1.
    for j in range(N_OSM):
        col_sum = matrix[:, j].sum()
        if col_sum > 1e-10:
            matrix[:, j] /= col_sum
    # Optional: weight by per-column selectivity so weak columns fade.
    if selectivity_weight:
        sel = _column_selectivity(matrix)
        matrix = matrix * sel[np.newaxis, :]
    return matrix


def compute_num_bins(step_meters, max_meters):
    return int(np.ceil(float(max_meters) / float(step_meters)))


def build_cooccurrence_height_class(scan_data_list, step_meters=1.0, max_meters=30.0):
    """Fixed-metric height bins measured upward from the bottom-most lidar point of
    each scan (z_local = projection onto lidar z). For each scan,
    z_base = min(z_local) is the zero reference; bin = floor((z_local - z_base)/step),
    clamped to [0, num_bins-1] where num_bins = ceil(max_meters / step_meters).
    Making height relative to the scan's bottom removes dependence on the sensor's
    absolute z.

    counts_h[bin][class_idx] is a pure GT-point histogram — each valid point contributes
    unit weight. The distribution P(class | z_bin) is a property of the data, NOT of
    OSM overlap, so no OSM signal is used here (fixes prior behavior that biased the
    histogram toward OSM-overlapping points)."""
    num_bins = compute_num_bins(step_meters, max_meters)
    n_rows = N_CLASSES - 1
    counts_h = np.zeros((num_bins, n_rows), dtype=np.float64)
    for item in scan_data_list:
        z_local, gt = item[0], item[1]  # OSM component intentionally ignored
        gt = np.asarray(gt, dtype=np.int32)
        z_arr = np.asarray(z_local, dtype=np.float64)
        if z_arr.size == 0:
            continue
        z_base = float(z_arr.min())
        mask = (gt >= 1) & (gt < N_CLASSES)
        if not mask.any():
            continue
        z_m = z_arr[mask] - z_base
        gt_m = gt[mask]
        bins = np.clip(np.floor(z_m / float(step_meters)).astype(np.int32), 0, num_bins - 1)
        cols = gt_m - 1
        np.add.at(counts_h, (bins, cols), 1.0)
    return counts_h


def derive_height_matrix(counts_h, low_percentile=10.0, high_percentile=90.0):
    """Derive class-indexed height trust matrix: H[bin][class_idx] in [0, 1].

    For each common class c (column), H[:, c-1] is the bin-histogram of class c points
    normalized so that the peak bin equals 1.0. A high value means "class c is typical
    at this height"; a low value means "unusual for class c at this height". Classes
    with no observations are set to neutral (1.0).

    Outliers are smoothed by clipping each column to [low_percentile, high_percentile]
    of that column's values before the final per-column max-normalization."""
    num_bins = counts_h.shape[0]
    n_rows = counts_h.shape[1]
    matrix = np.zeros((num_bins, n_rows))
    for c in range(n_rows):
        col = counts_h[:, c].astype(np.float64)
        if col.sum() <= 1e-10:
            matrix[:, c] = 1.0
            continue
        lo = np.percentile(col, low_percentile)
        hi = np.percentile(col, high_percentile)
        col = np.clip(col, lo, hi)
        peak = float(col.max())
        if peak > 1e-10:
            matrix[:, c] = col / peak
        else:
            matrix[:, c] = 1.0
    matrix = np.clip(matrix, 0.0, 1.0)
    return matrix


# ═══════════════════════════════════════════════════════════════════
# 6. Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_points_and_osm_heatmap(map_pts_xy, gt_labels, geom, osm_vecs, save_path=None, show=True,
                                subsample=10, grid_res=2.0):
    """Plot GT points (colored by class) over OSM geometry and OSM prior heatmap."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.patches import Polygon
        from matplotlib.collections import LineCollection
    except ImportError:
        print("Visualization skipped: matplotlib not installed (pip install matplotlib)")
        return

    xy = np.asarray(map_pts_xy, dtype=np.float64)
    labels = np.asarray(gt_labels, dtype=np.int32)
    osm = np.asarray(osm_vecs, dtype=np.float64)

    # Subsample for scatter
    if subsample > 1:
        idx = np.arange(0, len(xy), subsample)
        xy_plot = xy[idx]
        labels_plot = labels[idx]
    else:
        xy_plot, labels_plot = xy, labels

    # Class colors (match labels_common.yaml roughly)
    class_colors = [
        (0, 0, 0), (0.5, 0.25, 0.5), (0.9, 0.14, 0.96), (0.63, 0.67, 0.98),
        (0.29, 0, 0.69), (1, 0.78, 0), (1, 0.47, 0.2), (1, 0.94, 0.59),
        (1, 0, 0), (0, 0.69, 0), (0.96, 0.9, 0.39), (0.39, 0.59, 0.96),
        (0.2, 1, 1),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # --- Left: OSM geometry heatmap + points ---
    ax1 = axes[0]
    # Rasterize OSM prior (max over 8 categories) into grid
    xmin, xmax = xy[:, 0].min(), xy[:, 0].max()
    ymin, ymax = xy[:, 1].min(), xy[:, 1].max()
    nx = int((xmax - xmin) / grid_res) + 1
    ny = int((ymax - ymin) / grid_res) + 1
    heatmap = np.zeros((ny, nx))
    for i in range(len(xy)):
        ix = int((xy[i, 0] - xmin) / grid_res)
        iy = int((xy[i, 1] - ymin) / grid_res)
        if 0 <= ix < nx and 0 <= iy < ny:
            heatmap[iy, ix] = max(heatmap[iy, ix], osm[i].max())
    extent = (xmin, xmax, ymin, ymax)
    ax1.imshow(heatmap, origin="lower", extent=extent, aspect="auto",
               cmap="Greens", alpha=0.5, vmin=0, vmax=1)

    # OSM geometry outlines (use outer ring for polygons with holes)
    for poly in geom["buildings"][:200]:  # limit for speed
        ring = _poly_outer(poly)
        if len(ring) >= 3:
            p = Polygon(ring, fill=False, edgecolor="brown", linewidth=0.5, alpha=0.7)
            ax1.add_patch(p)
    for poly in geom["roads"][:100]:
        ring = _poly_outer(poly)
        if len(ring) >= 2:
            ax1.plot([r[0] for r in ring], [r[1] for r in ring], "k-", linewidth=0.8, alpha=0.5)
    for poly in geom["grasslands"][:50]:
        ring = _poly_outer(poly)
        if len(ring) >= 3:
            p = Polygon(ring, fill=False, edgecolor="green", linewidth=0.4, alpha=0.5)
            ax1.add_patch(p)
    for poly in geom["trees"][:30] + geom["forests"][:30]:
        ring = _poly_outer(poly)
        if len(ring) >= 3:
            p = Polygon(ring, fill=False, edgecolor="darkgreen", linewidth=0.3, alpha=0.5)
            ax1.add_patch(p)

    # Points colored by class
    for cls in range(1, N_CLASSES):
        mask = labels_plot == cls
        if mask.sum() == 0:
            continue
        c = class_colors[cls]
        ax1.scatter(xy_plot[mask, 0], xy_plot[mask, 1], c=[c], s=1, alpha=0.6,
                    label=CLASS_NAMES[cls], rasterized=True)
    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("GT points (colored by class) + OSM geometry + prior heatmap")
    ax1.legend(loc="upper left", fontsize=6, ncol=2)
    ax1.set_aspect("equal")
    ax1.grid(True, alpha=0.3)

    # --- Right: Point density heatmap (all points) + OSM outlines ---
    ax2 = axes[1]
    h, _, _ = np.histogram2d(xy[:, 0], xy[:, 1], bins=(nx, ny),
                             range=[[xmin, xmax], [ymin, ymax]])
    vmax = np.percentile(h[h > 0], 99) if (h > 0).any() else 1.0
    ax2.imshow(h.T, origin="lower", extent=extent, aspect="auto",
               cmap="viridis", vmin=0, vmax=max(vmax, 1))
    for poly in geom["roads"][:150]:
        ring = _poly_outer(poly)
        if len(ring) >= 2:
            ax2.plot([r[0] for r in ring], [r[1] for r in ring], "gray", linewidth=0.6, alpha=0.8)
    for poly in geom["buildings"][:300]:
        ring = _poly_outer(poly)
        if len(ring) >= 3:
            p = Polygon(ring, fill=False, edgecolor="brown", linewidth=0.4, alpha=0.6)
            ax2.add_patch(p)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("Point density heatmap + OSM geometry (roads, buildings)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved points+OSM visualization to {save_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_osm_confusion_matrix(matrix, save_path=None, show=True):
    """Plot the OSM confusion matrix as a heatmap."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("Visualization skipped: matplotlib not installed (pip install matplotlib)")
        return

    fig, ax = plt.subplots(figsize=(max(14, N_OSM * 1.2), 8))
    # Diverging colormap: blue = negative (suppress), white = 0, red = positive (boost)
    cmap = plt.cm.RdBu_r
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(range(N_OSM))
    ax.set_xticklabels(OSM_COLUMNS, rotation=45, ha="right")
    ax.set_yticks(range(N_CLASSES - 1))
    ax.set_yticklabels([CLASS_NAMES[i + 1] for i in range(N_CLASSES - 1)])

    # Add value annotations
    for i in range(N_CLASSES - 1):
        for j in range(N_OSM):
            v = matrix[i, j]
            text_color = "white" if abs(v) > 0.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=text_color)

    ax.set_xlabel("OSM category")
    ax.set_ylabel("Semantic class")
    ax.set_title("OSM confusion matrix (optimized)\nblue=suppress, red=boost, white=neutral")
    plt.colorbar(im, ax=ax, label="prior bias [-1, 1]")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


# ═══════════════════════════════════════════════════════════════════
# 7. YAML output
# ═══════════════════════════════════════════════════════════════════

def write_osm_cm_yaml(matrix, output_path, height_matrix=None, geometry_params=None,
                       height_step_meters=None, height_max_meters=None):
    val_comment = "# Values in [0.0, 1.0]: each column is a probability distribution that sums to 1."
    cols_str = ", ".join(OSM_COLUMNS)
    n_rows = N_CLASSES - 1
    class_legend = ", ".join(f"{i+1}: {CLASS_NAMES[i+1]}" for i in range(n_rows))
    lines = [
        "# Optimized OSM confusion matrix (derived from GT co-occurrence analysis).",
        f"# Rows  = common taxonomy semantic classes (1-{n_rows}).",
        f"# Cols  = OSM prior categories: [{cols_str}]",
        val_comment,
        "#",
        "# Common taxonomy:",
        f"#   {class_legend}",
        "",
        f"osm_prior_columns: [{cols_str}]",
        "",
        "confusion_matrix:",
    ]
    for i in range(N_CLASSES - 1):
        vals = ", ".join(f"{matrix[i, j]:6.2f}" for j in range(N_OSM))
        lines.append(f"  {i+1}:  [{vals}]   # {CLASS_NAMES[i+1]}")
    if height_matrix is not None:
        n_bins = height_matrix.shape[0]
        n_class_cols = height_matrix.shape[1]
        class_cols_str = ", ".join(CLASS_NAMES[i + 1] for i in range(n_class_cols))
        step_str = f"{height_step_meters:.3f}" if height_step_meters is not None else "?"
        max_str = f"{height_max_meters:.3f}" if height_max_meters is not None else "?"
        step_v = float(height_step_meters) if height_step_meters is not None else 0.0
        lines += [
            "",
            f"# OSM height confusion matrix (fixed-metric bins measured upward from the "
            f"bottom-most lidar point of each scan). Step = {step_str} m, max height = "
            f"{max_str} m, num_bins = {n_bins}. Bin i (1-indexed) covers "
            f"height_above_bottom in [(i-1)*step, i*step] m; bin 1 is the bottom-most, "
            f"bin {n_bins} is the top-most. Cols = common-taxonomy classes "
            f"1..{n_class_cols}: [{class_cols_str}]. Multipliers in [0, 1] applied to "
            "p_super[class] after the OSM→common projection.",
            f"osm_height_bin_step_meters: {step_str}",
            f"osm_height_max_meters: {max_str}",
            "osm_height_confusion_matrix:",
        ]
        for b in range(n_bins):
            z_lo = b * step_v
            z_hi = z_lo + step_v
            vals = ", ".join(f"{height_matrix[b, j]:6.2f}" for j in range(n_class_cols))
            lines.append(f"  {b+1}:  [{vals}]   # BIN {b+1} h_above_bottom in [{z_lo:.2f}, {z_hi:.2f}] m")
    lines += [
        "", "osm_class_map:",
        *[f"  {name}: {j}" for j, name in enumerate(OSM_COLUMNS)],
    ]
    if geometry_params is not None:
        lines += [
            "",
            "# Geometry parameters used for polygon-band OSM priors and visualization.",
            "osm_geometry_parameters:",
            f"  osm_decay_meters: {geometry_params['osm_decay_meters']:.3f}",
            f"  road_width_meters: {geometry_params['road_width_meters']:.3f}",
            f"  sidewalk_width_meters: {geometry_params['sidewalk_width_meters']:.3f}",
            f"  cycleway_width_meters: {geometry_params['cycleway_width_meters']:.3f}",
            f"  fence_width_meters: {geometry_params['fence_width_meters']:.3f}",
            f"  tree_point_radius_meters: {geometry_params['tree_point_radius_meters']:.3f}",
        ]
    lines += [
        "", "label_to_matrix_idx:",
        *[f"  {i+1}: {i}" for i in range(N_CLASSES - 1)],
        "",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════

def _process_sequence_scans(
    seq, cfg, args, data_dir, *,
    gt_mapping, inferred_mapping, learning_map_inv,
    inferred_label_prefix, inferred_use_multiclass,
    body_to_lidar, lidar_to_body,
    decay_m, keyframe_dist, max_range, ds_resolution, tree_radius,
    all_gt, all_osm, all_inferred, scan_data_list, all_map_pts,
):
    """Load one sequence's poses + OSM + scans, and extend the passed accumulator lists.

    The sequence is processed in its own first-pose-relative frame; statistics are
    frame-invariant so data from multiple sequences can be concatenated for a single
    derivation at the end. Returns the trimmed `geom` dict for the sequence (useful
    for downstream visualization), or None if no scans were found.
    """
    # Resolve per-sequence suffix-based paths (shallow-merged onto cfg).
    seq_cfg = dict(cfg)
    if seq:
        if seq_cfg.get("lidar_pose_suffix"):
            seq_cfg["lidar_pose_file"] = f"{seq}/{seq_cfg['lidar_pose_suffix']}"
        if seq_cfg.get("input_data_suffix"):
            seq_cfg["input_data_prefix"] = f"{seq}/{seq_cfg['input_data_suffix']}"
        if seq_cfg.get("gt_label_suffix"):
            seq_cfg["gt_label_prefix"] = f"{seq}/{seq_cfg['gt_label_suffix']}"
        if seq_cfg.get("input_label_suffix"):
            seq_cfg["input_label_prefix"] = f"{seq}/{seq_cfg['input_label_suffix']}"

    pose_file = os.path.join(data_dir, seq_cfg.get("lidar_pose_file", ""))
    gt_label_dir = os.path.join(data_dir, seq_cfg.get("gt_label_prefix", ""))
    scan_dir = os.path.join(data_dir, seq_cfg.get("input_data_prefix", ""))
    osm_file = os.path.join(data_dir, seq_cfg.get("osm_file", ""))
    origin_lat = seq_cfg.get("osm_origin_lat", 0.0)
    origin_lon = seq_cfg.get("osm_origin_lon", 0.0)
    seq_inferred_dir = os.path.join(data_dir, inferred_label_prefix) if args.use_inferred_row else None

    label_tag = seq or "default"
    print(f"\n=== Sequence: {label_tag} ===")
    print(f"  pose={pose_file}")
    print(f"  gt_labels={gt_label_dir}")
    print(f"  scans={scan_dir}")
    print(f"  osm={osm_file}")

    print(f"  Loading poses...")
    poses = load_poses(pose_file)
    print(f"  Loaded {len(poses)} poses")
    if not poses:
        print(f"  No poses; skipping sequence.")
        return None

    # Transform poses to first-pose-relative frame so inter-sequence coordinate
    # systems don't collide (co-occurrence statistics are frame-invariant).
    first_pose = poses[0][1].copy()
    first_inv = np.linalg.inv(first_pose)
    for i in range(len(poses)):
        poses[i] = (poses[i][0], first_inv @ poses[i][1])

    print(f"  Parsing OSM geometry from {osm_file}")
    geom = parse_osm_xml(osm_file, origin_lat, origin_lon)
    for cat in geom:
        print(f"    {cat}: {len(geom[cat])} items")

    def _transform_ring(ring):
        for k in range(len(ring)):
            x, y = ring[k]
            p = first_inv @ np.array([x, y, 0, 1])
            ring[k] = (float(p[0]), float(p[1]))
    for cat in ("buildings", "roads", "grasslands", "trees", "forests",
                "parking", "fences", "sidewalks", "cycleways"):
        if cat not in geom:
            continue
        for item in geom[cat]:
            if _poly_holes(item):
                outer, holes = _poly_outer(item), _poly_holes(item)
                _transform_ring(outer)
                for h in holes:
                    _transform_ring(h)
            else:
                ring = _poly_outer(item)
                _transform_ring(ring)
    for pt_key in ("tree_points",):
        if pt_key not in geom:
            continue
        for k in range(len(geom[pt_key])):
            x, y = geom[pt_key][k]
            p = first_inv @ np.array([x, y, 0, 1])
            geom[pt_key][k] = (float(p[0]), float(p[1]))

    # Build keyframe-filtered scan list.
    scan_list = []
    last_keyframe_pos = None
    for idx, T in poses:
        sp = os.path.join(scan_dir, f"{idx:010d}.bin")
        lp = os.path.join(gt_label_dir, f"{idx:010d}.bin")
        if not (os.path.isfile(sp) and os.path.isfile(lp)):
            continue
        if args.use_inferred_row:
            ip = os.path.join(seq_inferred_dir, f"{idx:010d}.bin")
            if not os.path.isfile(ip):
                continue
        else:
            ip = None
        pos = T[:3, 3]
        if last_keyframe_pos is not None and keyframe_dist > 0:
            if np.linalg.norm(pos - last_keyframe_pos) < keyframe_dist:
                continue
        scan_list.append((idx, T, sp, lp, ip))
        last_keyframe_pos = pos
        if len(scan_list) >= args.max_scans:
            break

    if not scan_list:
        print(f"  No valid scans found; skipping sequence.")
        return None

    xs = [T[0, 3] for _, T, _, _, _ in scan_list]
    ys = [T[1, 3] for _, T, _, _, _ in scan_list]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    margin = max_range + decay_m
    geom_trimmed = trim_osm_to_bbox(geom, xmin, xmax, ymin, ymax, margin)
    n_before = sum(len(geom[c]) for c in geom)
    n_after = sum(len(geom_trimmed[c]) for c in geom_trimmed)
    print(f"  OSM trim: kept {n_after}/{n_before} items in bbox "
          f"[x:{xmin:.0f}..{xmax:.0f}, y:{ymin:.0f}..{ymax:.0f}] margin={margin:.0f}m")
    geom = geom_trimmed

    print("  Precomputing OSM prior grid...")
    precomputed_grid, osm_index = precompute_osm_grid(
        geom, xmin, xmax, ymin, ymax, margin, args.grid_res, decay_m, tree_radius
    )
    print(f"    Grid has {len(precomputed_grid)} cells "
          f"(Shapely={'on' if osm_index else 'off'})")

    # Lidar up reference (first scan of this sequence; after first-pose normalization
    # poses[0][1] is identity, so lidar_to_map = lidar_to_body).
    _first_lidar_to_map = poses[0][1] @ lidar_to_body
    lidar_up_ref = _first_lidar_to_map[:3, 2].astype(np.float64)
    _n = np.linalg.norm(lidar_up_ref)
    if _n > 1e-6:
        lidar_up_ref = lidar_up_ref / _n

    print(f"  Processing {len(scan_list)} scans "
          f"(keyframe_dist={keyframe_dist}m, max={args.max_scans}, grid_res={args.grid_res}m)")
    for si, (idx, T, scan_path, label_path, inferred_path) in enumerate(
            tqdm(scan_list, desc=f"scans [{label_tag}]", unit="scan")):
        pts = read_scan_bin(scan_path)
        labels_raw = read_label_bin(label_path)
        n = min(len(pts), len(labels_raw))
        pts, labels_raw = pts[:n], labels_raw[:n]
        gt_common = np.array([gt_mapping.get(int(l), 0) for l in labels_raw], dtype=np.int32)

        if args.use_inferred_row and inferred_path is not None:
            if inferred_use_multiclass:
                inferred_common = read_multiclass_labels(
                    inferred_path, n, learning_map_inv, inferred_mapping
                )
            else:
                inf_raw = read_label_bin(inferred_path)
                inf_raw = inf_raw[:n]
                inferred_common = np.array(
                    [inferred_mapping.get(int(l), 0) for l in inf_raw], dtype=np.int32
                )
            if inferred_common is None:
                inferred_common = gt_common.copy()
        else:
            inferred_common = None

        lidar_to_map = T @ lidar_to_body
        xyz_h = np.hstack([pts[:, :3], np.ones((n, 1), dtype=np.float32)])
        map_pts = (lidar_to_map @ xyz_h.T).T[:, :3]

        if ds_resolution > 0:
            vkeys = np.floor(map_pts / ds_resolution).astype(np.int64)
            _, uidx = np.unique(vkeys, axis=0, return_index=True)
            map_pts = map_pts[uidx]
            gt_common = gt_common[uidx]
            if inferred_common is not None:
                inferred_common = inferred_common[uidx]

        osm_vecs = build_osm_grid(
            map_pts[:, :2], geom, decay_m, tree_radius, args.grid_res,
            precomputed_grid=precomputed_grid, osm_index=osm_index,
        )

        all_gt.append(gt_common)
        all_osm.append(osm_vecs)
        if all_inferred is not None and inferred_common is not None:
            all_inferred.append(inferred_common)
        if all_map_pts is not None:
            all_map_pts.append(map_pts[:, :2])
        if scan_data_list is not None:
            origin_map = lidar_to_map[:3, 3]
            z_local = (map_pts - origin_map) @ lidar_up_ref
            item = (z_local.astype(np.float64), gt_common.copy(), osm_vecs.copy())
            if all_inferred is not None and inferred_common is not None:
                item = (*item, inferred_common.copy())
            scan_data_list.append(item)

    return geom


def main():
    parser = argparse.ArgumentParser(description="Optimize OSM confusion matrix from GT data.")

    # Base config options
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config/methods/mcd.yaml"))
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "config/datasets/osm_confusion_matrix_optimized_MCD_NEW.yaml"))
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--sequences", type=str, default=None,
                        help=("Comma-separated list of sequence names to process "
                              "(e.g. 'kth_day_09,kth_day_10,kth_night_04'). Each sequence's "
                              "scans contribute to the SAME aggregated co-occurrence matrix. "
                              "If omitted, falls back to the single 'sequence_name' from --config."))
    parser.add_argument("--max-scans", type=int, default=50000,
                        help="Per-sequence scan cap (applied independently to each entry in --sequences).")
    parser.add_argument("--keyframe-dist", type=float, default=5.0)
    parser.add_argument("--decay", type=float, default=2.0)
    parser.add_argument("--tree-radius", type=float, default=5.0)
    parser.add_argument("--road-width", type=float, default=None, help="Override road width (m); default from osm_bki.yaml")
    parser.add_argument("--sidewalk-width", type=float, default=None, help="Override sidewalk width (m); default from osm_bki.yaml")
    parser.add_argument("--cycleway-width", type=float, default=None, help="Override cycleway width (m); default from osm_bki.yaml")
    parser.add_argument("--fence-width", type=float, default=None, help="Override fence width (m); default from osm_bki.yaml")
    parser.add_argument("--grid-res", type=float, default=0.5)

    # Height matrix options (fixed-metric bins along lidar z, symmetric about sensor origin)
    parser.add_argument("--height-step-meters", type=float, default=0.1,
                        help="Per-bin height step in meters (default: 1.0)")
    parser.add_argument("--height-max-meters", type=float, default=50.0,
                        help="Max |z_local| extent in meters; num_bins = 2*ceil(max/step) (default: 30.0)")
    parser.add_argument("--no-height-matrix", action="store_true",
                        help="Skip computing and writing osm_height_confusion_matrix")
    
    # Visualization options
    parser.add_argument("--visualize", action="store_true",
                        help="Plot the optimized matrix as a heatmap and optionally save to PNG")
    parser.add_argument("--visualize-points", action="store_true",
                        help="Plot GT points and OSM geometry with OSM prior heatmap (spatial view)")
    parser.add_argument("--no-show", action="store_true",
                        help="With --visualize/--visualize-points: save PNG but do not block to show the figure")
    parser.add_argument("--vis-output", type=str, default=None,
                        help="With --visualize: path for matrix PNG (default: <output>.png)")
    parser.add_argument("--vis-points-output", type=str, default=None,
                        help="With --visualize-points: path for points+OSM PNG (default: <output>_points_osm.png)")
    
    # Options for inferred-row variant (alternative to GT-row)
    parser.add_argument("--use-inferred-row", action="store_true",
                        help="Use inferred (model) labels as matrix rows. Optimizes OSM to correct inferred toward GT.")
    parser.add_argument("--inferred-prefix", type=str, default=None,
                        help="Override input_label_prefix (e.g. kth_day_09/inferred_labels/cenet_mcd_EDL/multiclass_confidence_scores)")
    parser.add_argument("--inferred-key", type=str, default=None,
                        help="Override inferred_labels_key (mcd or semkitti) for label mapping")
    
    # Row-weighting scheme for the confusion matrix (balances accuracy vs mIoU).
    parser.add_argument("--class-weight-mode", type=str, default="sqrt",
                        choices=list(CLASS_WEIGHT_MODES),
                        help=("Row weighting: points (accuracy-leaning), equal, "
                              "sqrt (compromise), median_freq (mIoU-leaning). "
                              "Default: points."))
    # Legacy aliases — preserved for backward compatibility.
    parser.add_argument("--scale-by-class-points", action="store_true", default=None,
                        dest="_legacy_scale_true",
                        help="[deprecated] alias for --class-weight-mode=points")
    parser.add_argument("--no-scale-by-class-points", action="store_true", default=None,
                        dest="_legacy_scale_false",
                        help="[deprecated] alias for --class-weight-mode=equal")
    parser.add_argument("--selectivity-weight", action="store_true", default=True,
                        help=("Scale each OSM column by its selectivity (1 - normalized "
                              "entropy). Weak/uniform columns fade toward zero so the "
                              "prior boost is proportional to how uniquely an OSM class "
                              "predicts a semantic class."))
    parser.add_argument("--shrinkage-kappa", type=float, default=None,
                        help=("Enable calibrated-posterior derivation via Bayesian "
                              "shrinkage: M[c,j] = n_j/(n_j+κ)·P(c|j) + κ/(n_j+κ)·P(c). "
                              "The prior P(c) is shaped by --class-weight-mode. Higher κ "
                              "shrinks data-poor columns toward the prior; lower κ trusts "
                              "the empirical posterior. Typical values 5–50. If unset, "
                              "the default lift-based derivation is used."))

    args = parser.parse_args()

    # Resolve row-weighting mode (explicit flag wins; otherwise legacy alias; otherwise "points").
    if args.class_weight_mode is None:
        if args._legacy_scale_false:
            args.class_weight_mode = "equal"
            print("[deprecation] --no-scale-by-class-points; use --class-weight-mode=equal")
        elif args._legacy_scale_true:
            args.class_weight_mode = "points"
            print("[deprecation] --scale-by-class-points; use --class-weight-mode=points")
        else:
            args.class_weight_mode = "points"

    print(f"Loading config from {args.config}")
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = raw_cfg
    if "/**" in cfg:
        cfg = cfg["/**"].get("ros__parameters", cfg)
    elif "ros__parameters" in cfg:
        cfg = cfg["ros__parameters"]

    # Resolve sequence list: --sequences CLI > single cfg[sequence_name] > sentinel [None]
    # (None means "use cfg paths as-is without suffix substitution").
    if args.sequences:
        seq_list = [s.strip() for s in args.sequences.split(",") if s.strip()]
    else:
        _single = cfg.get("sequence_name")
        seq_list = [_single] if _single else [None]

    # Data directory: --data-dir override, else data_root from config, else package data/<dataset_name>
    if args.data_dir is not None and args.data_dir.strip():
        data_dir = os.path.abspath(args.data_dir.strip())
    else:
        data_root = (cfg.get("data_root") or "").strip()
        dataset_name = cfg.get("dataset_name", "mcd")
        if data_root:
            data_dir = os.path.join(data_root, dataset_name)
        else:
            data_dir = os.path.join(SCRIPT_DIR, "data", dataset_name)
    print(f"Using data directory: {data_dir}")
    print(f"Processing {len(seq_list)} sequence(s): {seq_list}")

    # Shared (non per-sequence) config values.
    gt_labels_key = cfg.get("gt_labels_key", "mcd")
    inferred_label_prefix = args.inferred_prefix if args.inferred_prefix else cfg.get(
        "input_label_prefix", "kth_day_09/inferred_labels/cenet_semkitti/multiclass_confidence_scores"
    )
    inferred_labels_key = args.inferred_key if args.inferred_key else cfg.get("inferred_labels_key", "semkitti")
    inferred_use_multiclass = cfg.get("inferred_use_multiclass", True)
    calib_file = os.path.join(data_dir, "hhs_calib.yaml")
    decay_m = args.decay if args.decay is not None else cfg.get("osm_decay_meters", 5.0)

    # 1. Load defaults from osm_bki.yaml
    osm_bki_path = os.path.join(SCRIPT_DIR, "config/methods/osm_bki.yaml")
    geom_params_cfg = _load_geom_defaults(osm_bki_path)
    print(f"Loaded geometry defaults from {osm_bki_path}")

    # 2. Resize road/sidewalk/cycleway from the referenced confusion matrix yaml (if it exists)
    osm_cm_file = cfg.get("osm_confusion_matrix_file")
    if osm_cm_file:
        osm_cm_path = os.path.join(SCRIPT_DIR, "config/datasets", osm_cm_file)
        if os.path.isfile(osm_cm_path):
            with open(osm_cm_path) as _f:
                osm_cm_cfg = yaml.safe_load(_f)
            cm_geom = osm_cm_cfg.get("osm_geometry_parameters", {})
            for _key in ("road_width_meters", "sidewalk_width_meters", "cycleway_width_meters"):
                if _key in cm_geom:
                    geom_params_cfg[_key] = cm_geom[_key]
            print(f"Resized road/sidewalk/cycleway widths from {osm_cm_path}")

    # 3. Dataset config overrides
    geom_params_cfg["tree_point_radius_meters"] = cfg.get(
        "osm_tree_point_radius_meters", geom_params_cfg["tree_point_radius_meters"]
    )
    geom_params_cfg["road_width_meters"] = cfg.get("road_width_meters", geom_params_cfg["road_width_meters"])
    geom_params_cfg["sidewalk_width_meters"] = cfg.get("sidewalk_width_meters", geom_params_cfg["sidewalk_width_meters"])
    geom_params_cfg["cycleway_width_meters"] = cfg.get("cycleway_width_meters", geom_params_cfg["cycleway_width_meters"])
    geom_params_cfg["fence_width_meters"] = cfg.get("fence_width_meters", geom_params_cfg["fence_width_meters"])

    # 4. CLI overrides (only when explicitly provided)
    if args.road_width is not None:
        geom_params_cfg["road_width_meters"] = args.road_width
    if args.sidewalk_width is not None:
        geom_params_cfg["sidewalk_width_meters"] = args.sidewalk_width
    if args.cycleway_width is not None:
        geom_params_cfg["cycleway_width_meters"] = args.cycleway_width
    if args.fence_width is not None:
        geom_params_cfg["fence_width_meters"] = args.fence_width
    if args.tree_radius is not None:
        geom_params_cfg["tree_point_radius_meters"] = args.tree_radius

    GEOM_PARAMS.update(geom_params_cfg)
    tree_radius = GEOM_PARAMS["tree_point_radius_meters"]
    print("OSM geometry parameters:")
    for k in (
        "road_width_meters",
        "sidewalk_width_meters",
        "cycleway_width_meters",
        "fence_width_meters",
        "tree_point_radius_meters",
    ):
        print(f"  {k}: {GEOM_PARAMS[k]:.3f}")
    keyframe_dist = args.keyframe_dist if args.keyframe_dist is not None else cfg.get("keyframe_dist", 0.0)
    max_range = cfg.get("max_range", 200.0)
    ds_resolution = cfg.get("ds_resolution", 1.0)

    labels_common_path = os.path.join(SCRIPT_DIR, "config/datasets/labels_common.yaml")
    gt_mapping = load_label_mappings(labels_common_path, gt_labels_key)
    inferred_mapping = None
    learning_map_inv = {}
    if args.use_inferred_row:
        inferred_mapping = load_label_mappings(labels_common_path, inferred_labels_key)
        labels_config_path = os.path.join(SCRIPT_DIR, f"config/datasets/labels_{inferred_labels_key}.yaml")
        if os.path.isfile(labels_config_path):
            learning_map_inv = load_learning_map_inv(labels_config_path)
        print(f"Inferred-row mode: using {inferred_label_prefix} (key={inferred_labels_key}, multiclass={inferred_use_multiclass})")

    print(f"Loading calibration from {calib_file}")
    body_to_lidar = load_calibration(calib_file)
    lidar_to_body = np.linalg.inv(body_to_lidar)

    # Accumulators shared across sequences — co-occurrence statistics are frame-invariant,
    # so per-sequence first-pose-relative transforms don't interfere when concatenated.
    all_gt, all_osm = [], []
    all_inferred = [] if args.use_inferred_row else None
    all_map_pts = [] if args.visualize_points else None
    scan_data_list = [] if not args.no_height_matrix else None

    last_geom = None
    for seq in seq_list:
        result_geom = _process_sequence_scans(
            seq, cfg, args, data_dir,
            gt_mapping=gt_mapping,
            inferred_mapping=inferred_mapping,
            learning_map_inv=learning_map_inv,
            inferred_label_prefix=inferred_label_prefix,
            inferred_use_multiclass=inferred_use_multiclass,
            body_to_lidar=body_to_lidar,
            lidar_to_body=lidar_to_body,
            decay_m=decay_m,
            keyframe_dist=keyframe_dist,
            max_range=max_range,
            ds_resolution=ds_resolution,
            tree_radius=tree_radius,
            all_gt=all_gt,
            all_osm=all_osm,
            all_inferred=all_inferred,
            scan_data_list=scan_data_list,
            all_map_pts=all_map_pts,
        )
        if result_geom is not None:
            last_geom = result_geom

    if not all_gt:
        print("\nERROR: No valid scans found across the given sequence(s); cannot derive matrix.")
        return

    all_gt = np.concatenate(all_gt)
    all_osm = np.concatenate(all_osm)
    if all_inferred is not None and all_inferred:
        all_inferred = np.concatenate(all_inferred)
    print(f"\nTotal points across {len(seq_list)} sequence(s): {len(all_gt)}")

    # `geom` is used by the visualization paths below; use the last sequence's trimmed
    # geom (per-sequence coord frames make multi-sequence viz ambiguous).
    geom = last_geom

    # Single-pass optimization: in the class-indexed height-filter scheme, the OSM→common
    # confusion matrix M and the class-indexed height matrix H decouple (H is a post-hoc
    # per-class scalar applied after M·osm_vec), so there is no benefit to alternating.
    if args.use_inferred_row and all_inferred is not None:
        counts, class_totals = build_cooccurrence_inferred(all_inferred, all_gt, all_osm)
        print("Co-occurrence: inferred-row with GT compatibility")
    else:
        counts, class_totals = build_cooccurrence(all_gt, all_osm)
    if args.shrinkage_kappa is not None:
        matrix = derive_matrix_shrinkage(counts, class_totals,
                                         kappa=args.shrinkage_kappa,
                                         class_weight_mode=args.class_weight_mode,
                                         selectivity_weight=args.selectivity_weight)
    else:
        matrix = derive_matrix(counts, class_totals,
                               class_weight_mode=args.class_weight_mode,
                               selectivity_weight=args.selectivity_weight)

    height_matrix = None
    if (
        scan_data_list is not None
        and len(scan_data_list) > 0
        and not args.no_height_matrix
    ):
        counts_h = build_cooccurrence_height_class(
            scan_data_list,
            step_meters=args.height_step_meters,
            max_meters=args.height_max_meters,
        )
        height_matrix = derive_height_matrix(counts_h)
        num_bins_used = counts_h.shape[0]
        print(f"\nOSM height confusion matrix: {num_bins_used} bins x {N_CLASSES - 1} class cols")
        print(f"  (step={args.height_step_meters}m, max height={args.height_max_meters}m, "
              f"bin 1 = h_above_bottom [0, {args.height_step_meters:.2f}] m, "
              f"bin {num_bins_used} = h_above_bottom "
              f"[{(num_bins_used-1)*args.height_step_meters:.2f}, "
              f"{num_bins_used*args.height_step_meters:.2f}] m)")

    _mode_desc = {
        "points":      "rows/prior scaled by GT point count per class (accuracy-leaning)",
        "equal":       "all class rows/priors weighted equally",
        "sqrt":        "rows/prior scaled by sqrt(GT points); compromise between accuracy and mIoU",
        "median_freq": "rows/prior weighted by median-frequency balancing (mIoU-leaning)",
    }.get(args.class_weight_mode, args.class_weight_mode)
    if args.shrinkage_kappa is not None:
        print(f"Matrix derived via Bayesian shrinkage (kappa={args.shrinkage_kappa:g}, "
              f"prior={args.class_weight_mode!r}: {_mode_desc}).")
    else:
        print(f"Matrix derived with class_weight_mode={args.class_weight_mode!r} "
              f"({_mode_desc}).")
    if args.selectivity_weight:
        print("Columns scaled by selectivity (1 - normalized entropy); "
              "weak OSM columns fade, unique OSM→class associations stay full-strength.")
    print("\nOptimized OSM confusion matrix:")
    header = "                " + "  ".join(f"{c:>7s}" for c in OSM_COLUMNS)
    print(header)
    for i in range(N_CLASSES - 1):
        vals = "  ".join(f"{matrix[i, j]:7.2f}" for j in range(N_OSM))
        print(f"  {CLASS_NAMES[i+1]:>14s}: {vals}")

    print("\nPer-class OSM coverage:")
    for cls in range(1, N_CLASSES):
        mask = all_gt == cls
        if mask.sum() == 0:
            continue
        covered = (all_osm[mask].max(axis=1) > 0).mean() * 100
        print(f"  {CLASS_NAMES[cls]:>14s}: {covered:5.1f}% of {mask.sum()} pts covered by OSM")

    geometry_params_out = dict(GEOM_PARAMS)
    geometry_params_out["osm_decay_meters"] = decay_m
    write_osm_cm_yaml(
        matrix,
        args.output,
        height_matrix=height_matrix,
        geometry_params=geometry_params_out,
        height_step_meters=args.height_step_meters,
        height_max_meters=args.height_max_meters,
    )
    print(f"\nSaved optimized matrix to {args.output}")

    if args.visualize:
        vis_path = args.vis_output
        if vis_path is None:
            base, _ = os.path.splitext(args.output)
            vis_path = base + ".png"
        plot_osm_confusion_matrix(matrix, save_path=vis_path, show=not args.no_show)

    if args.visualize_points and all_map_pts is not None:
        map_pts_xy = np.concatenate(all_map_pts)
        vis_pts_path = args.vis_points_output
        if vis_pts_path is None:
            base, _ = os.path.splitext(args.output)
            vis_pts_path = base + "_points_osm.png"
        plot_points_and_osm_heatmap(
            map_pts_xy, all_gt, geom, all_osm,
            save_path=vis_pts_path, show=not args.no_show,
            subsample=10, grid_res=args.grid_res,
        )


if __name__ == "__main__":
    main()
