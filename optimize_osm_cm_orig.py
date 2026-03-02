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
                               [--max-scans N] [--skip-frames K]
                               [--grid-res G] [--visualize] [--visualize-points]
                               [--no-show] [--vis-output VIS_PNG] [--vis-points-output VIS_PNG]
                               [--height-bins N] [--no-height-matrix]

    --visualize           Plot the optimized matrix as a heatmap (blue=suppress, red=boost).
    --visualize-points    Plot GT points + OSM geometry + OSM prior heatmap (spatial view).
    --no-show             Save PNG only, do not block to show figure.
    --height-bins N       Number of per-scan height bins for osm_height_confusion_matrix (default: 20).
    --no-height-matrix    Skip computing and writing osm_height_confusion_matrix.
    --vis-output          Path for matrix PNG (default: <output>.png).
    --vis-points-output   Path for points+OSM PNG (default: <output>_points_osm.png).
    --use-inferred-row    Use inferred (model) labels as matrix rows. Optimizes for OSM to correct inferred toward GT.

Defaults are read from config/datasets/mcd.yaml relative to this script.
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

    # Build way_id -> coords for relation resolution
    ways_coords = {}
    for way in root.iter("way"):
        coords = _way_to_coords(way, nodes, origin_lat, origin_lon)
        if coords:
            ways_coords[way.attrib["id"]] = coords

    buildings, roads, grasslands, trees_poly, forests = [], [], [], [], []
    parking, fences, stairs, tree_points, pole_points = [], [], [], [], []
    sidewalks, cycleways, walls, water = [], [], [], []

    for nd in root.iter("node"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in nd.iter("tag")}
        if tags.get("natural") == "tree":
            tree_points.append(_latlon_to_xy(float(nd.attrib["lat"]), float(nd.attrib["lon"]), origin_lat, origin_lon))
        elif tags.get("highway") == "traffic_signals":
            pole_points.append(_latlon_to_xy(float(nd.attrib["lat"]), float(nd.attrib["lon"]), origin_lat, origin_lon))
        elif tags.get("power") in ("pole", "tower"):
            pole_points.append(_latlon_to_xy(float(nd.attrib["lat"]), float(nd.attrib["lon"]), origin_lat, origin_lon))
        elif tags.get("man_made") in ("street_cabinet", "mast"):
            pole_points.append(_latlon_to_xy(float(nd.attrib["lat"]), float(nd.attrib["lon"]), origin_lat, origin_lon))

    for way in root.iter("way"):
        tags = {t.attrib["k"]: t.attrib["v"] for t in way.iter("tag")}
        coords = ways_coords.get(way.attrib["id"])
        if coords is None:
            continue
        if "building" in tags:
            if len(coords) >= 3:
                buildings.append((coords, []))  # simple polygon, no holes
            continue
        amenity = tags.get("amenity", "")
        if amenity in ("parking", "parking_space") and len(coords) >= 3:
            parking.append((coords, []))
            continue
        if tags.get("barrier") == "fence":
            fences.append(coords)
            continue
        if tags.get("barrier") == "wall" or tags.get("man_made") == "wall":
            walls.append(coords)
            continue
        hw = tags.get("highway", "")
        if hw == "steps":
            stairs.append(coords)
            continue
        if hw in SIDEWALK_HIGHWAY_TYPES:
            sidewalks.append(coords)
            continue
        if hw in CYCLEWAY_HIGHWAY:
            cycleways.append(coords)
            continue
        if hw in ROAD_HIGHWAY_TYPES:
            roads.append(coords)
            continue
        if tags.get("natural") == "water" and len(coords) >= 3 and coords[0] == coords[-1]:
            water.append((coords, []))
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
        elif tags.get("natural") == "water":
            water.extend(polys)

    return dict(buildings=buildings, roads=roads, grasslands=grasslands,
                trees=trees_poly, forests=forests, parking=parking,
                fences=fences, stairs=stairs, tree_points=tree_points,
                sidewalks=sidewalks, cycleways=cycleways, walls=walls, water=water,
                pole_points=pole_points)


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
    for cat in ("buildings", "roads", "grasslands", "trees", "forests",
                "parking", "fences", "stairs", "sidewalks", "cycleways", "walls", "water"):
        out[cat] = [g for g in geom.get(cat, []) if _bbox_intersects_expanded(g, xmin, xmax, ymin, ymax, margin)]
    for pt_key in ("tree_points", "pole_points"):
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
    "buildings", "fences", "walls", "stairs", "water", "poles", "none"
]
N_OSM = len(OSM_COLUMNS)
_POLE_RADIUS = 2.0  # meters; radius around pole/traffic-sign nodes for prior
_STAIRS_WIDTH = 1.5  # meters; half-width perpendicular to stairs polyline (matches ROS bkioctomap)

# Search radius for STRtree: geometry beyond this does not affect prior (decay_m typically 3m)
_QUERY_BUFFER = 50.0  # meters


def _build_shapely_index(geom):
    """Build Shapely geometries + STRtrees for fast spatial queries. Returns dict or None if Shapely unavailable."""
    if not SHAPELY_AVAILABLE:
        return None
    idx = {"geom": geom, "tree_points": geom.get("tree_points", []), "pole_points": geom.get("pole_points", [])}
    for cat in ("roads", "sidewalks", "cycleways", "fences", "walls", "stairs"):
        geoms = []
        for coords in geom.get(cat, []):
            if len(coords) >= 2:
                geoms.append(LineString(coords))
        idx[cat] = (STRtree(geoms), geoms) if geoms else (None, [])
    for cat in ("buildings", "grasslands", "trees", "forests", "parking", "water"):
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


def _stairs_prior_at_point(x, y, stairs_polylines, decay_m, stairs_width=_STAIRS_WIDTH):
    """Compute stairs OSM prior using width band (matches ROS bkioctomap).
    Each polyline segment is expanded into a rectangle of half-width stairs_width/2.
    Points inside get prior 1; outside get decay from signed distance to boundary.
    """
    if not stairs_polylines:
        return 0.0
    hw = max(0.05, stairs_width * 0.5)
    min_positive_d = float("inf")
    for polyline in stairs_polylines:
        coords = list(polyline.coords) if hasattr(polyline, "coords") else polyline
        if len(coords) < 2:
            continue
        for i in range(len(coords) - 1):
            x1, y1 = coords[i][0], coords[i][1]
            x2, y2 = coords[i + 1][0], coords[i + 1][1]
            dx, dy = x2 - x1, y2 - y1
            L = math.sqrt(dx * dx + dy * dy) + 1e-12
            if L < 1e-6:
                continue
            nx, ny = -dy / L, dx / L
            c1 = (x1 + hw * nx, y1 + hw * ny)
            c2 = (x1 - hw * nx, y1 - hw * ny)
            c3 = (x2 - hw * nx, y2 - hw * ny)
            c4 = (x2 + hw * nx, y2 + hw * ny)
            rect = [c1, c2, c3, c4]
            sd = distance_to_polygon_boundary(x, y, rect)
            if sd <= 0:
                return 1.0
            if sd < min_positive_d:
                min_positive_d = sd
    if min_positive_d == float("inf"):
        return 0.0
    return osm_prior_from_signed_distance(min_positive_d, decay_m)


def _compute_single_prior_shapely(x, y, idx, cat, decay_m, tree_radius):
    """Fast prior using Shapely + STRtree."""
    pt = Point(x, y)

    def _prior_dist(d):
        return osm_prior_from_distance(d, decay_m)
    def _prior_signed(sd):
        return osm_prior_from_signed_distance(sd, decay_m)

    if cat == "roads":
        candidates = _query_candidates(idx, "roads", x, y)
        min_d = float("inf")
        for g in candidates:
            d = g.distance(pt)
            if d < min_d:
                min_d = d
        return _prior_dist(min_d) if min_d != float("inf") else 0.0

    elif cat == "sidewalks":
        candidates = _query_candidates(idx, "sidewalks", x, y)
        min_d = float("inf")
        for g in candidates:
            d = g.distance(pt)
            if d < min_d:
                min_d = d
        return _prior_dist(min_d) if min_d != float("inf") else 0.0

    elif cat == "cycleways":
        candidates = _query_candidates(idx, "cycleways", x, y)
        min_d = float("inf")
        for g in candidates:
            d = g.distance(pt)
            if d < min_d:
                min_d = d
        return _prior_dist(min_d) if min_d != float("inf") else 0.0

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
            d = g.distance(pt)
            if d < min_d:
                min_d = d
        return _prior_dist(min_d) if min_d != float("inf") else 0.0

    elif cat == "stairs":
        candidates = _query_candidates(idx, "stairs", x, y)
        return _stairs_prior_at_point(x, y, candidates, decay_m)

    elif cat == "walls":
        min_d = float("inf")
        for g in _query_candidates(idx, "walls", x, y):
            d = g.distance(pt)
            if d < min_d:
                min_d = d
        return _prior_dist(min_d) if min_d != float("inf") else 0.0

    elif cat == "water":
        min_d = float("inf")
        for g in _query_candidates(idx, "water", x, y):
            sd, ok = _shapely_signed_distance(g, pt)
            if not ok:
                continue
            if sd <= 0:
                return 1.0
            if sd < min_d:
                min_d = sd
        return _prior_signed(min_d) if min_d != float("inf") else 0.0

    elif cat == "poles":
        max_p = 0.0
        for cx, cy in idx.get("pole_points", []):
            d_center = math.sqrt((x - cx) ** 2 + (y - cy) ** 2)
            sd = d_center - _POLE_RADIUS
            p = _prior_signed(sd)
            if p > max_p:
                max_p = p
        return max_p

    return 0.0


def _compute_single_prior(x, y, geom, cat, decay_m, tree_radius):
    """Compute OSM prior for one category at (x,y)."""
    def _line_prior(lines_key):
        min_d = float("inf")
        for line in geom.get(lines_key, []):
            d = distance_to_polyline(x, y, line)
            if d < min_d:
                min_d = d
        return osm_prior_from_distance(min_d, decay_m)
    if cat == "roads":
        return _line_prior("roads")
    elif cat == "sidewalks":
        return _line_prior("sidewalks")
    elif cat == "cycleways":
        return _line_prior("cycleways")
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
        min_d = float("inf")
        for poly in geom["fences"]:
            d = distance_to_polyline(x, y, poly)
            if d < min_d: min_d = d
        return osm_prior_from_distance(min_d, decay_m)
    elif cat == "stairs":
        return _stairs_prior_at_point(x, y, geom.get("stairs", []), decay_m)
    elif cat == "walls":
        return _line_prior("walls")
    elif cat == "water":
        min_d = float("inf")
        for poly in geom.get("water", []):
            sd = distance_to_polygon_boundary(x, y, poly)
            if sd <= 0:
                return 1.0
            if sd < min_d:
                min_d = sd
        return osm_prior_from_signed_distance(min_d, decay_m)
    elif cat == "poles":
        max_p = 0.0
        for cx, cy in geom.get("pole_points", []):
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2) - _POLE_RADIUS
            p = osm_prior_from_signed_distance(d, decay_m)
            if p > max_p:
                max_p = p
        return max_p
    return 0.0


def _compute_cell_prior(cx, cy, geom, idx_shapely, decay_m, tree_radius, cats):
    """Compute OSM prior vector for one grid cell. Uses Shapely if available."""
    n_cats = len(cats)
    v = np.zeros(N_OSM, dtype=np.float32)
    if idx_shapely is not None:
        for ci, cat in enumerate(cats):
            v[ci] = _compute_single_prior_shapely(cx, cy, idx_shapely, cat, decay_m, tree_radius)
    else:
        for ci, cat in enumerate(cats):
            v[ci] = _compute_single_prior(cx, cy, geom, cat, decay_m, tree_radius)
    v[n_cats] = max(0.0, 1.0 - v[:n_cats].max())
    return v


def precompute_osm_grid(geom, xmin, xmax, ymin, ymax, margin, grid_res, decay_m, tree_radius):
    """Precompute OSM prior for all grid cells in trajectory bbox + margin. Done once before scan loop."""
    gx_min = int(np.floor((xmin - margin) / grid_res))
    gx_max = int(np.floor((xmax + margin) / grid_res))
    gy_min = int(np.floor((ymin - margin) / grid_res))
    gy_max = int(np.floor((ymax + margin) / grid_res))
    idx_shapely = _build_shapely_index(geom)
    cats = [c for c in OSM_COLUMNS if c != "none"]
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
    cats = [c for c in OSM_COLUMNS if c != "none"]

    if precomputed_grid is not None:
        # Fast path: lookup only (rare fallback for points outside bbox)
        for i in range(n):
            k = (int(keys[i, 0]), int(keys[i, 1]))
            if k in precomputed_grid:
                osm_vecs[i] = precomputed_grid[k]
            else:
                cx = (k[0] + 0.5) * grid_res
                cy = (k[1] + 0.5) * grid_res
                v = _compute_cell_prior(cx, cy, geom, osm_index, decay_m, tree_radius, cats)
                precomputed_grid[k] = v  # cache for reuse
                osm_vecs[i] = v
        return osm_vecs

    # Legacy path: per-scan cache (when no precomputed grid)
    cache = {}
    idx = osm_index if osm_index is not None else _build_shapely_index(geom)
    for i in range(n):
        k = (int(keys[i, 0]), int(keys[i, 1]))
        if k in cache:
            osm_vecs[i] = cache[k]
        else:
            cx = (k[0] + 0.5) * grid_res
            cy = (k[1] + 0.5) * grid_res
            v = _compute_cell_prior(cx, cy, geom, idx, decay_m, tree_radius, cats)
            cache[k] = v
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

N_CLASSES = 13
CLASS_NAMES = [
    "unlabeled", "road", "sidewalk", "parking", "other-ground",
    "building", "fence", "pole", "traffic-sign", "vegetation",
    "two-wheeler", "vehicle", "other-object",
]


# OSM column index -> set of GT common classes compatible with that OSM category.
# Used for inferred-row: only count OSM evidence when GT aligns (to learn correction).
OSM_GT_COMPATIBLE = {
    0: {1, 2},           # roads -> road, sidewalk
    1: {2},              # sidewalks -> sidewalk
    2: {10},             # cycleways -> two-wheeler
    3: {3},              # parking -> parking
    4: {4, 9},           # grasslands -> other-ground, vegetation
    5: {9},              # trees -> vegetation
    6: {9},              # forest -> vegetation
    7: {5},              # buildings -> building
    8: {6},              # fences -> fence
    9: {6, 4},           # walls -> fence, other-ground (retaining walls)
    10: {4},             # stairs -> other-ground
    11: {4},             # water -> other-ground
    12: {7, 8},          # poles -> pole, traffic-sign
    13: set(range(13)),  # none -> all (neutral)
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


def derive_matrix(counts, class_totals, positive_only=False):
    """Derive 12x9 matrix from co-occurrence (rows = classes 1..12).

    If positive_only=True, values are in [0, 1] (boost only, no suppression).
    Otherwise values are in [-1, 1] (positive=boost, negative=suppress).
    """
    total_points = class_totals.sum()
    if total_points == 0:
        return np.zeros((12, N_OSM))
    col_totals = counts.sum(axis=0)
    matrix = np.zeros((12, N_OSM))
    for ri, cls in enumerate(range(1, 13)):
        p_cls = class_totals[cls] / total_points
        for j in range(N_OSM):
            if col_totals[j] < 1.0 or p_cls < 1e-8:
                matrix[ri][j] = 0.0
            else:
                p_cls_given_osm = counts[cls][j] / col_totals[j]
                raw = np.clip(p_cls_given_osm / p_cls - 1.0, -1.0, 1.0)
                if positive_only:
                    # Rescale [-1, 1] -> [0, 1]: positive = boost, negative -> 0 (no suppression)
                    matrix[ri][j] = max(0.0, raw)
                else:
                    matrix[ri][j] = raw
    return matrix


def build_cooccurrence_height(scan_data_list, num_bins=20):
    """Per-scan height bins: for each (bin, OSM_col), accumulate weighted counts.
    scan_data_list: list of (map_pts, gt_common, osm_vecs). map_pts has (n, 3) with x,y,z.
    counts_h[bin][col] = sum of osm_vec[col] where GT is compatible with OSM col.
    totals_h[bin][col] = sum of osm_vec[col] over all points in bin (with prior > threshold).
    """
    counts_h = np.zeros((num_bins, N_OSM), dtype=np.float64)
    totals_h = np.zeros((num_bins, N_OSM), dtype=np.float64)
    osm_thresh = 0.01
    for map_pts, gt, osm in scan_data_list:
        z = map_pts[:, 2]
        min_z = z.min()
        max_z = z.max()
        z_range = max_z - min_z + 1e-6
        bins = np.clip(np.floor((z - min_z) / z_range * num_bins).astype(np.int32), 0, num_bins - 1)
        for i in range(len(gt)):
            b = int(bins[i])
            for j in range(N_OSM):
                if osm[i, j] < osm_thresh:
                    continue
                totals_h[b, j] += osm[i, j]
                if int(gt[i]) in OSM_GT_COMPATIBLE.get(j, set()):
                    counts_h[b, j] += osm[i, j]
    return counts_h, totals_h


def derive_height_matrix(counts_h, totals_h, num_bins=20, class_totals=None, total_points=None):
    """Derive height confusion matrix: matrix[bin][col] = trust multiplier in [0, 1].
    When totals_h[bin][col] is negligible, use 1.0 (neutral).
    If class_totals and total_points are provided, applies Bayesian smoothing toward
    class-frequency prior: prior_precision[j] = P(compatible | col j) from class marginals.
    """
    matrix = np.zeros((num_bins, N_OSM))
    prior_precision = None
    if class_totals is not None and total_points is not None and total_points > 0:
        prior_precision = np.zeros(N_OSM, dtype=np.float64)
        for j in range(N_OSM):
            compat = OSM_GT_COMPATIBLE.get(j, set())
            prior_precision[j] = sum(class_totals[c] for c in compat if 0 <= c < len(class_totals)) / total_points
        prior_precision = np.clip(prior_precision, 1e-6, 1.0 - 1e-6)
    k = 10.0 if prior_precision is not None else 0.0
    for b in range(num_bins):
        for j in range(N_OSM):
            if totals_h[b, j] < 1e-6:
                matrix[b, j] = prior_precision[j] if prior_precision is not None else 1.0
            else:
                raw = counts_h[b, j] / totals_h[b, j]
                if prior_precision is not None:
                    smoothed = (counts_h[b, j] + k * prior_precision[j]) / (totals_h[b, j] + k)
                    matrix[b, j] = np.clip(smoothed, 0.0, 1.0)
                else:
                    matrix[b, j] = np.clip(raw, 0.0, 1.0)
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
            heatmap[iy, ix] = max(heatmap[iy, ix], osm[i, :-1].max())  # exclude 'none'
    extent = (xmin, xmax, ymin, ymax)
    ax1.imshow(heatmap, origin="lower", extent=extent, aspect="auto",
               cmap="Greens", alpha=0.5, vmin=0, vmax=1)

    # OSM geometry outlines (use outer ring for polygons with holes)
    for poly in geom["buildings"][:200]:  # limit for speed
        ring = _poly_outer(poly)
        if len(ring) >= 3:
            p = Polygon(ring, fill=False, edgecolor="brown", linewidth=0.5, alpha=0.7)
            ax1.add_patch(p)
    for road in geom["roads"][:100]:
        if len(road) >= 2:
            ax1.plot([r[0] for r in road], [r[1] for r in road], "k-", linewidth=0.8, alpha=0.5)
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
        if len(poly) >= 2:
            ax2.plot([r[0] for r in poly], [r[1] for r in poly], "gray", linewidth=0.6, alpha=0.8)
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
    ax.set_yticks(range(12))
    ax.set_yticklabels([CLASS_NAMES[i + 1] for i in range(12)])

    # Add value annotations
    for i in range(12):
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

def write_osm_cm_yaml(matrix, output_path, positive_only=False, height_matrix=None):
    if positive_only:
        val_comment = "# Values in [0.0, 1.0]: boost only (no suppression)."
    else:
        val_comment = "# Values in [-1.0, 1.0]: positive = boost, negative = suppress, 0 = neutral."
    cols_str = ", ".join(OSM_COLUMNS)
    lines = [
        "# Optimized OSM confusion matrix (derived from GT co-occurrence analysis).",
        "# Rows  = common taxonomy semantic classes (1-12).",
        f"# Cols  = OSM prior categories: [{cols_str}]",
        val_comment,
        "#",
        "# Common taxonomy:",
        "#   1: road, 2: sidewalk, 3: parking, 4: other-ground, 5: building,",
        "#   6: fence, 7: pole, 8: traffic-sign, 9: vegetation, 10: two-wheeler,",
        "#   11: vehicle, 12: other-object",
        "",
        f"osm_prior_columns: [{cols_str}]",
        "",
        "confusion_matrix:",
    ]
    for i in range(12):
        vals = ", ".join(f"{matrix[i, j]:6.2f}" for j in range(N_OSM))
        lines.append(f"  {i+1}:  [{vals}]   # {CLASS_NAMES[i+1]}")
    if height_matrix is not None:
        n_bins = height_matrix.shape[0]
        lines += [
            "",
            "# OSM height confusion matrix (per-scan relative bins). Rows = height bins (1=lowest, "
            f"{n_bins}=highest). Cols = OSM categories. Multipliers in [0, 1].",
            "osm_height_confusion_matrix:",
        ]
        for b in range(n_bins):
            vals = ", ".join(f"{height_matrix[b, j]:6.2f}" for j in range(N_OSM))
            lines.append(f"  {b+1}:  [{vals}]   # BIN {b+1} Height")
    lines += [
        "", "osm_class_map:",
        *[f"  {name}: {j}" for j, name in enumerate(OSM_COLUMNS)],
        "", "label_to_matrix_idx:",
        *[f"  {i+1}: {i}" for i in range(12)],
        "",
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


# ═══════════════════════════════════════════════════════════════════
# 8. Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Optimize OSM confusion matrix from GT data.")
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config/datasets/mcd.yaml"))
    parser.add_argument("--output", default=os.path.join(SCRIPT_DIR, "config/datasets/osm_confusion_matrix_optimized.yaml"))
    parser.add_argument("--max-scans", type=int, default=10000)
    parser.add_argument("--skip-frames", type=int, default=20)
    parser.add_argument("--decay", type=float, default=3.0)
    parser.add_argument("--tree-radius", type=float, default=3.0)
    parser.add_argument("--grid-res", type=float, default=2.0,
                        help="Grid resolution (m) for OSM prior caching (default: 2.0)")
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
    parser.add_argument("--use-inferred-row", action="store_true",
                        help="Use inferred (model) labels as matrix rows. Optimizes OSM to correct inferred toward GT.")
    parser.add_argument("--inferred-prefix", type=str, default=None,
                        help="Override input_label_prefix (e.g. kth_day_09/inferred_labels/cenet_mcd_EDL/multiclass_confidence_scores)")
    parser.add_argument("--inferred-key", type=str, default=None,
                        help="Override inferred_labels_key (mcd or semkitti) for label mapping")
    parser.add_argument("--positive-only", action="store_true",
                        help="Output matrix with values in [0, 1] only (boost, no suppression)")
    parser.add_argument("--height-bins", type=int, default=20,
                        help="Number of per-scan height bins for osm_height_confusion_matrix (default: 20)")
    parser.add_argument("--no-height-matrix", action="store_true",
                        help="Skip computing and writing osm_height_confusion_matrix")
    args = parser.parse_args()

    print(f"Loading config from {args.config}")
    with open(args.config) as f:
        raw_cfg = yaml.safe_load(f)
    cfg = raw_cfg
    if "/**" in cfg:
        cfg = cfg["/**"].get("ros__parameters", cfg)
    elif "ros__parameters" in cfg:
        cfg = cfg["ros__parameters"]

    data_dir = os.path.join(SCRIPT_DIR, "data", "mcd")
    pose_file = os.path.join(data_dir, cfg["lidar_pose_file"])
    gt_label_dir = os.path.join(data_dir, cfg.get("gt_label_prefix", "kth_day_09/gt_labels"))
    gt_labels_key = cfg.get("gt_labels_key", "mcd")
    scan_dir = os.path.join(data_dir, cfg.get("input_data_prefix", "kth_day_09/lidar_bin/data"))
    inferred_label_prefix = args.inferred_prefix if args.inferred_prefix else cfg.get(
        "input_label_prefix", "kth_day_09/inferred_labels/cenet_semkitti/multiclass_confidence_scores"
    )
    inferred_labels_key = args.inferred_key if args.inferred_key else cfg.get("inferred_labels_key", "semkitti")
    inferred_use_multiclass = cfg.get("inferred_use_multiclass", True)
    calib_file = os.path.join(data_dir, "hhs_calib.yaml")
    osm_file = os.path.join(data_dir, cfg.get("osm_file", "kth_large.osm"))
    origin_lat = cfg.get("osm_origin_lat", 0.0)
    origin_lon = cfg.get("osm_origin_lon", 0.0)
    decay_m = args.decay if args.decay is not None else cfg.get("osm_decay_meters", 3.0)
    tree_radius = args.tree_radius if args.tree_radius is not None else cfg.get("osm_tree_point_radius_meters", 4.0)
    skip_frames = args.skip_frames if args.skip_frames is not None else cfg.get("skip_frames", 0)
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
        inferred_label_dir = os.path.join(data_dir, inferred_label_prefix)
        print(f"Inferred-row mode: using {inferred_label_prefix} (key={inferred_labels_key}, multiclass={inferred_use_multiclass})")

    print(f"Loading poses from {pose_file}")
    poses = load_poses(pose_file)
    print(f"  Loaded {len(poses)} poses")

    first_pose = poses[0][1].copy()
    first_inv = np.linalg.inv(first_pose)
    for i in range(len(poses)):
        poses[i] = (poses[i][0], first_inv @ poses[i][1])

    print(f"Loading calibration from {calib_file}")
    body_to_lidar = load_calibration(calib_file)
    lidar_to_body = np.linalg.inv(body_to_lidar)

    print(f"Parsing OSM geometry from {osm_file}")
    geom = parse_osm_xml(osm_file, origin_lat, origin_lon)
    for cat in geom:
        print(f"  {cat}: {len(geom[cat])} items")

    # Transform OSM coords to first-pose-relative frame
    def _transform_ring(ring):
        for k in range(len(ring)):
            x, y = ring[k]
            p = first_inv @ np.array([x, y, 0, 1])
            ring[k] = (float(p[0]), float(p[1]))
    for cat in ("buildings", "roads", "grasslands", "trees", "forests",
                "parking", "fences", "stairs", "sidewalks", "cycleways", "walls", "water"):
        if cat not in geom:
            continue
        for item in geom[cat]:
            if _poly_holes(item):  # polygon with holes: (outer, [hole1, hole2, ...])
                outer, holes = _poly_outer(item), _poly_holes(item)
                _transform_ring(outer)
                for h in holes:
                    _transform_ring(h)
            else:  # simple polyline or polygon
                ring = _poly_outer(item)
                _transform_ring(ring)
    for pt_key in ("tree_points", "pole_points"):
        if pt_key not in geom:
            continue
        for k in range(len(geom[pt_key])):
            x, y = geom[pt_key][k]
            p = first_inv @ np.array([x, y, 0, 1])
            geom[pt_key][k] = (float(p[0]), float(p[1]))

    # Select scans
    valid = 0
    scan_list = []
    inferred_label_dir = os.path.join(data_dir, inferred_label_prefix) if args.use_inferred_row else None
    for idx, T in poses:
        sp = os.path.join(scan_dir, f"{idx:010d}.bin")
        lp = os.path.join(gt_label_dir, f"{idx:010d}.bin")
        if not (os.path.isfile(sp) and os.path.isfile(lp)):
            continue
        if args.use_inferred_row:
            ip = os.path.join(inferred_label_dir, f"{idx:010d}.bin")
            if not os.path.isfile(ip):
                continue
        else:
            ip = None
        if skip_frames > 0 and valid % (skip_frames + 1) != 0:
            valid += 1
            continue
        scan_list.append((idx, T, sp, lp, ip))
        valid += 1
        if len(scan_list) >= args.max_scans:
            break

    # Trim OSM geometry to trajectory bbox to speed up prior computation
    xs = [T[0, 3] for _, T, _, _, _ in scan_list]
    ys = [T[1, 3] for _, T, _, _, _ in scan_list]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    margin = max_range + decay_m
    geom_trimmed = trim_osm_to_bbox(geom, xmin, xmax, ymin, ymax, margin)
    n_before = sum(len(geom[c]) for c in geom)
    n_after = sum(len(geom_trimmed[c]) for c in geom_trimmed)
    print(f"OSM trim: kept {n_after}/{n_before} items in bbox [x:{xmin:.0f}..{xmax:.0f}, y:{ymin:.0f}..{ymax:.0f}] margin={margin:.0f}m")
    geom = geom_trimmed

    print("Precomputing OSM prior grid (once for all scans)...")
    precomputed_grid, osm_index = precompute_osm_grid(
        geom, xmin, xmax, ymin, ymax, margin, args.grid_res, decay_m, tree_radius
    )
    print(f"  Grid has {len(precomputed_grid)} cells (Shapely={'on' if osm_index else 'off'})")

    print(f"Processing {len(scan_list)} scans (skip={skip_frames}, max={args.max_scans}, grid_res={args.grid_res}m)")

    all_gt, all_osm = [], []
    all_inferred = [] if args.use_inferred_row else None
    all_map_pts = [] if args.visualize_points else None
    scan_data_list = [] if not args.no_height_matrix else None

    for si, (idx, T, scan_path, label_path, inferred_path) in enumerate(tqdm(scan_list, desc="scans", unit="scan")):
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
                inferred_common = np.array([inferred_mapping.get(int(l), 0) for l in inf_raw], dtype=np.int32)
            if inferred_common is None:
                inferred_common = gt_common.copy()
        else:
            inferred_common = None

        lidar_to_map = T @ lidar_to_body
        xyz_h = np.hstack([pts[:, :3], np.ones((n, 1), dtype=np.float32)])
        map_pts = (lidar_to_map @ xyz_h.T).T[:, :3]

        # Voxel downsample
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
            scan_data_list.append((map_pts.copy(), gt_common.copy(), osm_vecs.copy()))

    all_gt = np.concatenate(all_gt)
    all_osm = np.concatenate(all_osm)
    if all_inferred is not None:
        all_inferred = np.concatenate(all_inferred)
    print(f"\nTotal points: {len(all_gt)}")

    if args.use_inferred_row and all_inferred is not None:
        counts, class_totals = build_cooccurrence_inferred(all_inferred, all_gt, all_osm)
        print("Co-occurrence: inferred-row with GT compatibility")
    else:
        counts, class_totals = build_cooccurrence(all_gt, all_osm)
    matrix = derive_matrix(counts, class_totals, positive_only=args.positive_only)

    height_matrix = None
    if scan_data_list is not None and len(scan_data_list) > 0:
        counts_h, totals_h = build_cooccurrence_height(scan_data_list, num_bins=args.height_bins)
        total_points = class_totals.sum()
        height_matrix = derive_height_matrix(
            counts_h, totals_h, num_bins=args.height_bins,
            class_totals=class_totals, total_points=total_points,
        )
        print(f"\nOSM height confusion matrix: {args.height_bins} bins x {N_OSM} cols")
        print("  (bin 1 = lowest z in scan, bin %d = highest)" % args.height_bins)

    print("\nOptimized OSM confusion matrix:")
    header = "                " + "  ".join(f"{c:>7s}" for c in OSM_COLUMNS)
    print(header)
    for i in range(12):
        vals = "  ".join(f"{matrix[i, j]:7.2f}" for j in range(N_OSM))
        print(f"  {CLASS_NAMES[i+1]:>14s}: {vals}")

    print("\nPer-class OSM coverage:")
    for cls in range(1, N_CLASSES):
        mask = all_gt == cls
        if mask.sum() == 0:
            continue
        covered = (all_osm[mask, :-1].max(axis=1) > 0).mean() * 100  # exclude 'none'
        print(f"  {CLASS_NAMES[cls]:>14s}: {covered:5.1f}% of {mask.sum()} pts covered by OSM")

    write_osm_cm_yaml(matrix, args.output, positive_only=args.positive_only, height_matrix=height_matrix)
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
