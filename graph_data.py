"""
graph_data.py — load a local OpenStreetMap XML extract into a directed,
weighted road graph, with optional geometry-preserving simplification and an
on-disk cache.

Design notes
------------
* We parse the .osm XML ourselves with ``xml.etree.ElementTree.iterparse`` so
  the project works fully offline with only the Python standard library for the
  loading step (no osmnx / geopandas / internet needed).
* Edge length is the sum of haversine distances along the road's *geometry*,
  not the straight line between the two endpoints.
* One-way handling honours ``oneway=yes/true/1/-1/reverse`` plus the OSM
  convention that motorways / *_link ramps are one-way by default.
* Simplification merges degree-2 "shape" chains into a single edge while
  preserving connectivity, direction, geometry and accumulated distance.
* Parallel edges between the same pair of nodes are preserved; every edge gets
  a unique integer id (``eid``) so a route can name the exact edges it used.
"""

from __future__ import annotations

import heapq
import math
import os
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from xml.etree.ElementTree import iterparse

import config

Coord = Tuple[float, float]  # (lat, lon)

EARTH_RADIUS_KM = 6371.0088

# Bump whenever graph construction changes, so stale caches are never reused.
GRAPH_FORMAT_VERSION = 5


def haversine_km(a: Coord, b: Coord) -> float:
    """Great-circle distance in kilometres between two (lat, lon) points."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(h)))


def polyline_length_km(coords: List[Coord]) -> float:
    """Length of a polyline (list of coords) summed segment by segment."""
    return sum(haversine_km(coords[i], coords[i + 1]) for i in range(len(coords) - 1))


@dataclass
class Edge:
    """A single directed road edge."""
    eid: int                       # unique id, stable within one graph
    u: int                         # from node id
    v: int                         # to node id
    length_km: float               # geometry length, not straight line
    way_id: int                    # originating OSM way id
    highway: str                   # road class, e.g. "primary"
    name: str                      # road name (may be a fallback label)
    geometry: List[Coord]          # ordered coords from u .. v (>= 2 points)
    reversed_dir: bool = False     # True if this edge runs against OSM node order


@dataclass
class RoadGraph:
    """Directed, weighted road graph."""
    nodes: Dict[int, Coord] = field(default_factory=dict)          # node id -> (lat, lon)
    adj: Dict[int, List[Edge]] = field(default_factory=lambda: defaultdict(list))
    edges: Dict[int, Edge] = field(default_factory=dict)          # eid -> Edge
    meta: dict = field(default_factory=dict)                       # stats / provenance

    # -- basic queries ---------------------------------------------------- #
    def neighbors(self, u: int) -> List[Edge]:
        return self.adj.get(u, [])

    def num_nodes(self) -> int:
        return len(self.nodes)

    def num_edges(self) -> int:
        return len(self.edges)

    def coord(self, n: int) -> Coord:
        return self.nodes[n]

    def node_label(self, n: int) -> str:
        """Human-friendly label for a node: a named road touching it, else id."""
        for e in self.adj.get(n, []):
            if e.name and not e.name.startswith("("):
                return f"{e.name} [{n}]"
        return f"node {n}"

    # -- connectivity ----------------------------------------------------- #
    def weakly_connected_components(self) -> List[set]:
        """Undirected connected components over the directed edge set."""
        und = defaultdict(set)
        for e in self.edges.values():
            und[e.u].add(e.v)
            und[e.v].add(e.u)
        for n in self.nodes:
            und.setdefault(n, set())
        seen: set = set()
        comps: List[set] = []
        for start in self.nodes:
            if start in seen:
                continue
            stack = [start]
            comp = set()
            seen.add(start)
            while stack:
                x = stack.pop()
                comp.add(x)
                for y in und[x]:
                    if y not in seen:
                        seen.add(y)
                        stack.append(y)
            comps.append(comp)
        return comps


# --------------------------------------------------------------------------- #
# OSM parsing
# --------------------------------------------------------------------------- #

_TRUE_ONEWAY = {"yes", "true", "1"}
_REVERSE_ONEWAY = {"-1", "reverse"}


def _is_oneway(tags: Dict[str, str]) -> int:
    """Return +1 (forward only), -1 (reverse only) or 0 (two-way)."""
    val = tags.get("oneway", "").strip().lower()
    if val in _TRUE_ONEWAY:
        return 1
    if val in _REVERSE_ONEWAY:
        return -1
    if val == "no":
        return 0
    # Default-oneway conventions.
    if tags.get("junction", "").lower() in {"roundabout", "circular"}:
        return 1
    if tags.get("highway", "") in config.DEFAULT_ONEWAY_HIGHWAYS:
        return 1
    return 0


# Most specific tag wins: motorcar > motor_vehicle > vehicle > access.
_ACCESS_KEYS = ("motorcar", "motor_vehicle", "vehicle", "access")
_BLOCKED_ACCESS = {"no", "private", "agricultural", "forestry", "delivery"}


def _car_allowed(tags: Dict[str, str]) -> bool:
    """True unless the most specific access tag forbids private cars."""
    for key in _ACCESS_KEYS:
        if key in tags:
            return tags[key].strip().lower() not in _BLOCKED_ACCESS
    return True


def _road_name(tags: Dict[str, str], way_id: int) -> str:
    """Prefer English name, then local name, else a non-inventive fallback."""
    for key in ("name:en", "name", "ref"):
        if tags.get(key):
            return tags[key]
    return f"(unnamed {tags.get('highway', 'road')} {way_id})"


def parse_osm(path: str) -> Tuple[Dict[int, Coord], List[dict], List[Tuple[str, float, float]]]:
    """
    Stream-parse an .osm file.

    Returns (node_coords, ways, named_points).  Each way is a dict with keys
    ``id``, ``nodes`` (ref list) and ``tags``.  ``named_points`` lists
    (name, lat, lon) for OSM nodes that carry a name tag (junctions, squares),
    used only for the searchable place list.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"OSM file not found: {path}\n"
            f"Pass --osm /path/to/file.osm or place the file at the default location."
        )

    node_coords: Dict[int, Coord] = {}
    ways: List[dict] = []
    named_points: List[Tuple[str, float, float]] = []

    cur_way: Optional[dict] = None
    cur_node_tags: Optional[Dict[str, str]] = None
    # iterparse with 'end' events; clear elements to keep memory flat.
    for event, elem in iterparse(path, events=("start", "end")):
        if event == "start":
            if elem.tag == "way":
                cur_way = {"id": int(elem.get("id")), "nodes": [], "tags": {}}
            elif elem.tag == "node":
                cur_node_tags = {}
            continue

        # event == "end"
        if elem.tag == "node":
            try:
                coord = (float(elem.get("lat")), float(elem.get("lon")))
                node_coords[int(elem.get("id"))] = coord
                name = (cur_node_tags or {}).get("name:en") or (cur_node_tags or {}).get("name")
                if name:
                    named_points.append((name, coord[0], coord[1]))
            except (TypeError, ValueError):
                pass
            cur_node_tags = None
            elem.clear()
        elif elem.tag == "tag" and cur_way is None and cur_node_tags is not None:
            cur_node_tags[elem.get("k")] = elem.get("v")
        elif elem.tag == "nd" and cur_way is not None:
            cur_way["nodes"].append(int(elem.get("ref")))
        elif elem.tag == "tag" and cur_way is not None:
            cur_way["tags"][elem.get("k")] = elem.get("v")
        elif elem.tag == "way":
            if cur_way is not None:
                ways.append(cur_way)
                cur_way = None
            elem.clear()

    return node_coords, ways, named_points


def build_graph(path: str, simplify: bool = True,
                use_largest_component: bool = True,
                verbose: bool = True) -> RoadGraph:
    """Parse OSM and build a directed weighted graph (optionally simplified)."""
    t0 = time.perf_counter()
    node_coords, ways, named_points = parse_osm(path)
    if verbose:
        print(f"[graph] parsed {len(node_coords):,} nodes and {len(ways):,} ways "
              f"in {time.perf_counter() - t0:.1f}s")

    g = RoadGraph()
    eid = 0
    kept_ways = 0
    excluded_ways = 0
    closed_ways = 0
    used_nodes: set = set()
    alt_names: Dict[int, str] = {}   # way id -> local-language name (e.g. Bangla)

    # Raw directed graph: one edge per consecutive node pair along each way.
    for w in ways:
        tags = w["tags"]
        hw = tags.get("highway")
        if hw is None:
            excluded_ways += 1
            continue
        if hw in config.CLOSED_HIGHWAYS:
            closed_ways += 1
            continue
        if hw not in config.DRIVABLE_HIGHWAYS:
            excluded_ways += 1
            continue
        if not _car_allowed(tags):
            excluded_ways += 1
            continue

        refs = [r for r in w["nodes"] if r in node_coords]
        if len(refs) < 2:
            continue

        kept_ways += 1
        name = _road_name(tags, w["id"])
        if tags.get("name") and tags["name"] != name:
            alt_names[w["id"]] = tags["name"]
        direction = _is_oneway(tags)

        for i in range(len(refs) - 1):
            a, b = refs[i], refs[i + 1]
            if a == b:
                continue
            ca, cb = node_coords[a], node_coords[b]
            # Distinct nodes at identical coordinates give a 0 km segment; keep
            # it, or the two nodes would be disconnected.
            seg_len = haversine_km(ca, cb)

            def add(u, v, cu, cv, rev):
                nonlocal eid
                e = Edge(eid=eid, u=u, v=v, length_km=seg_len, way_id=w["id"],
                         highway=hw, name=name, geometry=[cu, cv], reversed_dir=rev)
                g.edges[eid] = e
                g.adj[u].append(e)
                used_nodes.update((u, v))
                eid += 1

            if direction >= 0:            # forward allowed (two-way or oneway=yes)
                add(a, b, ca, cb, False)
            if direction <= 0:            # backward allowed (two-way or oneway=-1)
                add(b, a, cb, ca, True)

    for n in used_nodes:
        g.nodes[n] = node_coords[n]

    g.meta.update({
        "source_file": path,
        "named_points": named_points,
        "alt_names": alt_names,
        "raw_nodes": len(g.nodes),
        "raw_edges": len(g.edges),
        "kept_ways": kept_ways,
        "excluded_ways": excluded_ways,
        "closed_construction_ways": closed_ways,
    })
    if verbose:
        print(f"[graph] drivable raw graph: {g.num_nodes():,} nodes, "
              f"{g.num_edges():,} directed edges "
              f"(kept {kept_ways:,} ways, excluded {excluded_ways:,}, "
              f"construction-closed {closed_ways:,})")

    if simplify:
        g = simplify_graph(g, verbose=verbose)

    if use_largest_component:
        g = keep_largest_component(g, verbose=verbose)

    g.meta["build_seconds"] = round(time.perf_counter() - t0, 2)
    return g


# --------------------------------------------------------------------------- #
# Simplification: collapse degree-2 shape chains
# --------------------------------------------------------------------------- #

def _degree_maps(g: RoadGraph):
    """Return (out_deg, in_deg, undirected_neighbors)."""
    out_deg = defaultdict(int)
    in_deg = defaultdict(int)
    und = defaultdict(set)
    for e in g.edges.values():
        out_deg[e.u] += 1
        in_deg[e.v] += 1
        und[e.u].add(e.v)
        und[e.v].add(e.u)
    return out_deg, in_deg, und


def simplify_graph(g: RoadGraph, verbose: bool = True) -> RoadGraph:
    """
    Merge chains of pure shape points into single edges.

    A node is kept (a "real" node) unless ALL of these hold:
      * it has exactly two distinct undirected neighbours,
      * every edge touching it belongs to the same OSM way (so road names and
        classes stay exact), and
      * its traffic flows straight through: in=out=1 (one-way) or in=out=2
        (two-way).
    Each directed chain between real nodes becomes one edge whose geometry and
    length are the concatenation of the original segments.  Because interior
    nodes have exactly one way forward that isn't a U-turn, every original edge
    is absorbed exactly once, so total length and connectivity are preserved.
    """
    before_nodes, before_edges = g.num_nodes(), g.num_edges()
    out_deg, in_deg, und = _degree_maps(g)
    ways_at: Dict[int, set] = defaultdict(set)
    for e in g.edges.values():
        ways_at[e.u].add(e.way_id)
        ways_at[e.v].add(e.way_id)

    def is_interior(n: int) -> bool:
        if len(und[n]) != 2 or len(ways_at[n]) != 1:
            return False
        return (in_deg[n], out_deg[n]) in {(1, 1), (2, 2)}

    real_nodes = {n for n in g.nodes if not is_interior(n)}

    new = RoadGraph()
    new.meta = dict(g.meta)
    consumed: set = set()  # eids already absorbed into a simplified edge
    next_eid = 0
    dropped_loops = 0
    dropped_loop_km = 0.0

    def walk(start: int, e0: Edge) -> None:
        nonlocal next_eid, dropped_loops, dropped_loop_km
        geom = list(e0.geometry)
        length = e0.length_km
        consumed.add(e0.eid)
        prev, cur = start, e0.v
        while cur not in real_nodes:
            nxt = None
            for x in g.adj.get(cur, []):
                if x.eid not in consumed and x.v != prev:
                    nxt = x
                    break
            if nxt is None:
                # Cannot happen for a true interior node; promote it and stop.
                real_nodes.add(cur)
                break
            consumed.add(nxt.eid)
            geom = geom[:-1] + nxt.geometry  # stitch without duplicating the joint
            length += nxt.length_km
            prev, cur = cur, nxt.v
        if cur == start:
            # A loop that returns to its own junction is useless for routing.
            dropped_loops += 1
            dropped_loop_km += length
            return
        e = Edge(eid=next_eid, u=start, v=cur, length_km=length, way_id=e0.way_id,
                 highway=e0.highway, name=e0.name, geometry=geom,
                 reversed_dir=e0.reversed_dir)
        new.edges[next_eid] = e
        new.adj[start].append(e)
        next_eid += 1

    for start in sorted(real_nodes):
        for e0 in g.adj.get(start, []):
            if e0.eid not in consumed:
                walk(start, e0)

    # Isolated rings made only of interior nodes have no real node to start
    # from: promote one node per ring and walk it.
    for e0 in list(g.edges.values()):
        if e0.eid not in consumed:
            real_nodes.add(e0.u)
            walk(e0.u, e0)

    for e in new.edges.values():
        new.nodes[e.u] = g.nodes[e.u]
        new.nodes[e.v] = g.nodes[e.v]

    new.meta.update({
        "simplified": True,
        "nodes_before_simplify": before_nodes,
        "edges_before_simplify": before_edges,
        "nodes_after_simplify": new.num_nodes(),
        "edges_after_simplify": new.num_edges(),
        "dropped_loop_edges": dropped_loops,
        "dropped_loop_km": round(dropped_loop_km, 3),
    })
    if verbose:
        print(f"[graph] simplified: {before_nodes:,} -> {new.num_nodes():,} nodes, "
              f"{before_edges:,} -> {new.num_edges():,} edges")
    return new


# --------------------------------------------------------------------------- #
# Keep the largest weakly-connected component
# --------------------------------------------------------------------------- #

def strongly_connected_components(g: RoadGraph) -> List[set]:
    """Kosaraju's algorithm, iterative so large graphs don't hit recursion limits."""
    order: List[int] = []
    seen: set = set()
    for s in g.nodes:
        if s in seen:
            continue
        seen.add(s)
        stack = [(s, iter(g.adj.get(s, [])))]
        while stack:
            u, it = stack[-1]
            for e in it:
                if e.v not in seen:
                    seen.add(e.v)
                    stack.append((e.v, iter(g.adj.get(e.v, []))))
                    break
            else:
                order.append(u)
                stack.pop()
    radj: Dict[int, List[int]] = defaultdict(list)
    for e in g.edges.values():
        radj[e.v].append(e.u)
    assigned: set = set()
    comps: List[set] = []
    for s in reversed(order):
        if s in assigned:
            continue
        comp = {s}
        assigned.add(s)
        stack2 = [s]
        while stack2:
            u = stack2.pop()
            for w in radj.get(u, ()):
                if w not in assigned:
                    assigned.add(w)
                    comp.add(w)
                    stack2.append(w)
        comps.append(comp)
    return comps


def keep_largest_component(g: RoadGraph, verbose: bool = True) -> RoadGraph:
    """
    Keep the largest *strongly* connected component, so every junction can
    reach every other one.  Weak connectivity is not enough on a road network:
    one-way streets cut at the edge of the extract create junctions you can
    enter but never leave.
    """
    weak = g.weakly_connected_components()
    comps = strongly_connected_components(g)
    if not comps:
        return g
    largest = max(comps, key=len)
    if len(largest) == g.num_nodes():
        g.meta["components"] = len(weak)
        g.meta["strong_components"] = len(comps)
        g.meta["largest_component"] = len(largest)
        return g

    new = RoadGraph()
    new.meta = dict(g.meta)
    for e in g.edges.values():
        if e.u in largest and e.v in largest:
            new.edges[e.eid] = e
            new.adj[e.u].append(e)
    for n in largest:
        if n in g.nodes:
            new.nodes[n] = g.nodes[n]
    new.meta.update({
        "components": len(weak),
        "strong_components": len(comps),
        "largest_component": len(largest),
        "kept_largest_component": True,
    })
    if verbose:
        print(f"[graph] {len(weak)} weak / {len(comps)} strong components; kept the "
              f"largest strong one: {new.num_nodes():,} nodes / {new.num_edges():,} edges")
    return new


# --------------------------------------------------------------------------- #
# Region subgraph (for classroom demos / expensive algorithms)
# --------------------------------------------------------------------------- #

def subgraph_bbox(g: RoadGraph, bbox: Tuple[float, float, float, float],
                  verbose: bool = True) -> RoadGraph:
    """Return the subgraph whose *both* endpoints fall inside the bbox."""
    min_lat, min_lon, max_lat, max_lon = bbox

    def inside(n):
        la, lo = g.nodes[n]
        return min_lat <= la <= max_lat and min_lon <= lo <= max_lon

    new = RoadGraph()
    new.meta = dict(g.meta)
    for e in g.edges.values():
        if inside(e.u) and inside(e.v):
            new.edges[e.eid] = e
            new.adj[e.u].append(e)
    for e in new.edges.values():
        new.nodes[e.u] = g.nodes[e.u]
        new.nodes[e.v] = g.nodes[e.v]
    new.meta["region_bbox"] = bbox
    if verbose:
        print(f"[graph] region subgraph: {new.num_nodes():,} nodes / "
              f"{new.num_edges():,} edges inside {bbox}")
    return keep_largest_component(new, verbose=verbose)


# --------------------------------------------------------------------------- #
# Nearest-node lookup and searchable place names
# --------------------------------------------------------------------------- #

class NodeIndex:
    """Grid-bucket spatial index for fast nearest-node queries."""

    CELL_DEG = 0.005  # ~0.5 km

    def __init__(self, graph: RoadGraph):
        self.graph = graph
        self.cells: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for n, (la, lo) in graph.nodes.items():
            self.cells[self._cell(la, lo)].append(n)

    def _cell(self, lat: float, lon: float) -> Tuple[int, int]:
        return (int(math.floor(lat / self.CELL_DEG)), int(math.floor(lon / self.CELL_DEG)))

    def nearest(self, lat: float, lon: float, max_rings: int = 40
                ) -> Tuple[Optional[int], float]:
        """Return (node, distance_km) of the nearest graph node, or (None, inf)."""
        ci, cj = self._cell(lat, lon)
        best, best_d = None, math.inf
        for r in range(max_rings + 1):
            for i in range(ci - r, ci + r + 1):
                for j in range(cj - r, cj + r + 1):
                    if max(abs(i - ci), abs(j - cj)) != r:
                        continue  # only the ring at distance r
                    for n in self.cells.get((i, j), ()):
                        d = haversine_km(self.graph.nodes[n], (lat, lon))
                        if d < best_d:
                            best, best_d = n, d
            # Anything in a farther ring is at least r cells away.
            if best is not None and best_d <= r * self.CELL_DEG * 100:
                break
        return best, best_d


def place_index(graph: RoadGraph, max_snap_km: float = 0.3) -> Dict[str, int]:
    """
    Searchable label -> node for every place in the graph:
      * each distinct named road, placed at the start node of its longest
        edge, labelled "Road name";
      * each named OSM point (junction, square) within ``max_snap_km`` of the
        network, labelled "Place name (place)".
    Unnamed roads are left out rather than given invented names.
    """
    labels: Dict[str, int] = {}
    longest: Dict[str, Edge] = {}
    for e in graph.edges.values():
        if e.name.startswith("("):
            continue
        cur = longest.get(e.name)
        if cur is None or e.length_km > cur.length_km:
            longest[e.name] = e
    for name, e in longest.items():
        labels[name] = e.u

    # Local-language (Bangla) road names, so they can be searched too.
    alt = graph.meta.get("alt_names", {})
    way_longest: Dict[int, Edge] = {}
    for e in graph.edges.values():
        if e.way_id in alt:
            cur = way_longest.get(e.way_id)
            if cur is None or e.length_km > cur.length_km:
                way_longest[e.way_id] = e
    for way_id, e in way_longest.items():
        labels.setdefault(alt[way_id], e.u)

    idx = NodeIndex(graph)
    for name, la, lo in graph.meta.get("named_points", []):
        # Skip bare codes such as "042" that are not useful place names.
        if len(name) < 3 or not any(ch.isalpha() for ch in name):
            continue
        node, d = idx.nearest(la, lo)
        if node is not None and d <= max_snap_km:
            labels.setdefault(f"{name} (place)", node)

    # Landmark aliases, anchored on names that exist in this map.
    for label, anchors in config.LANDMARK_ALIASES:
        for anchor in anchors:
            if anchor in labels:
                labels[label] = labels[anchor]
                break
    return dict(sorted(labels.items(), key=lambda kv: kv[0].lower()))


def junction_index(graph: RoadGraph) -> Dict[str, int]:
    """
    One searchable entry for EVERY node in the graph, so nothing can be missing
    from the selection list.  The label ("📍 roads — lat, lon") names up to two
    roads that meet at the node and always ends with the node's coordinates,
    which also makes every label unique.

    About half the nodes lie on lanes that have no name in OpenStreetMap.  No
    name is invented for them: they are labelled "near <road> (<m> m)" with the
    closest named road reached along the network (ignoring one-way rules).
    """
    names_at: Dict[int, set] = defaultdict(set)
    for e in graph.edges.values():
        if not e.name.startswith("("):
            names_at[e.u].add(e.name)
            names_at[e.v].add(e.name)

    # Multi-source Dijkstra from every node on a named road, over the
    # undirected network, remembering which road each search started from.
    und: Dict[int, List[Tuple[int, float]]] = defaultdict(list)
    for e in graph.edges.values():
        und[e.u].append((e.v, e.length_km))
        und[e.v].append((e.u, e.length_km))
    near: Dict[int, Tuple[float, str]] = {}
    heap: List[Tuple[float, str, int]] = []
    for n, names in names_at.items():
        heap.append((0.0, min(names), n))
    heapq.heapify(heap)
    while heap:
        d, name, n = heapq.heappop(heap)
        if n in near:
            continue
        near[n] = (d, name)
        for m, w in und.get(n, ()):
            if m not in near:
                heapq.heappush(heap, (d + w, name, m))

    labels: Dict[str, int] = {}
    for n, (la, lo) in graph.nodes.items():
        names = sorted(names_at.get(n, ()))
        if names:
            head = " × ".join(names[:2]) + (f" +{len(names) - 2}" if len(names) > 2 else "")
        elif n in near:
            d, name = near[n]
            head = f"near {name} ({d * 1000:.0f} m)"
        else:                              # isolated from every named road
            head = "unnamed road"
        label = f"📍 {head} — {la:.5f}, {lo:.5f}"
        if label in labels:               # two nodes within ~1 m: keep both
            label += f" #{n}"
        labels[label] = n
    return dict(sorted(labels.items(), key=lambda kv: kv[0].lower()))


# --------------------------------------------------------------------------- #
# Caching
# --------------------------------------------------------------------------- #

def _cache_key(path: str, simplify: bool, largest: bool) -> str:
    try:
        mtime = int(os.path.getmtime(path))
        size = os.path.getsize(path)
    except OSError:
        mtime, size = 0, 0
    base = os.path.basename(path)
    return (f"{base}.{size}.{mtime}.s{int(simplify)}.l{int(largest)}"
            f".v{GRAPH_FORMAT_VERSION}.pkl")


def load_graph(cfg: "config.AppConfig", use_cache: bool = True,
               verbose: bool = True) -> RoadGraph:
    """Load a graph for the given config, using the on-disk cache when valid."""
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(
        config.CACHE_DIR,
        _cache_key(cfg.osm_file, cfg.simplify, cfg.use_largest_component),
    )
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as fh:
                g = pickle.load(fh)
            if verbose:
                print(f"[graph] loaded from cache: {cache_path}")
            return g
        except Exception as exc:  # noqa: BLE001 - corrupt cache is non-fatal
            if verbose:
                print(f"[graph] cache unreadable ({exc}); rebuilding")

    # Fast path for bundled extract: check for a prebuilt serialized graph in data/
    bundled_gz = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data",
        f"dhaka_graph_v{GRAPH_FORMAT_VERSION}.pkl.gz",
    )
    if use_cache and cfg.simplify and cfg.use_largest_component and os.path.exists(bundled_gz):
        if os.path.basename(cfg.osm_file).lower() == os.path.basename(config.BUNDLED_OSM_FILE).lower():
            try:
                import gzip
                with gzip.open(bundled_gz, "rb") as fh:
                    g = pickle.load(fh)
                if verbose:
                    print(f"[graph] loaded from bundled prebuilt cache: {bundled_gz}")
                try:
                    with open(cache_path, "wb") as cfh:
                        pickle.dump(g, cfh, protocol=pickle.HIGHEST_PROTOCOL)
                except Exception:
                    pass
                return g
            except Exception as exc:  # noqa: BLE001
                if verbose:
                    print(f"[graph] bundled prebuilt cache unreadable ({exc}); rebuilding")

    g = build_graph(cfg.osm_file, simplify=cfg.simplify,
                    use_largest_component=cfg.use_largest_component, verbose=verbose)
    if use_cache:
        try:
            with open(cache_path, "wb") as fh:
                pickle.dump(g, fh, protocol=pickle.HIGHEST_PROTOCOL)
            if verbose:
                print(f"[graph] cached to {cache_path}")
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"[graph] could not write cache ({exc})")
    return g


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Inspect the OSM road graph.")
    ap.add_argument("--osm", default=config.DEFAULT_OSM_FILE)
    ap.add_argument("--no-simplify", action="store_true")
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    cfg = config.AppConfig(osm_file=args.osm, simplify=not args.no_simplify)
    graph = load_graph(cfg, use_cache=not args.no_cache)
    print("\n--- graph meta ---")
    for k, v in graph.meta.items():
        print(f"{k:28s}: {v}")
