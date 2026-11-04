"""
app.py — Streamlit dashboard for "Intelligent Vehicle Routing in Dhaka".

Run:
    streamlit run app.py -- --osm ~/Downloads/dhaka_central_roads.osm

Features: start/destination selection (map click or place search), map-file and
network choice, scenario + day/night, weight sliders, incident/closure editing,
run a single algorithm or Compare All Six, browser-side expansion replay with
Play/Pause, comparison table + charts, CSV and map export.
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import uuid
from copy import deepcopy
from typing import Dict, List

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import config
from graph_data import (RoadGraph, NodeIndex, junction_index, load_graph,
                        place_index, subgraph_bbox)
from scenarios import (build_scenario, manual_scenario, scenario_stats,
                       SCENARIO_NAMES, Scenario, LOCAL_EFFECTS, CLOSING_EFFECTS,
                       apply_local_conditions, with_reverse)
from cost_function import explain_example
from heuristics import verify_heuristic
from search_algorithms import ALGORITHMS, ProblemInstance, run_algorithm, SearchResult
import map_coverage
import path_view
import simulation
import visualizer as viz
from road_edits import PARAMETERS, replay_edits, replace_edit, compare_routes

st.set_page_config(page_title="Dhaka AI Routing", layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- #
# CLI args (Streamlit passes extras after `--`)
# --------------------------------------------------------------------------- #
def parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--osm", default=config.DEFAULT_OSM_FILE)
    ap.add_argument("--no-simplify", action="store_true")
    known, _ = ap.parse_known_args()
    return known


CLI = parse_cli()


def map_file_options() -> List[str]:
    """The --osm file plus the known Dhaka extracts that exist on disk."""
    candidates = [
        os.path.abspath(os.path.expanduser(CLI.osm)),
        os.path.expanduser("~/Downloads/dhaka_central_roads.osm"),
        config.BUNDLED_OSM_FILE,
    ]
    # The same map can exist in two places (Downloads and the bundled data/ copy);
    # list each file name once, preferring the earlier path.
    seen, out = set(), []
    for p in candidates:
        p = os.path.abspath(p)
        name = os.path.basename(p).lower()
        if name not in seen and os.path.exists(p):
            seen.add(name)
            out.append(p)
    return out or [CLI.osm]


# --------------------------------------------------------------------------- #
# Cached heavy resources (keyed by file + options, shared across sessions)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading & building road graph …")
def get_full_graph(osm_file: str, simplify: bool) -> RoadGraph:
    cfg = config.AppConfig(osm_file=osm_file, simplify=simplify)
    return load_graph(cfg, use_cache=True, verbose=False)


@st.cache_resource(show_spinner="Extracting demo region …")
def get_region_graph(osm_file: str, simplify: bool) -> RoadGraph:
    g = get_full_graph(osm_file, simplify)
    return subgraph_bbox(g, config.DEMO_REGION_BBOX, verbose=False)


@st.cache_resource(show_spinner="Indexing places …")
def get_indexes(osm_file: str, simplify: bool, network: str):
    g = (get_region_graph(osm_file, simplify) if network == REGION
         else get_full_graph(osm_file, simplify))

    # Fast path for bundled full network: load prebuilt place & junction indices
    if network != REGION and simplify and os.path.basename(osm_file).lower() == os.path.basename(config.BUNDLED_OSM_FILE).lower():
        from graph_data import GRAPH_FORMAT_VERSION
        import gzip
        bundled_idx = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "data",
            f"dhaka_indices_v{GRAPH_FORMAT_VERSION}.pkl.gz",
        )
        if os.path.exists(bundled_idx):
            try:
                with gzip.open(bundled_idx, "rb") as fh:
                    p_idx, j_idx = pickle.load(fh)
                return NodeIndex(g), p_idx, j_idx
            except Exception:
                pass
    return NodeIndex(g), place_index(g), junction_index(g)


@st.cache_resource
def get_coverage(osm_file: str, simplify: bool, network: str) -> dict:
    g = (get_region_graph(osm_file, simplify) if network == REGION
         else get_full_graph(osm_file, simplify))
    return map_coverage.analyse(g, get_node_names(osm_file, simplify, network))


@st.cache_resource
def get_node_names(osm_file: str, simplify: bool, network: str) -> Dict[int, str]:
    """node -> short road label (the junction label without pin and coordinates)."""
    junctions = get_indexes(osm_file, simplify, network)[2]
    return {n: label[2:].split(" — ")[0] for label, n in junctions.items()}


COLOR_COST = "Total route cost (follows my weights)"
COLOR_TRAFFIC = "Traffic only"
FULL = "Full network (whole map file)"
REGION = "Central demo region (small)"


@st.cache_resource(show_spinner="Drawing roads …", max_entries=6)
def get_base_map_blob(_graph: RoadGraph, _scenario: Scenario, osm_file: str,
                      network: str, scenario_name: str, rev: int,
                      offline: bool, color_mode: str, weights: tuple,
                      night: bool, limits: tuple, simplify: bool,
                      _session_id: str = "") -> bytes:
    """
    The road layer, drawn once and cached as pickled bytes.  Only the arguments
    without a leading underscore form the cache key; ``rev`` changes whenever
    roads are edited.

    st_folium attaches the trip layer (route, markers, explored nodes) to the
    map object it is given, so every rerun must start from a fresh copy of the
    pristine map.  Because the road layer is identical each time, st_folium sees
    an unchanged map: it keeps the user's pan/zoom and only swaps the trip layer.
    """
    m = viz.base_map(_graph, offline=offline, zoom=13 if network == REGION else 12)
    cost = color_mode == COLOR_COST
    viz.add_roads(m, _graph, _scenario, max_edges=9000,
                  include_minor=(network == REGION),
                  weights=weights if cost else None, night=night)
    viz.add_legend(m, settings=[
        f"Network: {'central demo region' if network == REGION else 'full network'} "
        f"({_graph.num_nodes():,} nodes)",
        f"Scenario: {scenario_name}, {'night' if night else 'day'}",
        f"Weights: wT {weights[0]:.2f} · wC {weights[1]:.2f} · wS {weights[2]:.2f}",
        f"Limits: {limits[0]:g} s · {limits[1]:,} expansions · IDS depth {limits[2]}",
    ], cost_mode=cost)
    return pickle.dumps((m, m.get_bounds()), protocol=pickle.HIGHEST_PROTOCOL)


def load_base_map(blob: bytes):
    """Fresh copy of the cached map, with its bounds precomputed."""
    m, bounds = pickle.loads(blob)
    m.get_bounds = lambda: bounds      # st_folium calls this on every rerun
    return m


def frame_view(nodes) -> None:
    """Centre and zoom the map on the given nodes (used for a new trip)."""
    pts = [graph.coord(n) for n in nodes if n in graph.nodes]
    if not pts:
        st.session_state.view = None
        return
    if len(set(pts)) == 1:
        st.session_state.view = (pts[0], 15)
        return
    lats, lons = [p[0] for p in pts], [p[1] for p in pts]
    clat, clon = (min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2
    span_lat = max(max(lats) - min(lats), 1e-4) * 1.5
    span_lon = max(max(lons) - min(lons), 1e-4) * 1.5
    # Degrees visible at zoom z in a ~700 x 560 px map: 984/2^z (lon), 722/2^z (lat)
    z = min(math.log2(984 / span_lon), math.log2(722 / span_lat))
    st.session_state.view = ((clat, clon), int(max(10, min(17, math.floor(z)))))


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
def init_state():
    ss = st.session_state
    ss.setdefault("start", None)
    ss.setdefault("goal", None)
    ss.setdefault("results", {})          # name -> SearchResult
    ss.setdefault("scenario_obj", None)
    ss.setdefault("scenario_sig", None)   # signature to detect rebuild need
    ss.setdefault("last_click", None)     # last map click already handled
    ss.setdefault("flash", None)          # message to show after a rerun
    ss.setdefault("scenario_rev", 0)      # bumped on every road edit
    ss.setdefault("local_edits", [])      # stretches changed in Edit roads
    ss.setdefault("view", None)           # ((lat, lon), zoom) to frame, or None
    ss.setdefault("map_session_id", uuid.uuid4().hex)
    ss.setdefault("edit_reference", None)
    ss.setdefault("replan_comparison", [])


init_state()


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.title("🚗 Dhaka AI Routing")
st.sidebar.caption("Classical AI search on a real OpenStreetMap road network")

osm_file = st.sidebar.selectbox(
    "Map file", map_file_options(), format_func=os.path.basename,
    help="Central Dhaka road extract from OpenStreetMap.")
simplify = not CLI.no_simplify
network = st.sidebar.radio(
    "Network", [FULL, REGION],
    help="DFS and IDS are slow on the full network; the small central region "
         "lets them finish.")
graph = (get_region_graph(osm_file, simplify) if network == REGION
         else get_full_graph(osm_file, simplify))
node_index, named_places, junctions = get_indexes(osm_file, simplify, network)
# Named roads and landmarks first, then every single junction.
places = {**named_places, **junctions}

st.sidebar.markdown(
    f"**Graph:** {graph.num_nodes():,} nodes · {graph.num_edges():,} directed edges  \n"
    f"**Searchable:** {len(named_places):,} roads/places + "
    f"{len(junctions):,} junctions (every node)")

scenario_name = st.sidebar.selectbox("Scenario", SCENARIO_NAMES, index=0)
night = st.sidebar.toggle("Night-time (use S_night)",
                          value=(scenario_name == "night"))

st.sidebar.subheader("Cost weights (must sum to 1)")
wT = st.sidebar.slider("Traffic  wT", 0.0, 1.0, 0.5, 0.05)
wC = st.sidebar.slider("Condition wC", 0.0, 1.0, 0.3, 0.05)
wS = st.sidebar.slider("Safety   wS", 0.0, 1.0, 0.2, 0.05)
if wT + wC + wS <= 0:
    st.sidebar.error("At least one weight must be above 0; using the defaults.")
    weights = config.Weights()
else:
    if abs(wT + wC + wS - 1.0) > 1e-9:
        st.sidebar.warning(f"Weights sum to {wT + wC + wS:.2f}; they are "
                           "normalised to 1.")
    weights = config.Weights.normalized(wT, wC, wS)
st.sidebar.caption(f"Effective: wT={weights.traffic:.2f}, wC={weights.condition:.2f}, "
                   f"wS={weights.safety:.2f}")

st.sidebar.subheader("Resource limits")
timeout_s = st.sidebar.number_input("Timeout per search (s)", 1.0, 120.0, 10.0, 1.0)
exp_budget = st.sidebar.number_input("Expansion budget", 1000, 5_000_000,
                                     400_000, 1000)
ids_depth = st.sidebar.number_input("IDS max depth", 5, 2000, 120, 5)
limits = config.Limits(timeout_s=timeout_s, expansion_budget=int(exp_budget),
                       ids_max_depth=int(ids_depth))

cfg = config.AppConfig(osm_file=osm_file, weights=weights, limits=limits)

st.sidebar.subheader("Display")
st.sidebar.slider("Map height (px)", 380, 1000, 540, 20, key="map_height",
                  help="Make the map taller on a big screen, shorter on a small laptop.")
st.sidebar.radio("Road colours show", [COLOR_COST, COLOR_TRAFFIC], key="color_mode",
                 help="Total cost uses your weights, the day/night switch and the "
                      "scenario, so the colours change when you change them.")

# Everything in the sidebar that changes a route's cost or a search's limits.  When
# one of them changes, routes computed under the old values are dropped and, if a
# start and destination are set, recomputed (Auto-run).
def queue_settings_replan(message: str) -> None:
    if "pending_replan" not in st.session_state:
        st.session_state.pending_replan = {
            "before": dict(st.session_state.results),
            "trip": (osm_file, network, st.session_state.start, st.session_state.goal),
        }
    st.session_state.results = {}
    st.session_state.previous_routes = {}
    st.session_state.replan_comparison = []
    st.session_state.auto_run = True
    if st.session_state.start is not None and st.session_state.goal is not None:
        st.session_state.flash = message + " Auto-run recalculates; otherwise press Run."


settings_sig = (round(weights.traffic, 4), round(weights.condition, 4),
                round(weights.safety, 4), bool(night), float(timeout_s),
                int(exp_budget), int(ids_depth))
if st.session_state.get("settings_sig") not in (None, settings_sig):
    queue_settings_replan("Weights, day/night or search limits changed.")
st.session_state.settings_sig = settings_sig
map_style_args = (st.session_state.get("color_mode", COLOR_COST),
                  (round(weights.traffic, 4), round(weights.condition, 4),
                   round(weights.safety, 4)), bool(night),
                  (float(timeout_s), int(exp_budget), int(ids_depth)))


# --------------------------------------------------------------------------- #
# Scenario (frozen snapshot) — rebuild only when inputs change
# --------------------------------------------------------------------------- #
def rebuild_scenario():
    if scenario_name == "manual":
        sc_ = manual_scenario(graph, night=night, seed=cfg.seed)
    else:
        sc_ = build_scenario(graph, scenario_name, seed=cfg.seed)
    st.session_state.scenario_obj = sc_
    st.session_state.baseline_scenario = deepcopy(sc_)
    st.session_state.local_edits = []         # stretches the user changed
    st.session_state.edit_reference = None
    st.session_state.replan_comparison = []
    st.session_state.previous_routes = {}
    st.session_state.results = {}
    st.session_state.scenario_rev += 1


scenario_sig = (osm_file, network, scenario_name, cfg.seed, simplify)
if st.session_state.scenario_sig != scenario_sig or st.session_state.scenario_obj is None:
    previous_sig = st.session_state.scenario_sig
    if previous_sig is not None and previous_sig[:2] == scenario_sig[:2]:
        queue_settings_replan("Scenario changed: using its new road conditions.")
    rebuild_scenario()
    st.session_state.scenario_sig = scenario_sig
    st.session_state.view = None
if st.sidebar.button("↺ Reset road edits", help="Discard closures/edits and "
                     "restore the selected scenario's simulated values."):
    st.session_state.reset_before = dict(st.session_state.results)
    rebuild_scenario()
    st.session_state.flash = "Road edits reset."
sc: Scenario = st.session_state.scenario_obj
sc.night = night

# A start/goal from another map file or network may not exist in this graph.
for key in ("start", "goal"):
    if st.session_state[key] is not None and st.session_state[key] not in graph.nodes:
        st.session_state[key] = None
        st.session_state.results = {}


# --------------------------------------------------------------------------- #
# Start / destination selection helpers
# --------------------------------------------------------------------------- #
# Callbacks run before the next script run, so they read the place list from
# session state rather than from a variable of the previous run.
st.session_state.places_map = places


def set_point(which: str, node: int, label: str = "") -> None:
    st.session_state[which] = node
    st.session_state[f"{which}_label"] = label     # what the user picked, if anything
    st.session_state.results = {}
    st.session_state.auto_run = True
    st.session_state.edit_reference = None
    st.session_state.replan_comparison = []
    st.session_state.previous_routes = {}
    st.session_state.pop("pending_replan", None)


def _picked(which: str) -> None:
    label = st.session_state.get(f"pick_{which}")
    node = st.session_state.get("places_map", {}).get(label) if label else None
    if node is None:
        return
    set_point(which, node, label)
    frame_view([n for n in (st.session_state.start, st.session_state.goal)
                if n is not None])


def pick_start_cb() -> None:
    _picked("start")


def pick_goal_cb() -> None:
    _picked("goal")


# --------------------------------------------------------------------------- #
# Header + tabs (compact, so the map fits a laptop screen)
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
  .block-container { padding-top: 2.6rem; padding-bottom: 1rem; }
  section[data-testid="stSidebar"] { width: 270px !important; min-width: 270px !important; }
  .app-title { font-size: 1.35rem; font-weight: 700; margin: 0; line-height: 1.3; }
  .app-sub { font-size: 0.78rem; color: #6b6b6b; margin: 0 0 0.2rem 0; }
  div[data-testid="stTabs"] button p { font-size: 0.95rem; }
</style>
<p class="app-title">🚗 Intelligent Vehicle Routing in Dhaka</p>
<p class="app-sub">Map data © OpenStreetMap contributors (ODbL). Traffic, condition and
safety values are <b>simulated</b>, not live data.</p>
""", unsafe_allow_html=True)
if st.session_state.flash:
    st.success(st.session_state.flash)
    st.session_state.flash = None

tab_map, tab_compare, tab_edit, tab_about = st.tabs(
    ["🗺️ Map & Run", "📊 Compare", "🛠️ Edit roads", "📄 About / validation"])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
VIEW_MAP = "🗺️ Map"
VIEW_SIM = "🎬 Simulation"
VIEW_PATH = "🧭 Path"
VIEW_COVER = "📍 Coverage"


def build_problem() -> ProblemInstance:
    return ProblemInstance(graph, sc.get, cfg.weights, sc.night, cfg.limits)


def run_selected(names: List[str], show_simulation: bool = False) -> None:
    s, g = st.session_state.start, st.session_state.goal
    if s is None or g is None:
        st.warning("Select a start and a destination first.")
        return
    prob = build_problem()
    results: Dict[str, SearchResult] = {}
    # Say what is running and what has finished, so a slow IDS does not
    # look like a frozen app.
    slow_note = {
        "IDS": f"IDS repeats DFS for every depth limit, so it is the slowest "
               f"(it stops after {cfg.limits.timeout_s:.0f} s at most).",
    }
    prog = st.progress(0.0, text="Running searches …")
    done: List[str] = []
    for i, name in enumerate(names):
        lines = [f"**Running {name}** ({i + 1} of {len(names)}) …"]
        if name in slow_note:
            lines.append(slow_note[name])
        if done:
            lines.append("Finished: " + " · ".join(done))
        prog.progress(i / len(names), text="  \n".join(lines))
        r = results[name] = run_algorithm(name, prob, s, g, record_trace=True)
        done.append(f"{name} {'✓' if r.found else '✗'} {r.time_ms / 1000:.1f} s")
    prog.empty()
    st.session_state.results = results
    best = next((results[n] for n in ("A*", "UCS", "GBFS", "BFS", "DFS", "IDS")
                 if n in results and results[n].found), None)
    frame_view(best.node_path if best else [s, g])
    if show_simulation:
        # Show the searches as soon as they finish.  (Not after map clicks:
        # the map must stay clickable for the next trip.)  One algorithm on
        # its own opens its node-by-node path; several open the simulation.
        found = [n for n in names if results[n].found]
        if len(names) == 1 and found:
            st.session_state.pending_path_algo = found[0]
            st.session_state.pending_view = VIEW_PATH
        else:
            st.session_state.pending_view = VIEW_SIM
            st.session_state.sim_autoplay = True


def commit_road_edits(edits, message: str) -> None:
    """Rebuild edits from a clean snapshot and re-run the same algorithms."""
    global sc
    before = dict(st.session_state.results)
    updated = replay_edits(graph, st.session_state.baseline_scenario, edits)
    updated.night = night
    sc = updated
    st.session_state.scenario_obj = updated
    st.session_state.local_edits = edits
    st.session_state.scenario_rev += 1
    st.session_state.results = {}
    st.session_state.previous_routes = before
    st.session_state.pending_view = VIEW_MAP
    if st.session_state.start is not None and st.session_state.goal is not None:
        names = list(before) or st.session_state.get("algo_choice") or ["UCS", "A*", "GBFS"]
        run_selected(names)
    st.session_state.replan_comparison = compare_routes(
        before, st.session_state.results, sc, cfg.weights, night)
    st.session_state.flash = message + " Routes recalculated with the new values."
    st.rerun()


# Reset is handled after run_selected exists, so it also recalculates the routes.
if "reset_before" in st.session_state:
    before = st.session_state.pop("reset_before")
    if before and st.session_state.start is not None and st.session_state.goal is not None:
        run_selected(list(before))
        st.session_state.previous_routes = before
        st.session_state.replan_comparison = compare_routes(
            before, st.session_state.results, sc, cfg.weights, night)


def show_html(page: str, height: int) -> None:
    """Embed a self-contained page with automatic Leaflet container resizing."""
    auto_resize_script = """
<script>
(function() {
    function fixLeaflet() {
        for (var key in window) {
            try {
                if (window[key] && typeof window[key].invalidateSize === 'function') {
                    window[key].invalidateSize();
                }
            } catch(e) {}
        }
    }
    window.addEventListener('resize', fixLeaflet);
    window.addEventListener('focus', fixLeaflet);
    document.addEventListener('visibilitychange', function() {
        if (!document.hidden) fixLeaflet();
    });
    if (window.ResizeObserver) {
        var ro = new ResizeObserver(function() { fixLeaflet(); });
        if (document.body) ro.observe(document.body);
        if (document.documentElement) ro.observe(document.documentElement);
    }
    var count = 0;
    var timer = setInterval(function() {
        fixLeaflet();
        count++;
        if (count > 25) clearInterval(timer);
    }, 200);
})();
</script>
"""
    if "</body>" in page:
        page = page.replace("</body>", auto_resize_script + "</body>", 1)
    elif "</html>" in page:
        page = page.replace("</html>", auto_resize_script + "</html>", 1)
    else:
        page = page + auto_resize_script

    if hasattr(st, "iframe"):                 # st.components.v1.html is being removed
        st.iframe(page, height=height)
    else:
        components.html(page, height=height, scrolling=False)


def node_text(which: str) -> str:
    n = st.session_state[which]
    if n is None:
        return "not set"
    la, lo = graph.coord(n)
    picked = st.session_state.get(f"{which}_label")
    if picked:
        what = picked.split(" — ")[0] if picked.startswith("📍") else picked
        return f"{what} ({la:.5f}, {lo:.5f})"
    road = graph.node_label(n).rsplit(" [", 1)[0]
    if road.startswith("node "):
        return f"map point ({la:.5f}, {lo:.5f})"
    return f"map point at {road} ({la:.5f}, {lo:.5f})"


# Read by the map-click callback, which runs before the next script run.
st.session_state.node_index_obj = node_index


def map_click_cb() -> None:
    """
    Handle a map click BEFORE the rerun starts, so the new marker is drawn in the
    same rerun and no click is lost while the app is busy.
    1st click = start, 2nd = destination, 3rd starts a new trip.
    """
    out = st.session_state.get("main_map") or {}
    click = out.get("last_clicked")
    if not click:
        return
    click_key = (round(click["lat"], 7), round(click["lng"], 7))
    if click_key == st.session_state.last_click:
        return
    st.session_state.last_click = click_key
    nn, dist_km = st.session_state.node_index_obj.nearest(click["lat"], click["lng"])
    if nn is None or dist_km > 0.5:
        st.session_state.map_note = "That click is more than 500 m from any road in this network."
        return
    st.session_state.map_note = None
    had_picks = bool(st.session_state.get("pick_start") or st.session_state.get("pick_goal"))
    if st.session_state.start is None or st.session_state.goal is not None:
        set_point("start", nn)
        st.session_state.goal = None
        st.session_state.goal_label = ""
        st.session_state.pick_start = None
        st.session_state.pick_goal = None
    else:
        set_point("goal", nn)
        st.session_state.pick_goal = None
    # A finished trip (or stale names in the From/To boxes) needs the whole app
    # refreshed; a lone start point only needs the map.
    if st.session_state.goal is not None or had_picks:
        st.session_state.needs_app_rerun = True


st.session_state.setdefault("map_view", VIEW_MAP)
st.session_state.setdefault("map_note", None)
map_height = int(st.session_state.get("map_height", 540))


@st.fragment
def map_area() -> None:
    """
    The map / simulation panel.  It is a fragment: a map click reruns only this
    panel, not the sidebar, the place lists or the other tabs, so it stays fast.
    """
    if st.session_state.pop("needs_app_rerun", False):
        st.rerun(scope="app")
    pending = st.session_state.pop("pending_view", None)
    if pending:
        st.session_state.map_view = pending
    pending_algo = st.session_state.pop("pending_path_algo", None)
    if pending_algo:
        st.session_state.path_algo = pending_algo

    top_l, top_r = st.columns([3, 2])
    with top_l:
        st.markdown(f"🟢 **Start:** {node_text('start')}  \n"
                    f"🔴 **Destination:** {node_text('goal')}")
    with top_r:
        view = st.radio("View", [VIEW_MAP, VIEW_SIM, VIEW_PATH, VIEW_COVER], key="map_view",
                        horizontal=True, label_visibility="collapsed")

    found_algos = [name for name in ("A*", "UCS", "GBFS", "BFS", "DFS", "IDS")
                   if name in st.session_state.results and st.session_state.results[name].found]
    if len(found_algos) > 1 and view == VIEW_MAP:
        st.radio("Highlight route on map", ["All algorithms", *found_algos],
                 horizontal=True, key="focus_map_algo")
    res_dict = st.session_state.results
    if res_dict and not any(r.found for r in res_dict.values()):
        st.error("⛔ **Destination Unreachable:** All search algorithms failed to find an alternative route because roads leading to the destination are closed/blocked. "
                 "Go to **🛠️ Edit roads** to remove or reduce the blockage so a detour can be found.")
    if st.session_state.map_note:
        st.caption("⚠️ " + st.session_state.map_note)

    import folium

    def trip_layer() -> "folium.FeatureGroup":
        fg = folium.FeatureGroup(name="trip")
        focus_algo = st.session_state.get("focus_map_algo", "All algorithms")

        # Draw previous route(s) (before road edit)
        if st.session_state.get("show_previous_route", True):
            for old_name, old in st.session_state.get("previous_routes", {}).items():
                if old.found and (focus_algo == "All algorithms" or focus_algo == old_name):
                    coords = [coord for edge in old.edge_path for coord in edge.geometry]
                    folium.PolyLine(coords, color="#64748b", weight=6,
                                    opacity=0.75, dash_array="8 6",
                                    tooltip=f"⏮️ Previous {old_name} route (before edit)",
                                    interactive=False).add_to(fg)

        # Draw current active route(s) (after road edit)
        for name, res in st.session_state.results.items():
            if res.found:
                if focus_algo == "All algorithms":
                    viz.add_route(fg, res.edge_path,
                                  color=viz.ROUTE_COLORS.get(name, "#333"), weight=5)
                elif focus_algo == name:
                    viz.add_route(fg, res.edge_path,
                                  color=viz.ROUTE_COLORS.get(name, "#2563eb"), weight=7)

        # Draw edited / blocked roads with high visibility hazard styling & markers
        edited_ids = {eid for edit in st.session_state.local_edits for eid in edit["eids"]}
        for eid in edited_ids:
            if eid in graph.edges:
                edge = graph.edges[eid]
                attr = sc.get(eid)
                geom = edge.geometry
                mid_pt = geom[len(geom) // 2]
                if attr.closed:
                    # High-visibility black & red hazard barrier
                    folium.PolyLine(geom, color="#000000", weight=11, opacity=0.9, interactive=False).add_to(fg)
                    folium.PolyLine(geom, color="#ef4444", weight=6, opacity=1.0, dash_array="6 6", interactive=False).add_to(fg)
                    folium.Marker(
                        mid_pt,
                        icon=folium.Icon(color="red", icon="ban", prefix="fa"),
                        tooltip=f"⛔ ROAD CLOSED: {edge.name}",
                        popup=f"<b>⛔ ROAD CLOSED / BLOCKED</b><br>{edge.name}<br>Length: {edge.length_km*1000:.0f} m",
                    ).add_to(fg)
                elif attr.incident:
                    inc_color = "#0284c7" if attr.incident == "flood" else "#ea580c"
                    icon_name = "tint" if attr.incident == "flood" else "warning"
                    folium.PolyLine(geom, color=inc_color, weight=8, opacity=0.9, interactive=False).add_to(fg)
                    folium.Marker(
                        mid_pt,
                        icon=folium.Icon(color="orange" if attr.incident != "flood" else "blue", icon=icon_name, prefix="fa"),
                        tooltip=f"⚠️ {attr.incident.title()}: {edge.name}",
                        popup=f"<b>⚠️ {attr.incident.title()}</b><br>{edge.name}",
                    ).add_to(fg)
                else:
                    color = "#b91c1c" if attr.traffic >= 4.0 else "#c225c2"
                    folium.PolyLine(geom, color=color, weight=7, opacity=0.85, dash_array="4 6", interactive=False).add_to(fg)
                    folium.Marker(
                        mid_pt,
                        icon=folium.Icon(color="darkred" if attr.traffic >= 4.0 else "purple", icon="tachometer", prefix="fa"),
                        tooltip=f"Edited road: {edge.name} (Traffic: {attr.traffic:.1f}, Cond: {attr.condition:.1f})",
                    ).add_to(fg)

        if st.session_state.start is not None:
            folium.Marker(graph.coord(st.session_state.start),
                          icon=folium.Icon(color="green", icon="play")).add_to(fg)
        if st.session_state.goal is not None:
            folium.Marker(graph.coord(st.session_state.goal),
                          icon=folium.Icon(color="red", icon="flag")).add_to(fg)
        return fg

    offline = bool(st.session_state.get("offline_tiles", False))
    results = st.session_state.results

    if view == VIEW_COVER:
        info = get_coverage(osm_file, simplify, network)
        e = info["extremes"]
        covered = [a["name"] for a in info["areas"] if a["status"] == "covered"]
        thin = [a["name"] for a in info["areas"] if a["status"] == "thin"]
        none = [a["name"] for a in info["areas"] if a["status"] == "none"]
        st.markdown(
            f"**This map spans about {info['width_km']:.0f} km (west–east) × "
            f"{info['height_km']:.0f} km (south–north)** — {info['n_nodes']:,} road "
            f"junctions. Northmost: *{e['north']['road']}*; southmost: "
            f"*{e['south']['road']}*; eastmost: *{e['east']['road']}*; westmost: "
            f"*{e['west']['road']}*.")
        cov_key = (osm_file, network, offline, map_height)
        cached = st.session_state.get("cover_cache")
        if cached and cached[0] == cov_key:
            page = cached[1]
        else:
            page = map_coverage.build_html(graph, info, offline, height=map_height)
            st.session_state.cover_cache = (cov_key, page)
        show_html(page, map_height + 4)
        st.markdown(
            "🟢 **Covered:** " + ", ".join(covered) + "\n\n"
            + ("🟠 **Only a little:** " + ", ".join(thin) + "\n\n" if thin else "")
            + ("⚪ **Not in this map:** " + ", ".join(none) if none else ""))
        st.caption("Blue shading = where road junctions exist (darker = more); red dashed "
                   "box = the extent; red flags = the four extreme junctions. Area centres "
                   "are approximate positions used only to count nearby junctions within "
                   f"{map_coverage.RADIUS_KM:g} km, not official boundaries.")
        return

    if view == VIEW_PATH:
        found = [n for n in simulation.ALGO_ORDER if n in results and results[n].found]
        if not found:
            if results:
                st.error("⛔ **No route found:** The destination is currently unreachable under current road closure(s).")
                st.info("💡 **Why this happened:** The road edit or closure completely blocked the only connected path to the destination. "
                        "Go to **🛠️ Edit roads** and click **Remove** on the closure or edit fewer segments to allow an alternative detour.")
                for name, res in results.items():
                    st.caption(f"• **{name}**: ⚠️ {res.status.value} ({res.nodes_expanded:,} nodes explored)")
            else:
                st.info("Run an algorithm that finds a path; its route will be listed "
                        "here node by node.")
            return
        if st.session_state.get("path_algo") not in found:
            st.session_state.path_algo = found[-1]          # A* if it was run
        algo = st.radio("Path of", found, key="path_algo", horizontal=True,
                        label_visibility="collapsed")
        res = results[algo]
        pv_key = (algo, id(res), offline, map_height)
        cached = st.session_state.get("path_cache")
        if cached and cached[0] == pv_key:
            page = cached[1]
        else:
            page = path_view.build_html(
                graph, res, algo, get_node_names(osm_file, simplify, network),
                offline, height=map_height)
            st.session_state.path_cache = (pv_key, page)
        show_html(page, map_height + 4)
        return

    if view == VIEW_SIM:
        if not results or st.session_state.start is None or st.session_state.goal is None:
            st.info("Choose a start and destination and run some algorithms; they "
                    "will explore the map here together.")
        else:
            autoplay = st.session_state.pop("sim_autoplay", False)
            sim_key = (tuple((n, id(r)) for n, r in results.items()), offline, map_height)
            cached = st.session_state.get("sim_cache")
            if cached and cached[0] == sim_key:
                page = cached[1]
            else:
                with st.spinner("Preparing the simulation …"):
                    page = simulation.build_html(graph, results, st.session_state.start,
                                                 st.session_state.goal, offline,
                                                 map_height=map_height)
                st.session_state.sim_cache = (sim_key, page)
            if page:
                if autoplay:
                    page = page.replace("var AUTOPLAY = false;", "var AUTOPLAY = true;", 1)
                show_html(page, simulation.page_height(len(results), map_height))
            else:
                st.info("Nothing to replay (start equals destination).")
            if st.toggle("Offline replay (plain chart, one algorithm, no internet needed)",
                         value=False):
                a1, a2 = st.columns([1, 2])
                with a1:
                    anim_algo = st.selectbox("Algorithm to replay", list(results))
                with a2:
                    frame_ms = st.slider("Speed (ms per frame; lower is faster)",
                                         20, 600, 120, 20)
                res = results[anim_algo]
                replay_key = (anim_algo, frame_ms, id(res))
                cached = st.session_state.get("replay")
                if cached and cached[0] == replay_key:
                    fig = cached[1]
                else:
                    fig = viz.expansion_animation(graph, res, frame_ms=frame_ms)
                    st.session_state.replay = (replay_key, fig)
                if fig is not None:
                    st.plotly_chart(fig, width="stretch")
        return

    base = load_base_map(get_base_map_blob(
        graph, sc, osm_file, network, scenario_name,
        st.session_state.scenario_rev, offline, *map_style_args,
        simplify=simplify, _session_id=st.session_state.map_session_id))
    view_pos = st.session_state.view
    use_interactive = st.session_state.get("interactive_click_mode", False)
    trip = trip_layer()
    if use_interactive:
        try:
            from streamlit_folium import st_folium
            st_folium(base, key="main_map", height=map_height, use_container_width=True,
                      feature_group_to_add=trip,
                      center=view_pos[0] if view_pos else None,
                      zoom=view_pos[1] if view_pos else None,
                      returned_objects=["last_clicked"], on_change=map_click_cb)
        except Exception:
            trip.add_to(base)
            show_html(base.get_root().render(), map_height + 4)
    else:
        trip.add_to(base)
        show_html(base.get_root().render(), map_height + 4)

    if st.button("Prepare map for download (HTML)"):
        full = viz.base_map(graph, offline=offline,
                            zoom=13 if network == REGION else 12)
        viz.add_roads(full, graph, sc, max_edges=9000,
                      include_minor=(network == REGION),
                      weights=map_style_args[1] if map_style_args[0] == COLOR_COST else None,
                      night=night)
        trip_layer().add_to(full)
        viz.add_legend(full, cost_mode=map_style_args[0] == COLOR_COST)
        st.download_button("⬇ Download route_map.html",
                           full.get_root().render().encode("utf-8"),
                           file_name="route_map.html", mime="text/html")


# =========================================================================== #
# TAB: MAP & RUN
# =========================================================================== #
with tab_map:
    left, right = st.columns([2.6, 1])

    with right:
        # A map click or swap clears the boxes so they never show a stale choice.
        for key in st.session_state.pop("clear_picks", ()):
            st.session_state[key] = None
        if network == REGION:
            search_options = list(places)
        else:
            search_options = list(named_places)
            for p in (st.session_state.get("start_label"), st.session_state.get("goal_label")):
                if p and p in places and p not in named_places:
                    search_options.append(p)

        st.selectbox("From (start)", search_options, index=None, key="pick_start",
                     placeholder="Type to search, e.g. Dhaka University",
                     on_change=pick_start_cb)
        st.selectbox("To (destination)", search_options, index=None, key="pick_goal",
                     placeholder="Type to search, e.g. Motijheel",
                     on_change=pick_goal_cb)
        st.caption("Select your start and destination above from 4,000+ named places & roads. "
                   "Turn on 'Interactive map clicks' below if you want to click on the map.")

        rc1, rc2 = st.columns(2)
        with rc1:
            run_clicked = st.button("▶ Run", type="primary", width="stretch",
                                    help="Run the algorithms selected below")
        with rc2:
            all_clicked = st.button("⚖ All six", width="stretch",
                                    help="Run all six algorithms")
        algo_choice = st.multiselect("Algorithms", list(ALGORITHMS.keys()),
                                     default=["UCS", "A*", "GBFS"], key="algo_choice")
        auto_on = st.checkbox("Auto-run", value=True,
                              help="Run the selected algorithms as soon as both points are set.")

        pending_auto = st.session_state.pop("auto_run", False)
        pending_change = st.session_state.get("pending_replan")
        if pending_change and pending_change["trip"] != (
                osm_file, network, st.session_state.start, st.session_state.goal):
            st.session_state.pop("pending_replan", None)
            pending_change = None
        names_to_run = None
        if run_clicked:
            names_to_run = algo_choice or list(ALGORITHMS.keys())
        elif all_clicked:
            names_to_run = list(ALGORITHMS.keys())
        elif (pending_auto and auto_on and st.session_state.start is not None
              and st.session_state.goal is not None):
            if st.session_state.start == st.session_state.goal:
                st.info("Start and destination are the same place.")
            else:
                names_to_run = (list(pending_change["before"])
                                if pending_change and pending_change["before"]
                                else algo_choice or list(ALGORITHMS.keys()))
        if names_to_run:
            run_selected(names_to_run, show_simulation=run_clicked or all_clicked)
            if pending_change:
                before = pending_change["before"]
                st.session_state.previous_routes = before
                st.session_state.replan_comparison = compare_routes(
                    before, st.session_state.results, sc, cfg.weights, night)
                st.session_state.pop("pending_replan", None)

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⇄ Swap", width="stretch",
                         disabled=st.session_state.start is None
                         and st.session_state.goal is None):
                st.session_state.start, st.session_state.goal = (
                    st.session_state.goal, st.session_state.start)
                st.session_state.start_label, st.session_state.goal_label = (
                    st.session_state.get("goal_label"), st.session_state.get("start_label"))
                st.session_state.results = {}
                st.session_state.auto_run = True
                st.session_state.clear_picks = ["pick_start", "pick_goal"]
                st.session_state.replan_comparison = []
                st.session_state.previous_routes = {}
                st.session_state.edit_reference = None
                st.rerun()
        with c2:
            if st.button("🔄 Reset", width="stretch"):
                st.session_state.start = st.session_state.goal = None
                st.session_state.start_label = st.session_state.goal_label = ""
                st.session_state.results = {}
                st.session_state.view = None
                st.session_state.map_view = VIEW_MAP
                st.session_state.replan_comparison = []
                st.session_state.previous_routes = {}
                st.session_state.edit_reference = None
                st.session_state.clear_picks = ["pick_start", "pick_goal"]
                st.rerun()

        if st.session_state.results:
            st.markdown("**Results**")
            for name, r in st.session_state.results.items():
                st.caption(f"{name}: " + (f"✅ {r.distance_km:.2f} km, cost "
                                         f"{r.weighted_cost:.2f}, {r.nodes_expanded:,} "
                                         f"expanded, {r.time_ms:.1f} ms"
                                         if r.found else f"⚠️ {r.status.value} "
                                         f"({r.nodes_expanded:,} expanded)"))
        st.toggle("Offline map (no tiles)", key="offline_tiles",
                  help="Draw roads without downloading map tiles.")
        st.toggle("Interactive map clicks (experimental)", key="interactive_click_mode",
                  help="Uses streamlit-folium for map clicking. Keep off if your browser or network has trouble loading the custom component.")

    with left:
        map_area()


# =========================================================================== #
# TAB: COMPARE
# =========================================================================== #
with tab_compare:
    st.subheader("Comparison table")
    if not st.session_state.results:
        st.info("Run some algorithms on the Map & Run tab first.")
    else:
        results = st.session_state.results
        ucs = results.get("UCS")
        ucs_cost = ucs.weighted_cost if (ucs and ucs.found) else None
        rows = []
        for name, r in results.items():
            rows.append({
                "algorithm": name,
                "status": r.status.value,
                "distance_km": round(r.distance_km, 3) if r.found else None,
                "weighted_cost": round(r.weighted_cost, 3) if r.found else None,
                "hops": r.hops if r.found else None,
                "nodes_expanded": r.nodes_expanded,
                "unique_expanded": r.unique_states_expanded,
                "nodes_generated": r.nodes_generated,
                "peak_frontier": r.peak_frontier,
                "time_ms": round(r.time_ms, 3),
                "cost_ratio_vs_UCS": (round(r.weighted_cost / ucs_cost, 3)
                                      if (r.found and ucs_cost) else None),
                "limit_hit": r.extra.get("limit", r.extra.get("reason", "")),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, width="stretch", hide_index=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇ Download comparison CSV", csv,
                           file_name="comparison.csv", mime="text/csv")

        st.subheader("Charts")
        st.caption("Use the camera icon on each chart to save it as PNG.")
        cc = st.columns(2)
        with cc[0]:
            st.plotly_chart(viz.comparison_bar(results, "weighted_cost",
                            "Weighted path cost"), width="stretch")
            st.plotly_chart(viz.comparison_bar(results, "nodes_expanded",
                            "Nodes expanded"), width="stretch")
        with cc[1]:
            st.plotly_chart(viz.comparison_bar(results, "time_ms",
                            "Search runtime (ms)"), width="stretch")
            st.plotly_chart(viz.comparison_bar(results, "distance_km",
                            "Physical path length (km)"), width="stretch")

        a = results.get("A*")
        if ucs and a and ucs.found and a.found:
            agree = abs(ucs.weighted_cost - a.weighted_cost) < 1e-6
            st.success(f"UCS and A* optimal-cost agreement: {agree} "
                       f"(UCS={ucs.weighted_cost:.4f}, A*={a.weighted_cost:.4f})")


# =========================================================================== #
# TAB: EDIT ROADS
# =========================================================================== #
with tab_edit:
    st.subheader("Change a road and compare the new route")
    st.caption("Choose one or several parameters. Apply recalculates the algorithms you ran. "
               "A*/UCS change route only when another route is cheaper. BFS/DFS/IDS react "
               "to closures, not traffic or safety costs. Incident labels alone do not change cost.")
    st.checkbox("Show previous routes on Map (grey dashed); edited roads are magenta",
                value=True, key="show_previous_route")

    comparison = st.session_state.get("replan_comparison")
    if comparison:
        st.markdown("### 📊 Re-routing Impact & Map Comparison")
        comp_algos = [row["algorithm"] for row in comparison if row.get("before_status") == "found"]
        if comp_algos:
            sel_options = ["All algorithms", *comp_algos] if len(comp_algos) > 1 else comp_algos
            sel_algo = st.selectbox("Inspect Before/After detour on map", sel_options, key="comp_algo_select")
            row = next((r for r in comparison if r["algorithm"] == sel_algo), None)
            if row:
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Path Result", "🔄 Detour Taken" if row["path_changed"] else "✅ Route Unchanged")
                m2.metric("Distance", f"{row['after_km']:.2f} km" if row['after_km'] else "Blocked",
                          delta=f"{(row['after_km'] - row['before_km']):+.2f} km" if row['after_km'] and row['before_km'] else None)
                m3.metric("Old Cost", f"{row['previous_cost']:.2f}" if row['previous_cost'] is not None else "N/A")
                m4.metric("New Cost", f"{row['new_path_cost']:.2f}" if row['new_path_cost'] is not None else "∞ (Blocked)",
                          delta=f"{(row['new_path_cost'] - row['previous_cost']):+.2f}" if row['new_path_cost'] and row['previous_cost'] else None,
                          delta_color="inverse")
            elif sel_algo == "All algorithms":
                detours = sum(1 for r in comparison if r.get("path_changed"))
                st.info(f"Showing all algorithms together on the map: **{detours}** algorithm(s) took an alternative detour around the blockage.")

            # Dedicated visual comparison map right here in Edit roads
            import folium
            comp_map = viz.base_map(graph, offline=offline, zoom=13 if network == REGION else 12)
            cfg_trip = folium.FeatureGroup(name="comp_trip")

            if sel_algo == "All algorithms":
                # Draw all old routes in grey dashed
                for old_name, old_res in st.session_state.previous_routes.items():
                    if old_res and old_res.found:
                        coords = [coord for edge in old_res.edge_path for coord in edge.geometry]
                        folium.PolyLine(coords, color="#64748b", weight=5, opacity=0.6,
                                        dash_array="8 6", tooltip=f"⏮️ {old_name} Before Edit").add_to(cfg_trip)
                # Draw all new routes in distinct algorithm colors
                for name, new_res in st.session_state.results.items():
                    if new_res and new_res.found:
                        coords = [coord for edge in new_res.edge_path for coord in edge.geometry]
                        folium.PolyLine(coords, color=viz.ROUTE_COLORS.get(name, "#2563eb"), weight=6, opacity=0.85,
                                        tooltip=f"⏭️ {name} New Detour Route").add_to(cfg_trip)
            else:
                # 1. Old route before edit in grey dashed
                old_res = st.session_state.previous_routes.get(sel_algo)
                if old_res and old_res.found:
                    coords = [coord for edge in old_res.edge_path for coord in edge.geometry]
                    folium.PolyLine(coords, color="#64748b", weight=6, opacity=0.75,
                                    dash_array="8 6", tooltip=f"⏮️ {sel_algo} Original Route (Before Edit)").add_to(cfg_trip)

                # 2. New route after edit in bright blue
                new_res = st.session_state.results.get(sel_algo)
                if new_res and new_res.found:
                    coords = [coord for edge in new_res.edge_path for coord in edge.geometry]
                    folium.PolyLine(coords, color="#2563eb", weight=6, opacity=0.9,
                                    tooltip=f"⏭️ {sel_algo} New Detour Route (After Edit)").add_to(cfg_trip)

            # 3. Highlight the blocked road in bold hazard barrier style with ⛔ marker
            edited_ids = {eid for edit in st.session_state.local_edits for eid in edit["eids"]}
            for eid in edited_ids:
                if eid in graph.edges:
                    edge = graph.edges[eid]
                    attr = sc.get(eid)
                    geom = edge.geometry
                    mid_pt = geom[len(geom) // 2]
                    if attr.closed:
                        folium.PolyLine(geom, color="#000000", weight=11, opacity=0.9).add_to(cfg_trip)
                        folium.PolyLine(geom, color="#ef4444", weight=6, opacity=1.0, dash_array="6 6").add_to(cfg_trip)
                        folium.Marker(mid_pt, icon=folium.Icon(color="red", icon="ban", prefix="fa"),
                                      tooltip=f"⛔ ROAD BLOCKED: {edge.name}").add_to(cfg_trip)
                    else:
                        folium.PolyLine(geom, color="#ea580c", weight=8, opacity=0.9).add_to(cfg_trip)
                        folium.Marker(mid_pt, icon=folium.Icon(color="orange", icon="warning", prefix="fa"),
                                      tooltip=f"⚠️ Edited Road: {edge.name}").add_to(cfg_trip)

            if st.session_state.start is not None:
                folium.Marker(graph.coord(st.session_state.start),
                              icon=folium.Icon(color="green", icon="play")).add_to(cfg_trip)
            if st.session_state.goal is not None:
                folium.Marker(graph.coord(st.session_state.goal),
                              icon=folium.Icon(color="red", icon="flag")).add_to(cfg_trip)

            cfg_trip.add_to(comp_map)
            show_html(comp_map.get_root().render(), 380)
            st.caption("Map Legend: ⚪ Grey Dashed = Original Route (Before) | 🔵 Blue Solid = New Rerouted Path (After) | ⛔ Red = Blockage")

        st.markdown("**Complete Multi-Algorithm Impact Table**")
        st.dataframe(pd.DataFrame(comparison), width="stretch", hide_index=True)
        st.caption("Compare old_path_at_new_cost with new_path_cost: both use the new "
                   "road values. An infinite old-path cost means that route is now blocked. "
                   "If the path is unchanged, the edit may still have changed its cost.")

    current = st.session_state.results
    reference = st.session_state.get("edit_reference")
    trip_sig = (osm_file, network, st.session_state.start, st.session_state.goal)
    if reference and reference.get("trip") != trip_sig:
        reference = None
        st.session_state.edit_reference = None
    available = [name for name in ALGORITHMS if name in current and current[name].found]
    if reference and reference.get("algorithm") not in available:
        available.append(reference["algorithm"])
    edit_algorithm = reference.get("algorithm") if reference else st.session_state.get("edit_algorithm", "A*")
    candidate_edges = []
    if available:
        if st.session_state.get("edit_algorithm") not in available:
            st.session_state.edit_algorithm = "A*" if "A*" in available else available[0]
        edit_algorithm = st.selectbox("Edit the path of algorithm", available,
                                      key="edit_algorithm")
        latest = current.get(edit_algorithm)
        if reference is None or reference.get("algorithm") != edit_algorithm:
            if latest and latest.found:
                reference = {"trip": trip_sig, "algorithm": edit_algorithm,
                             "eids": [e.eid for e in latest.edge_path]}
                st.session_state.edit_reference = reference
        if st.button("Use this algorithm's latest path for editing",
                     disabled=not (latest and latest.found)):
            st.session_state.edit_reference = {
                "trip": trip_sig, "algorithm": edit_algorithm,
                "eids": [e.eid for e in latest.edge_path]}
            st.rerun()
        if reference:
            candidate_edges = [graph.edges[eid] for eid in reference.get("eids", []) if eid in graph.edges]
        st.caption("The selected path stays available after rerouting, so you can change "
                   "or restore the same road. Use the button above to select the latest path.")
    elif reference:
        candidate_edges = [graph.edges[eid] for eid in reference.get("eids", []) if eid in graph.edges]

    source = st.radio("Pick edge from", ["Part of current route", "Current route",
                                         "Search by road name"], horizontal=True)
    part = []
    label = ""
    if source == "Part of current route" and candidate_edges:
        marks, distance = [], 0.0
        for i, edge in enumerate(candidate_edges):
            marks.append(f"{i + 1}. {edge.name} @ {distance:.2f} km")
            distance += edge.length_km
        route_key = "route_range_" + "_".join(str(e.eid) for e in candidate_edges)
        if len(marks) == 1:
            st.write(marks[0])
            part = candidate_edges
            lo = hi = 0
        else:
            mid = len(marks) // 2
            default_range = (marks[mid], marks[mid])
            st.caption("💡 **Tip:** By default, 1 road segment in the middle is selected so search algorithms can find a detour around it. Drag the handles to expand the range.")
            low, high = st.select_slider("Part of the route to change (start to destination)",
                                        options=marks, value=default_range, key=route_key)
            lo, hi = marks.index(low), marks.index(high)
            part = candidate_edges[lo:hi + 1]
        label = f"{edit_algorithm}: segments {lo + 1} to {hi + 1}"
    elif source == "Current route" and candidate_edges:
        options = {f"{i + 1}. {e.name} [{e.u} to {e.v}] (eid {e.eid})": e
                   for i, e in enumerate(candidate_edges)}
        selected = st.selectbox("Route edge", list(options))
        part = [options[selected]]
        label = selected
    elif source == "Search by road name":
        query = st.text_input("Road name contains", "")
        matches = [e for e in graph.edges.values()
                   if query and query.casefold() in e.name.casefold()][:200]
        if matches:
            options = {f"{e.name} [{e.u} to {e.v}] (eid {e.eid})": e for e in matches}
            selected = st.selectbox("Matching edge", list(options))
            part, label = [options[selected]], selected
        elif query:
            st.info("No matching named roads.")
    else:
        st.info("Run any algorithm that finds a route first, or search by road name.")

    if part:
        both = st.checkbox("Apply to both directions of the same road", value=True,
                           key="edit_both_directions")
        targets = with_reverse(graph, part) if both else part
        ids = [e.eid for e in targets]
        scope = "_".join(map(str, sorted(ids)))
        # Changing selected roads/revision resets values instead of carrying old slider values.
        widget_scope = scope + "_" + str(st.session_state.scenario_rev)
        st.write(f"**Selected:** {len(part)} segment(s), "
                 f"{sum(e.length_km for e in part):.3f} km; "
                 f"{len(targets)} directed edge(s) will be updated.")
        st.dataframe(pd.DataFrame([{
            "edge": e.eid, "road": e.name, "direction": f"{e.u} to {e.v}",
            "traffic": sc.get(e.eid).traffic, "condition": sc.get(e.eid).condition,
            "safety_day": sc.get(e.eid).safety_day,
            "safety_night": sc.get(e.eid).safety_night,
            "closed": sc.get(e.eid).closed, "incident": sc.get(e.eid).incident,
        } for e in targets]), width="stretch", hide_index=True)

        st.markdown("**Apply individual parameters or several together**")
        fields = st.multiselect("Parameters to change", list(PARAMETERS),
                                 format_func=PARAMETERS.get, default=["traffic"],
                                 key="parameter_fields_" + scope)
        values = {}
        for field in fields:
            key = "road_value_" + field + "_" + widget_scope
            if field == "closed":
                values[field] = st.checkbox("Close selected road",
                    value=all(sc.get(e.eid).closed for e in targets), key=key)
            elif field == "incident":
                values[field] = st.selectbox("Incident label (visual only)",
                                             ["", "flood", "construction", "accident", "other"], key=key)
            else:
                upper = 5.0 if field in {"traffic", "condition"} else 4.0
                average = sum(getattr(sc.get(e.eid), field) for e in targets) / len(targets)
                values[field] = st.slider(PARAMETERS[field], 1.0, upper,
                                          round(average, 1), 0.1, key=key)
        st.caption("Only the selected parameters change; every other value on each road "
                   "stays as it is. Night safety affects cost only when Night-time is on, "
                   "and a parameter with zero weight cannot change the route cost.")
        if st.button("Apply parameters and replan", type="primary", disabled=not fields):
            edit = {"kind": "values", "label": label, "eids": ids, "values": values}
            commit_road_edits(replace_edit(st.session_state.local_edits, edit),
                              "Selected parameters applied.")

        st.markdown("**Or combine road conditions**")
        previous = next((ed for ed in reversed(st.session_state.local_edits)
                         if ed["kind"] == "conditions" and set(ed["eids"]) == set(ids)), None)
        effects = st.multiselect("Conditions on this part", list(LOCAL_EFFECTS),
                                 format_func=LOCAL_EFFECTS.get,
                                 default=previous["effects"] if previous else [],
                                 key="effects_" + widget_scope)
        st.caption("Conditions use this scenario's original values plus the selected effects. "
                   "Untick a condition and apply to remove it. An empty selection removes "
                   "this condition edit. Rain/flood/accident presets change the relevant values; "
                   "deep flood, construction and road closed block the selected road.")
        if st.button("Apply conditions and replan"):
            edit = {"kind": "conditions", "label": label, "eids": ids,
                    "effects": list(effects)}
            commit_road_edits(replace_edit(st.session_state.local_edits, edit),
                              "Road conditions updated.")

    edits = st.session_state.local_edits
    if edits:
        st.markdown("**Your road edits: remove any edit to restore its values**")
        for index, edit in enumerate(edits):
            left, right = st.columns([4, 1])
            detail = (", ".join(LOCAL_EFFECTS[e] for e in edit["effects"])
                      if edit["kind"] == "conditions" else
                      ", ".join(f"{PARAMETERS[k]} = {v}" for k, v in edit["values"].items()))
            left.write(f"{index + 1}. {edit['label']}: {detail}")
            if right.button("Remove", key=f"rm_edit_{index}"):
                commit_road_edits(edits[:index] + edits[index + 1:], "Road edit removed.")

    st.divider()
    st.subheader("Incident: replan from current position (demo)")
    # The vehicle belongs to the latest successful route, not the saved edit reference.
    latest_path = next((current[n].edge_path for n in ("A*", "UCS", "GBFS", "BFS", "DFS", "IDS")
                        if n in current and current[n].found), [])
    if latest_path:
        half = len(latest_path) // 2
        prob_now = build_problem()
        mid = next((latest_path[i] for i in sorted(range(len(latest_path)), key=lambda i: abs(i - half))
                    if len({e.v for e in prob_now.successors(latest_path[i].u)}) >= 2),
                   latest_path[half])
        st.write(f"Vehicle at junction {mid.u}; the road ahead ({mid.name}) closes.")
        if st.button("Block the road ahead and replan from here"):
            st.session_state.start = mid.u
            st.session_state.start_label = ""
            st.session_state.clear_picks = ["pick_start"]
            st.session_state.edit_reference = None
            # The origin changed; a before/after comparison of different trips is invalid.
            st.session_state.results = {}
            edit = {"kind": "values", "label": f"Road ahead: {mid.name}",
                    "eids": [mid.eid], "values": {"closed": True}}
            commit_road_edits(replace_edit(edits, edit), "Road ahead closed; replanned from the vehicle.")
    else:
        st.info("Run an algorithm that finds a route to show the incident demo.")
    st.write("Scenario snapshot stats:")
    st.json(scenario_stats(graph, sc))


# =========================================================================== #
# TAB: ABOUT / VALIDATION
# =========================================================================== #
with tab_about:
    st.subheader("Cost function")
    st.code(explain_example(cfg.weights), language="text")
    st.subheader("Heuristic validation (on the active graph)")
    if st.session_state.goal is None:
        st.info("Select a destination to validate the heuristic toward it.")
    elif st.button("Check heuristic admissibility & consistency"):
        with st.spinner("Checking every edge …"):
            rep = verify_heuristic(graph, st.session_state.goal)
        st.code(rep.summary(), language="text")
    st.subheader("Graph provenance & stats")
    st.json({k: v for k, v in graph.meta.items()
             if k not in ("named_points", "alt_names")})
    st.caption("Source: local OSM XML pulled from OpenStreetMap via the Overpass API.")
