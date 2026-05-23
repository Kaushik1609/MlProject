"""
FastAPI Backend for Dynamic Route Rationalization
==================================================
Serves REST endpoints for:
  - POST /optimize-route   -> best route via Dijkstra + ML congestion
  - GET  /traffic-status   -> live congestion for all segments
  - GET  /all-routes       -> top-3 alternative routes
  - GET  /nodes            -> intersection coordinates
  - GET  /edges            -> road segment metadata

The road network is modelled as an undirected NetworkX graph whose edges
map to ML-predicted congestion scores from the XGBoost model.
"""

import os
import sys
import numpy as np
import networkx as nx
import joblib
from datetime import datetime
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional
from collections import defaultdict

# ---------------------------------------------------------------
# App Initialization
# ---------------------------------------------------------------
app = FastAPI(
    title="Dynamic Route Rationalization API",
    description="AI-powered traffic route optimization for Indian cities",
    version="1.0.0",
)

# Allow the frontend (served from file:// or localhost) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------
# Load ML Artefacts (model.pkl + weather_encoder.pkl)
# ---------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "ml", "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "ml", "weather_encoder.pkl")

try:
    # Load the XGBoost model and weather encoder from separate files
    ml_model = joblib.load(MODEL_PATH)
    weather_encoder = joblib.load(ENCODER_PATH)
    print(f"[OK] ML model loaded from {MODEL_PATH}")
    print(f"[OK] Weather encoder loaded from {ENCODER_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("Run ml/train_model.py first!")
    ml_model = None
    weather_encoder = None

# ---------------------------------------------------------------
# Load RL Q-Table (q_table.pkl)
# ---------------------------------------------------------------
Q_TABLE_PATH = os.path.join(BASE_DIR, "ml", "q_table.pkl")
RL_ENV_AVAILABLE = False

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

try:
    from ml.rl_env import (
        NashikRoutingEnv, NODE_TO_IDX, IDX_TO_NODE,
        ADJACENCY, WEATHER_MAP, time_to_bucket,
    )
    RL_ENV_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] ml.rl_env not importable — RL routing disabled: {e}")

q_table = None
try:
    if os.path.exists(Q_TABLE_PATH):
        q_table = joblib.load(Q_TABLE_PATH)
        print(f"[OK] RL Q-table loaded from {Q_TABLE_PATH}")
    else:
        print(f"[INFO] Q-table not found at {Q_TABLE_PATH} — run ml/train_rl.py")
except Exception as e:
    print(f"[WARN] Failed to load Q-table: {e}")

# ---------------------------------------------------------------
# Nashik City Road Network -- Node Definitions
# Each node is a major intersection with real lat/lng
# ---------------------------------------------------------------
NODES = {
    "CBS Chowk":       {"lat": 20.0059, "lng": 73.7898},
    "Gangapur Road":   {"lat": 20.0219, "lng": 73.7677},
    "Dwarka Circle":   {"lat": 19.9975, "lng": 73.7654},
    "College Road":    {"lat": 20.0110, "lng": 73.7750},
    "Nashik Road":     {"lat": 19.9801, "lng": 73.8309},
    "Indira Nagar":    {"lat": 20.0287, "lng": 73.8012},
    "Panchavati":      {"lat": 20.0113, "lng": 73.7654},
    "Mhasrul":         {"lat": 20.0480, "lng": 73.8012},
    "Satpur":          {"lat": 19.9975, "lng": 73.7450},
    "Ambad":           {"lat": 20.0113, "lng": 73.7450},
    "Cidco":           {"lat": 20.0000, "lng": 73.7550},
    "Nashik Phata":    {"lat": 20.0400, "lng": 73.7750},
    "Canada Corner":   {"lat": 20.0168, "lng": 73.7805},
    "Sharanpur Road":  {"lat": 20.0059, "lng": 73.7700},
    "Mumbai Naka":     {"lat": 19.9900, "lng": 73.7898},
    "Deolali":         {"lat": 19.9500, "lng": 73.8309},
    "Trimbak Road":    {"lat": 20.0350, "lng": 73.7600},
    "Pathardi Phata":  {"lat": 19.9850, "lng": 73.7750},
    "Old Agra Road":   {"lat": 20.0200, "lng": 73.8100},
    "Commissioner Office":   {"lat": 20.0059, "lng": 73.7800},
}

EDGES = [
    ("CBS Chowk",     "College Road",    {"road_name": "MG Road",           "segment_id": 1,  "base_travel_time": 8,  "geometry": [], "landmarks": []}),
    ("CBS Chowk",     "Sharanpur Road",  {"road_name": "Sharanpur Road",    "segment_id": 2,  "base_travel_time": 6,  "geometry": [], "landmarks": []}),
    ("CBS Chowk",     "Canada Corner",   {"road_name": "CBS-Canada Road",   "segment_id": 3,  "base_travel_time": 5,  "geometry": [], "landmarks": []}),
    ("CBS Chowk",     "Mumbai Naka",     {"road_name": "Old Pune Road",     "segment_id": 4,  "base_travel_time": 10, "geometry": [], "landmarks": []}),
    ("CBS Chowk",     "Panchavati",      {"road_name": "Panchavati Road",   "segment_id": 5,  "base_travel_time": 7,  "geometry": [], "landmarks": []}),
    ("College Road",  "Canada Corner",   {"road_name": "College Road",      "segment_id": 6,  "base_travel_time": 5,  "geometry": [], "landmarks": []}),
    ("College Road",  "Gangapur Road",   {"road_name": "Gangapur Road",     "segment_id": 7,  "base_travel_time": 9,  "geometry": [], "landmarks": []}),
    ("College Road",  "Indira Nagar",    {"road_name": "Indira Nagar Road", "segment_id": 8,  "base_travel_time": 8,  "geometry": [], "landmarks": []}),
    ("Canada Corner", "Commissioner Office",   {"road_name": "Commissioner Office Road","segment_id": 9,  "base_travel_time": 4,  "geometry": [], "landmarks": []}),
    ("Canada Corner", "Nashik Phata",    {"road_name": "Nashik Phata Road", "segment_id": 10, "base_travel_time": 12, "geometry": [], "landmarks": []}),
    ("Gangapur Road", "Trimbak Road",    {"road_name": "Trimbak Road",      "segment_id": 11, "base_travel_time": 11, "geometry": [], "landmarks": []}),
    ("Gangapur Road", "Mhasrul",         {"road_name": "Mhasrul Road",      "segment_id": 12, "base_travel_time": 13, "geometry": [], "landmarks": []}),
    ("Gangapur Road", "Nashik Phata",    {"road_name": "Nashik Phata Road", "segment_id": 13, "base_travel_time": 10, "geometry": [], "landmarks": []}),
    ("Dwarka Circle", "Satpur",          {"road_name": "Satpur Road",       "segment_id": 14, "base_travel_time": 9,  "geometry": [], "landmarks": []}),
    ("Dwarka Circle", "Cidco",           {"road_name": "Cidco Road",        "segment_id": 15, "base_travel_time": 7,  "geometry": [], "landmarks": []}),
    ("Dwarka Circle", "Pathardi Phata",  {"road_name": "Pathardi Road",     "segment_id": 16, "base_travel_time": 8,  "geometry": [], "landmarks": []}),
    ("Dwarka Circle", "Sharanpur Road",  {"road_name": "Dwarka-CBS Road",   "segment_id": 17, "base_travel_time": 6,  "geometry": [], "landmarks": []}),
    ("Satpur",        "Ambad",           {"road_name": "Ambad Link Road",   "segment_id": 18, "base_travel_time": 8,  "geometry": [], "landmarks": []}),
    ("Satpur",        "Cidco",           {"road_name": "MIDC Road",         "segment_id": 19, "base_travel_time": 6,  "geometry": [], "landmarks": []}),
    ("Ambad",         "Cidco",           {"road_name": "Industrial Road",   "segment_id": 20, "base_travel_time": 5,  "geometry": [], "landmarks": []}),
    ("Ambad",         "Nashik Phata",    {"road_name": "Ambad-Phata Road",  "segment_id": 21, "base_travel_time": 10, "geometry": [], "landmarks": []}),
    ("Nashik Road",   "Mumbai Naka",     {"road_name": "Nashik-Mumbai Road","segment_id": 22, "base_travel_time": 12, "geometry": [], "landmarks": []}),
    ("Nashik Road",   "Deolali",         {"road_name": "Deolali Road",      "segment_id": 23, "base_travel_time": 14, "geometry": [], "landmarks": []}),
    ("Nashik Road",   "Pathardi Phata",  {"road_name": "Pathardi Road",     "segment_id": 24, "base_travel_time": 8,  "geometry": [], "landmarks": []}),
    ("Indira Nagar",  "Old Agra Road",   {"road_name": "Agra Road",         "segment_id": 25, "base_travel_time": 9,  "geometry": [], "landmarks": []}),
    ("Indira Nagar",  "Mhasrul",         {"road_name": "Mhasrul Link",      "segment_id": 26, "base_travel_time": 11, "geometry": [], "landmarks": []}),
    ("Panchavati",    "Sharanpur Road",  {"road_name": "Sharanpur Road",    "segment_id": 27, "base_travel_time": 5,  "geometry": [], "landmarks": []}),
    ("Panchavati",    "Trimbak Road",    {"road_name": "Trimbak Road",      "segment_id": 28, "base_travel_time": 8,  "geometry": [], "landmarks": []}),
    ("Mumbai Naka",   "Pathardi Phata",  {"road_name": "Pathardi Road",     "segment_id": 29, "base_travel_time": 7,  "geometry": [], "landmarks": []}),
    ("Old Agra Road", "Mhasrul",         {"road_name": "Agra Road",         "segment_id": 30, "base_travel_time": 10, "geometry": [], "landmarks": []}),
]


def build_graph():
    """
    Build an undirected weighted graph representing the Nashik road network.
    Nodes carry lat/lng; edges carry road metadata + travel times.
    """
    G = nx.Graph()
    for node, coords in NODES.items():
        G.add_node(node, **coords)
    for u, v, attrs in EDGES:
        G.add_edge(u, v, **attrs)
    return G


# Global graph instance -- built once at startup
graph = build_graph()
print(f"[OK] Road graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


# ---------------------------------------------------------------
# ML Prediction Helpers
# ---------------------------------------------------------------
def estimate_vehicle_density(time_of_day: int, day_of_week: int) -> int:
    """
    Deterministically estimate vehicle density from time and day.
    Returns a value 0-100 representing percent road capacity used.
    Mirrors the distribution used by the data generator.
    """
    if 8 <= time_of_day <= 10:
        base = 80
    elif 17 <= time_of_day <= 20:
        base = 85
    elif 12 <= time_of_day <= 14:
        base = 55
    elif 6 <= time_of_day <= 7:
        base = 42
    elif 21 <= time_of_day <= 23:
        base = 28
    else:
        base = 15

    # Weekends see ~35% less traffic
    if day_of_week >= 5:
        base = int(base * 0.65)

    return min(max(base, 0), 100)


def predict_congestion(
    segment_id: int, time_of_day: int, day_of_week: int, weather: str
) -> float:
    """
    Predict congestion score for a single road segment using the
    trained XGBoost model.  Returns a float clamped to [0, 1].
    """
    if ml_model is None:
        raise HTTPException(status_code=500, detail="ML model not loaded")

    try:
        weather_encoded = int(weather_encoder.transform([weather])[0])
    except (ValueError, AttributeError):
        weather_encoded = 0  # default to 'clear'

    vehicle_density = estimate_vehicle_density(time_of_day, day_of_week)

    features = np.array(
        [[time_of_day, day_of_week, weather_encoded, vehicle_density]]
    )
    score = float(ml_model.predict(features)[0])
    return round(min(max(score, 0.0), 1.0), 4)


def effective_travel_time(base_time: float, congestion: float) -> float:
    """
    Calculate real-world travel time by inflating the base time
    proportionally to the predicted congestion level.
    Formula: t_eff = base * (1 + congestion)
    """
    return round(base_time * (1.0 + congestion), 2)


# ---------------------------------------------------------------
# Route Optimization Engine
# ---------------------------------------------------------------
def _update_edge_weights(time_of_day: int, day_of_week: int, weather: str):
    """
    Refresh every edge's congestion_score and effective_travel_time
    using the ML model for the given traffic conditions.
    """
    for u, v, data in graph.edges(data=True):
        cong = predict_congestion(
            data["segment_id"], time_of_day, day_of_week, weather
        )
        eff = effective_travel_time(data["base_travel_time"], cong)
        graph[u][v]["congestion_score"] = cong
        graph[u][v]["effective_travel_time"] = eff


def find_optimal_route(
    source: str, destination: str,
    time_of_day: int, day_of_week: int, weather: str,
):
    """
    Find the fastest route from source to destination using
    Dijkstra's algorithm with ML-predicted congestion weights.
    Returns the path, total time, and per-segment breakdown.
    """
    _update_edge_weights(time_of_day, day_of_week, weather)

    try:
        path = nx.dijkstra_path(
            graph, source, destination, weight="effective_travel_time"
        )
    except nx.NetworkXNoPath:
        return None

    total_time = 0.0
    segments = []
    for i in range(len(path) - 1):
        edge = graph[path[i]][path[i + 1]]
        total_time += edge["effective_travel_time"]
        segments.append({
            "from": path[i],
            "to": path[i + 1],
            "road_name": edge["road_name"],
            "segment_id": edge["segment_id"],
            "base_travel_time": edge["base_travel_time"],
            "congestion_score": edge["congestion_score"],
            "effective_travel_time": edge["effective_travel_time"],
            "effective_time": edge["effective_travel_time"],
            "geometry": edge.get("geometry", []),
            "landmarks": edge.get("landmarks", []),
        })

    return {"path": path, "total_time": round(total_time, 2), "segments": segments}


# ---------------------------------------------------------------
# RL-Based Route Optimization
# ---------------------------------------------------------------
def find_optimal_route_rl(
    source: str, destination: str,
    time_of_day: int, day_of_week: int, weather: str,
):
    """
    Find the optimal route using the trained Q-learning agent.
    The agent greedily follows the best Q-values from source to
    destination.
    """
    if q_table is None or not RL_ENV_AVAILABLE:
        return None

    if source not in NODE_TO_IDX or destination not in NODE_TO_IDX:
        return None

    src_idx = NODE_TO_IDX[source]
    dst_idx = NODE_TO_IDX[destination]
    tb = time_to_bucket(time_of_day)
    wi = WEATHER_MAP.get(weather, 0)

    # Refresh edge weights for travel time computation
    _update_edge_weights(time_of_day, day_of_week, weather)

    current = src_idx
    path = [source]
    total_time = 0.0
    segments = []
    visited = {src_idx}
    max_steps = 20

    for _ in range(max_steps):
        if current == dst_idx:
            break

        state = (current, dst_idx, tb, wi)
        neighbours = ADJACENCY[current]
        n_actions = len(neighbours)

        if n_actions == 0:
            break

        # Greedy action selection from Q-table
        q_vals = [
            q_table.get(state, {}).get(a, 0.0)
            for a in range(n_actions)
        ]
        action = int(np.argmax(q_vals))

        next_node, seg_id, base_time = neighbours[action]

        # Look up edge in the graph for travel time info
        from_name = IDX_TO_NODE[current]
        to_name = IDX_TO_NODE[next_node]

        # Get the edge data from the NetworkX graph
        if graph.has_edge(from_name, to_name):
            edge = graph[from_name][to_name]
        else:
            edge = {
                "road_name": f"Road {seg_id}",
                "segment_id": seg_id,
                "base_travel_time": base_time,
                "congestion_score": 0.3,
                "effective_travel_time": base_time * 1.3,
            }

        eff_time = edge.get("effective_travel_time", base_time * 1.3)
        cong = edge.get("congestion_score", 0.3)
        total_time += eff_time

        segments.append({
            "from": from_name,
            "to": to_name,
            "road_name": edge.get("road_name", f"Road {seg_id}"),
            "segment_id": edge.get("segment_id", seg_id),
            "base_travel_time": edge.get("base_travel_time", base_time),
            "congestion_score": cong,
            "effective_travel_time": eff_time,
            "effective_time": eff_time,
            "geometry": edge.get("geometry", []),
            "landmarks": edge.get("landmarks", []),
        })

        path.append(to_name)
        visited.add(next_node)
        current = next_node

    # Check if destination was reached
    if current != dst_idx:
        return None

    return {
        "path": path,
        "total_time": round(total_time, 2),
        "segments": segments,
        "method": "reinforcement_learning",
    }


def find_k_shortest_routes(
    source: str, destination: str,
    time_of_day: int, day_of_week: int, weather: str,
    k: int = 3,
):
    """
    Enumerate the top-k shortest simple paths (by effective travel time)
    using NetworkX's Yen's algorithm implementation.
    """
    _update_edge_weights(time_of_day, day_of_week, weather)

    routes = []
    try:
        for rank, path in enumerate(
            nx.shortest_simple_paths(
                graph, source, destination, weight="effective_travel_time"
            ),
            start=1,
        ):
            if rank > k:
                break
            total_time = 0.0
            segments = []
            for i in range(len(path) - 1):
                edge = graph[path[i]][path[i + 1]]
                total_time += edge["effective_travel_time"]
                segments.append({
                    "from": path[i],
                    "to": path[i + 1],
                    "road_name": edge["road_name"],
                    "segment_id": edge["segment_id"],
                    "congestion_score": edge["congestion_score"],
                    "effective_travel_time": edge["effective_travel_time"],
                    "effective_time": edge["effective_travel_time"],
                    "geometry": edge.get("geometry", []),
                    "landmarks": edge.get("landmarks", []),
                })
            routes.append({
                "rank": rank,
                "path": path,
                "total_time": round(total_time, 2),
                "segments": segments,
                "road_names": [seg["road_name"] for seg in segments],
            })
    except nx.NetworkXNoPath:
        pass

    return routes


# ---------------------------------------------------------------
# Pydantic Request Model
# ---------------------------------------------------------------
class RouteRequest(BaseModel):
    """Request body for the /optimize-route endpoint."""
    source: str = Field(..., description="Starting intersection name")
    destination: str = Field(..., description="Ending intersection name")
    time_of_day: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Mon, 6=Sun)")
    weather: str = Field(..., description="Weather: clear, rain, or fog")
    method: str = Field("dijkstra", description="Routing method: 'dijkstra' or 'rl'")


# ---------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------
@app.post("/optimize-route")
def optimize_route_endpoint(req: RouteRequest):
    """
    Find the single best route between two intersections.
    Supports two methods:
      - 'dijkstra' (default): Dijkstra over ML-predicted congestion weights
      - 'rl': Reinforcement Learning agent (Q-learning)
    """
    if req.source not in NODES or req.destination not in NODES:
        return {"error": "Invalid source or destination node"}
    if req.source == req.destination:
        return {"error": "Source and destination must be different"}

    result = None
    method_used = req.method.lower()

    if method_used == "rl":
        result = find_optimal_route_rl(
            req.source, req.destination,
            req.time_of_day, req.day_of_week, req.weather,
        )
        if result is None:
            # Fall back to Dijkstra if RL fails
            result = find_optimal_route(
                req.source, req.destination,
                req.time_of_day, req.day_of_week, req.weather,
            )
            if result:
                result["method"] = "dijkstra_fallback"
    else:
        result = find_optimal_route(
            req.source, req.destination,
            req.time_of_day, req.day_of_week, req.weather,
        )
        if result:
            result["method"] = "dijkstra"

    if result is None:
        return {"error": "No path found between the selected nodes"}
    return result


@app.get("/traffic-status")
def traffic_status(
    time_of_day: Optional[int] = None,
    day_of_week: Optional[int] = None,
    weather: str = "clear",
):
    """
    Return the predicted congestion score for every road segment.
    Defaults to the current system time if no params are supplied.
    """
    now = datetime.now()
    tod = time_of_day if time_of_day is not None else now.hour
    dow = day_of_week if day_of_week is not None else now.weekday()

    _update_edge_weights(tod, dow, weather)

    segments = []
    for u, v, data in graph.edges(data=True):
        segments.append({
            "from": u,
            "to": v,
            "road_name": data["road_name"],
            "segment_id": data["segment_id"],
            "base_travel_time": data["base_travel_time"],
            "congestion_score": data["congestion_score"],
            "effective_travel_time": data["effective_travel_time"],
            "from_coords": [NODES[u]["lat"], NODES[u]["lng"]],
            "to_coords":   [NODES[v]["lat"], NODES[v]["lng"]],
            "geometry": data.get("geometry", []),
            "landmarks": data.get("landmarks", []),
        })

    return {
        "timestamp": now.isoformat(),
        "time_of_day": tod,
        "day_of_week": dow,
        "weather": weather,
        "segments": segments,
        "traffic": segments,
    }


@app.get("/all-routes")
def all_routes(
    source: str = Query(..., description="Starting intersection"),
    destination: str = Query(..., description="Ending intersection"),
    time_of_day: int = Query(12, ge=0, le=23),
    day_of_week: int = Query(0, ge=0, le=6),
    weather: str = Query("clear"),
):
    """
    Return the top-3 alternative routes ranked by estimated travel time.
    """
    if source not in NODES or destination not in NODES:
        return {"error": "Invalid source or destination node"}

    routes = find_k_shortest_routes(
        source, destination, time_of_day, day_of_week, weather, k=3
    )
    return {"routes": routes}


@app.get("/nodes")
def get_nodes():
    """Return all intersection nodes with their GPS coordinates."""
    return {"nodes": NODES}


@app.get("/rl-status")
def rl_status():
    """Check if the RL agent is loaded and available."""
    return {
        "rl_available": q_table is not None and RL_ENV_AVAILABLE,
        "q_table_loaded": q_table is not None,
        "q_table_entries": len(q_table) if q_table else 0,
        "rl_env_available": RL_ENV_AVAILABLE,
    }


import math

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.get("/nearest-node")
def get_nearest_node(lat: float, lng: float):
    """Find the closest node to the given coordinates."""
    nearest = None
    min_dist = float('inf')
    
    for name, coords in NODES.items():
        dist = haversine(lat, lng, coords["lat"], coords["lng"])
        if dist < min_dist:
            min_dist = dist
            nearest = name
            
    if nearest:
        return {
            "nearest_node": nearest,
            "distance_meters": int(min_dist),
            "lat": NODES[nearest]["lat"],
            "lng": NODES[nearest]["lng"]
        }
    return {"error": "No nodes available"}


@app.get("/edges")
def get_edges():
    """Return all road segments with metadata and endpoint coordinates."""
    edges_out = []
    for u, v, data in graph.edges(data=True):
        edges_out.append({
            "from": u,
            "to": v,
            "from_coords": [NODES[u]["lat"], NODES[u]["lng"]],
            "to_coords":   [NODES[v]["lat"], NODES[v]["lng"]],
            "road_name": data.get("road_name"),
            "segment_id": data.get("segment_id"),
            "base_travel_time": data.get("base_travel_time"),
            "geometry": data.get("geometry", []),
            "landmarks": data.get("landmarks", []),
        })
    return {"edges": edges_out}


@app.get("/graph-data")
def get_graph_data():
    """Return all nodes and edges in a single structured JSON."""
    nodes_out = []
    for name, coords in NODES.items():
        nodes_out.append({
            "name": name,
            "lat": coords["lat"],
            "lon": coords["lng"],
        })
    
    edges_out = []
    for u, v, data in graph.edges(data=True):
        edges_out.append({
            "from": u,
            "to": v,
            "segment_id": data.get("segment_id"),
            "road_name": data.get("road_name"),
            "base_travel_time": data.get("base_travel_time"),
            "geometry": data.get("geometry", []),
            "landmarks": data.get("landmarks", []),
        })
    
    return {"nodes": nodes_out, "edges": edges_out}


# ---------------------------------------------------------------
# Serve frontend HTML (convenience -- open http://localhost:8000)
# ---------------------------------------------------------------
@app.get("/")
def serve_frontend():
    """Serve the frontend dashboard if opened in a browser."""
    html_path = os.path.join(BASE_DIR, "frontend", "map.html")
    html_path = os.path.abspath(html_path)
    if os.path.exists(html_path):
        return FileResponse(html_path, media_type="text/html")
    return {"message": "API is running. Open frontend/map.html in a browser."}


@app.get("/style.css")
def serve_style():
    """Serve the stylesheet for the dashboard."""
    style_path = os.path.join(BASE_DIR, "frontend", "style.css")
    style_path = os.path.abspath(style_path)
    if os.path.exists(style_path):
        return FileResponse(style_path, media_type="text/css")
    raise HTTPException(status_code=404, detail="style.css not found")


# ---------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("  Dynamic Route Rationalization -- API Server")
    print("  http://localhost:8000")
    print("  http://localhost:8000/docs  (Swagger UI)")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
