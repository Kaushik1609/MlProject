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
import numpy as np
import networkx as nx
import joblib
from datetime import datetime
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import List, Optional

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
# Pune City Road Network -- Node Definitions
# Each node is a major intersection with real lat/lng
# ---------------------------------------------------------------
NODES = {
    "Shivajinagar":     {"lat": 18.5308, "lng": 73.8474},
    "Katraj":           {"lat": 18.4529, "lng": 73.8654},
    "Hinjewadi":        {"lat": 18.5912, "lng": 73.7389},
    "Wakad":            {"lat": 18.5997, "lng": 73.7601},
    "Kothrud":          {"lat": 18.5074, "lng": 73.8077},
    "Deccan":           {"lat": 18.5161, "lng": 73.8397},
    "Swargate":         {"lat": 18.5018, "lng": 73.8553},
    "Hadapsar":         {"lat": 18.5018, "lng": 73.9260},
    "Viman Nagar":      {"lat": 18.5679, "lng": 73.9143},
    "Magarpatta":       {"lat": 18.5117, "lng": 73.9280},
    "FC Road":          {"lat": 18.5264, "lng": 73.8400},
    "Baner":            {"lat": 18.5590, "lng": 73.7868},
    "Pune Station":     {"lat": 18.5284, "lng": 73.8742},
    "Pimpri":           {"lat": 18.6279, "lng": 73.7997},
    "Chinchwad":        {"lat": 18.6481, "lng": 73.7946},
}

EDGES = [
    ("Hinjewadi", "Wakad", {"road_name": "Hinjewadi Link Rd", "segment_id": 1, "base_travel_time": 6, "geometry": [[18.5912, 73.7389], [18.5935, 73.7460], [18.5960, 73.7530], [18.5997, 73.7601]], "landmarks": [{"name": "Hinjewadi Phase 1", "lat": 18.5925, "lng": 73.7425}, {"name": "Wipro Circle", "lat": 18.5955, "lng": 73.7510}]}),
    ("Wakad", "Baner", {"road_name": "Katraj Bypass Rd", "segment_id": 2, "base_travel_time": 8, "geometry": [[18.5997, 73.7601], [18.5870, 73.7570], [18.5740, 73.7650], [18.5660, 73.7740], [18.5590, 73.7868]], "landmarks": [{"name": "Balewadi Stadium", "lat": 18.5775, "lng": 73.7620}, {"name": "Radisson Corridor", "lat": 18.5680, "lng": 73.7710}]}),
    ("Wakad", "Pimpri", {"road_name": "Kaspate Wasti Rd", "segment_id": 3, "base_travel_time": 10, "geometry": [[18.5997, 73.7601], [18.6110, 73.7750], [18.6200, 73.7890], [18.6279, 73.7997]], "landmarks": [{"name": "Dange Chowk", "lat": 18.6080, "lng": 73.7720}, {"name": "Thergaon", "lat": 18.6185, "lng": 73.7850}]}),
    ("Pimpri", "Chinchwad", {"road_name": "Old Mumbai Hwy", "segment_id": 4, "base_travel_time": 5, "geometry": [[18.6279, 73.7997], [18.6360, 73.7960], [18.6430, 73.7930], [18.6481, 73.7946]], "landmarks": [{"name": "PCMC HQ", "lat": 18.6320, "lng": 73.7975}, {"name": "Chinchwad Station", "lat": 18.6410, "lng": 73.7940}]}),
    ("Baner", "Shivajinagar", {"road_name": "Baner Road", "segment_id": 5, "base_travel_time": 12, "geometry": [[18.5590, 73.7868], [18.5490, 73.8050], [18.5390, 73.8240], [18.5330, 73.8360], [18.5308, 73.8474]], "landmarks": [{"name": "Baner Phata", "lat": 18.5470, "lng": 73.8110}, {"name": "SPPU Campus", "lat": 18.5360, "lng": 73.8300}]}),
    ("Hinjewadi", "Kothrud", {"road_name": "Mulshi Bypass", "segment_id": 6, "base_travel_time": 18, "geometry": [[18.5912, 73.7389], [18.5710, 73.7420], [18.5430, 73.7610], [18.5210, 73.7780], [18.5074, 73.8077]], "landmarks": [{"name": "Bavdhan", "lat": 18.5480, "lng": 73.7580}, {"name": "Chandani Chowk", "lat": 18.5170, "lng": 73.7820}]}),
    ("Kothrud", "Deccan", {"road_name": "Karve Road", "segment_id": 7, "base_travel_time": 7, "geometry": [[18.5074, 73.8077], [18.5090, 73.8190], [18.5125, 73.8290], [18.5161, 73.8397]], "landmarks": [{"name": "Nal Stop", "lat": 18.5085, "lng": 73.8210}, {"name": "SNDT College", "lat": 18.5110, "lng": 73.8280}]}),
    ("Deccan", "FC Road", {"road_name": "FC Road Link", "segment_id": 8, "base_travel_time": 3, "geometry": [[18.5161, 73.8397], [18.5210, 73.8405], [18.5264, 73.8400]], "landmarks": [{"name": "Goodluck Cafe", "lat": 18.5185, "lng": 73.8402}]}),
    ("FC Road", "Shivajinagar", {"road_name": "FC Road Upper", "segment_id": 9, "base_travel_time": 4, "geometry": [[18.5264, 73.8400], [18.5290, 73.8430], [18.5308, 73.8474]], "landmarks": [{"name": "Fergusson College", "lat": 18.5280, "lng": 73.8415}]}),
    ("Deccan", "Swargate", {"road_name": "Tilak Road", "segment_id": 10, "base_travel_time": 8, "geometry": [[18.5161, 73.8397], [18.5120, 73.8460], [18.5070, 73.8510], [18.5018, 73.8553]], "landmarks": [{"name": "SP College", "lat": 18.5095, "lng": 73.8480}, {"name": "Alka Talkies", "lat": 18.5135, "lng": 73.8425}]}),
    ("Kothrud", "Swargate", {"road_name": "Sinhagad Road", "segment_id": 11, "base_travel_time": 10, "geometry": [[18.5074, 73.8077], [18.4980, 73.8150], [18.4930, 73.8320], [18.5018, 73.8553]], "landmarks": [{"name": "Rajaram Bridge", "lat": 18.4975, "lng": 73.8220}, {"name": "Anand Nagar", "lat": 18.4940, "lng": 73.8300}]}),
    ("Swargate", "Katraj", {"road_name": "Satara Road", "segment_id": 12, "base_travel_time": 12, "geometry": [[18.5018, 73.8553], [18.4870, 73.8580], [18.4720, 73.8610], [18.4529, 73.8654]], "landmarks": [{"name": "Dhankawadi", "lat": 18.4790, "lng": 73.8595}, {"name": "Balaji Nagar", "lat": 18.4680, "lng": 73.8620}]}),
    ("Shivajinagar", "Pune Station", {"road_name": "University Road", "segment_id": 13, "base_travel_time": 5, "geometry": [[18.5308, 73.8474], [18.5320, 73.8560], [18.5310, 73.8660], [18.5284, 73.8742]], "landmarks": [{"name": "Sancheti Hospital", "lat": 18.5315, "lng": 73.8510}, {"name": "RTO Office", "lat": 18.5295, "lng": 73.8690}]}),
    ("Pune Station", "Viman Nagar", {"road_name": "Nagar Road", "segment_id": 14, "base_travel_time": 14, "geometry": [[18.5284, 73.8742], [18.5410, 73.8880], [18.5520, 73.9010], [18.5679, 73.9143]], "landmarks": [{"name": "Bund Garden", "lat": 18.5440, "lng": 73.8920}, {"name": "Kalyani Nagar", "lat": 18.5580, "lng": 73.9060}]}),
    ("Pune Station", "Swargate", {"road_name": "Station Link", "segment_id": 15, "base_travel_time": 9, "geometry": [[18.5284, 73.8742], [18.5200, 73.8650], [18.5110, 73.8580], [18.5018, 73.8553]], "landmarks": [{"name": "Nana Peth", "lat": 18.5160, "lng": 73.8615}, {"name": "Seven Loves Chowk", "lat": 18.5080, "lng": 73.8570}]}),
    ("Swargate", "Hadapsar", {"road_name": "Solapur Road", "segment_id": 16, "base_travel_time": 15, "geometry": [[18.5018, 73.8553], [18.4980, 73.8740], [18.4960, 73.8960], [18.5018, 73.9260]], "landmarks": [{"name": "Fatima Nagar", "lat": 18.4990, "lng": 73.8820}, {"name": "Race Course", "lat": 18.4970, "lng": 73.9080}]}),
    ("Hadapsar", "Magarpatta", {"road_name": "Magarpatta Road", "segment_id": 17, "base_travel_time": 4, "geometry": [[18.5018, 73.9260], [18.5070, 73.9270], [18.5117, 73.9280]], "landmarks": [{"name": "Gliding Centre", "lat": 18.5050, "lng": 73.9265}]}),
    ("Magarpatta", "Viman Nagar", {"road_name": "Kharadi Bypass", "segment_id": 18, "base_travel_time": 10, "geometry": [[18.5117, 73.9280], [18.5320, 73.9290], [18.5510, 73.9240], [18.5679, 73.9143]], "landmarks": [{"name": "Kharadi Bypass", "lat": 18.5420, "lng": 73.9260}, {"name": "Amanora Park", "lat": 18.5230, "lng": 73.9285}]}),
    ("Pune Station", "Magarpatta", {"road_name": "Prince Road", "segment_id": 19, "base_travel_time": 11, "geometry": [[18.5284, 73.8742], [18.5210, 73.8910], [18.5140, 73.9100], [18.5117, 73.9280]], "landmarks": [{"name": "Empress Garden", "lat": 18.5180, "lng": 73.8990}]}),
    ("Shivajinagar", "Deccan", {"road_name": "JM Road", "segment_id": 20, "base_travel_time": 6, "geometry": [[18.5308, 73.8474], [18.5220, 73.8430], [18.5161, 73.8397]], "landmarks": [{"name": "JM Temple", "lat": 18.5240, "lng": 73.8445}]}),
]


def build_graph():
    """
    Build an undirected weighted graph representing the Pune road network.
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


# ---------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------
@app.post("/optimize-route")
def optimize_route_endpoint(req: RouteRequest):
    """
    Find the single best route between two intersections.
    Uses Dijkstra over ML-predicted congestion weights.
    """
    if req.source not in NODES or req.destination not in NODES:
        return {"error": "Invalid source or destination node"}
    if req.source == req.destination:
        return {"error": "Source and destination must be different"}

    result = find_optimal_route(
        req.source, req.destination,
        req.time_of_day, req.day_of_week, req.weather,
    )
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
