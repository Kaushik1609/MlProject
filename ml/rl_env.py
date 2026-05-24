"""
Reinforcement Learning Environment for Nashik Road Network
=============================================================
Models the Nashik road network as a Markov Decision Process (MDP).

State  : (current_node, destination_node, time_bucket, weather_idx)
Action : Index into the list of neighbours of the current node
Reward : Negative effective travel time for traversing the chosen edge
         +100 bonus for reaching the destination
         -50 penalty if the agent exhausts max steps without arriving

The environment uses the trained XGBoost model to predict congestion
scores, which determine the effective travel time on each edge.
"""

import os
import numpy as np
import joblib


# ---------------------------------------------------------------
# Nashik Road Network Definition (mirrors api/main.py)
# ---------------------------------------------------------------
NODES_LIST = [
    # Corridor 1 nodes (Gangapur → Nandur Naka)
    "Gangapur", "Jalapur", "Ganpat Nagar", "Kale Nagar",
    "Jehan Circle", "Thatte Nagar", "Panchavati", "Shivaji Nagar",
    "CBS", "Sharda Circle", "Dwarka Circle", "Gayatri Nagar",
    "Nandur Naka",
    # Corridor 2 nodes (Western branch + southern arc)
    "Dhruv Nagar", "Shramik Nagar", "Mahindra", "Shaneshwar Nagar",
    "Satpur Colony", "MIDC", "ABB Circle", "Parijat Nagar",
    "MICO Circle", "Mumbai Naka", "Garware",
    # Feeder route nodes
    "Samta Nagar", "Gandhi Nagar", "Nehru Nagar", "Datta Mandir",
    "Nashik Road Rly Stn",
    # Additional retained nodes
    "College Road", "Nashik Phata", "Deolali",
]

NODE_TO_IDX = {name: i for i, name in enumerate(NODES_LIST)}
IDX_TO_NODE = {i: name for i, name in enumerate(NODES_LIST)}

EDGES = [
    # Corridor 1 — Gangapur to Nandur Naka (11 edges)
    ("Gangapur",        "Jalapur",               1,  10),
    ("Jalapur",         "Ganpat Nagar",           2,  8),
    ("Ganpat Nagar",    "Kale Nagar",             3,  6),
    ("Kale Nagar",      "Jehan Circle",           4,  7),
    ("Jehan Circle",    "Thatte Nagar",           5,  5),
    ("Thatte Nagar",    "Panchavati",             6,  6),
    ("Panchavati",      "CBS",                    7,  8),
    ("CBS",             "Sharda Circle",          8,  5),
    ("Sharda Circle",   "Dwarka Circle",          9,  7),
    ("Dwarka Circle",   "Gayatri Nagar",         10,  8),
    ("Gayatri Nagar",   "Nandur Naka",           11, 10),
    # Corridor 2 — Western branch (5 edges)
    ("Gangapur",        "Dhruv Nagar",           12,  9),
    ("Dhruv Nagar",     "Shramik Nagar",         13,  8),
    ("Shramik Nagar",   "Mahindra",              14,  7),
    ("Mahindra",        "Shaneshwar Nagar",      15,  5),
    ("Shaneshwar Nagar","Satpur Colony",         16,  6),
    # Corridor 2 — Southern arc (7 edges)
    ("Satpur Colony",   "MIDC",                  17,  7),
    ("MIDC",            "ABB Circle",            18,  6),
    ("ABB Circle",      "Parijat Nagar",         19,  8),
    ("Parijat Nagar",   "MICO Circle",           20,  7),
    ("MICO Circle",     "Mumbai Naka",           21,  6),
    ("Mumbai Naka",     "CBS",                   22,  7),
    ("Satpur Colony",   "Garware",               23, 12),
    # Feeder route (5 edges)
    ("Dwarka Circle",   "Samta Nagar",           24,  7),
    ("Samta Nagar",     "Gandhi Nagar",          25,  6),
    ("Gandhi Nagar",    "Nehru Nagar",           26,  7),
    ("Nehru Nagar",     "Datta Mandir",          27,  5),
    ("Datta Mandir",    "Nashik Road Rly Stn",   28,  6),
    # Cross-links (22 edges)
    ("CBS",             "College Road",          29,  6),
    ("College Road",    "Gangapur",              30,  8),
    ("CBS",             "Shivaji Nagar",         31,  5),
    ("Shivaji Nagar",   "MICO Circle",           32,  6),
    ("Shivaji Nagar",   "Sharda Circle",         33,  5),
    ("College Road",    "Panchavati",            34,  5),
    ("College Road",    "Nashik Phata",          35, 10),
    ("Gangapur",        "Nashik Phata",          36, 12),
    ("Nashik Road Rly Stn", "Deolali",           37, 12),
    ("Mumbai Naka",     "Sharda Circle",         38,  5),
    ("Dwarka Circle",   "Parijat Nagar",         39,  9),
    ("MIDC",            "Garware",               40, 10),
    ("Shaneshwar Nagar","MIDC",                  41,  6),
    ("Kale Nagar",      "ABB Circle",            42,  8),
    ("Jehan Circle",    "Parijat Nagar",         43,  7),
    ("Gayatri Nagar",   "Samta Nagar",           44,  7),
    ("Nandur Naka",     "Nashik Road Rly Stn",   45, 11),
    ("Mumbai Naka",     "Gandhi Nagar",          46,  8),
    ("Shramik Nagar",   "Shaneshwar Nagar",      47,  7),
    ("College Road",    "Shivaji Nagar",         48,  6),
    ("Dwarka Circle",   "Mumbai Naka",           49,  8),
    ("Deolali",         "Samta Nagar",           50,  9),
]

# Build adjacency list:  node_idx -> [(neighbour_idx, segment_id, base_travel_time), ...]
ADJACENCY = {i: [] for i in range(len(NODES_LIST))}
for u_name, v_name, seg_id, btt in EDGES:
    u = NODE_TO_IDX[u_name]
    v = NODE_TO_IDX[v_name]
    ADJACENCY[u].append((v, seg_id, btt))
    ADJACENCY[v].append((u, seg_id, btt))

# Time buckets for discretized state
TIME_BUCKETS = [
    (0, 5),    # night
    (6, 7),    # early morning
    (8, 10),   # morning rush
    (11, 14),  # midday
    (15, 16),  # afternoon
    (17, 20),  # evening rush
    (21, 23),  # late evening
]

WEATHER_MAP = {"clear": 0, "rain": 1, "fog": 2}
WEATHER_LIST = ["clear", "rain", "fog"]


def time_to_bucket(hour: int) -> int:
    """Convert hour (0-23) to a bucket index (0-6)."""
    for idx, (lo, hi) in enumerate(TIME_BUCKETS):
        if lo <= hour <= hi:
            return idx
    return 0


# ---------------------------------------------------------------
# Vehicle density estimator (mirrors api/main.py)
# ---------------------------------------------------------------
def estimate_vehicle_density(time_of_day: int, day_of_week: int) -> int:
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
    if day_of_week >= 5:
        base = int(base * 0.65)
    return min(max(base, 0), 100)


# ---------------------------------------------------------------
# RL Environment
# ---------------------------------------------------------------
class NashikRoutingEnv:
    """
    Gym-style RL environment for route optimisation on the Nashik
    road network.

    Attributes
    ----------
    n_nodes        : int – number of intersection nodes
    max_neighbours : int – maximum degree of any node (for action space)
    ml_model       : trained XGBoost regressor (or None for heuristic)
    weather_encoder: fitted LabelEncoder (or None)
    """

    def __init__(self, ml_model=None, weather_encoder=None, max_steps=20):
        self.n_nodes = len(NODES_LIST)
        self.max_steps = max_steps
        self.ml_model = ml_model
        self.weather_encoder = weather_encoder

        # Maximum number of neighbours any node has (= action space size)
        self.max_neighbours = max(len(ADJACENCY[i]) for i in range(self.n_nodes))

        # State components
        self.current_node = 0
        self.destination = 0
        self.time_of_day = 8
        self.day_of_week = 0
        self.weather = "clear"
        self.time_bucket = 0
        self.weather_idx = 0
        self.steps = 0
        self.visited = set()

    # ---- state representation ----
    def _state_key(self):
        """Return a hashable state tuple for Q-table indexing."""
        return (self.current_node, self.destination,
                self.time_bucket, self.weather_idx)

    # ---- reset ----
    def reset(self, source=None, destination=None,
              time_of_day=None, day_of_week=None, weather=None):
        """
        Reset the environment with random or specified parameters.
        Returns the initial state key.
        """
        if source is None:
            source = np.random.randint(0, self.n_nodes)
        if destination is None:
            destination = np.random.randint(0, self.n_nodes)
            while destination == source:
                destination = np.random.randint(0, self.n_nodes)
        if time_of_day is None:
            time_of_day = np.random.randint(0, 24)
        if day_of_week is None:
            day_of_week = np.random.randint(0, 7)
        if weather is None:
            weather = np.random.choice(WEATHER_LIST)

        self.current_node = source
        self.destination = destination
        self.time_of_day = time_of_day
        self.day_of_week = day_of_week
        self.weather = weather
        self.time_bucket = time_to_bucket(time_of_day)
        self.weather_idx = WEATHER_MAP.get(weather, 0)
        self.steps = 0
        self.visited = {source}

        return self._state_key()

    # ---- congestion prediction ----
    def _predict_congestion(self, segment_id: int) -> float:
        """
        Predict congestion using the XGBoost model if available,
        otherwise fall back to a deterministic heuristic.
        """
        if self.ml_model is not None and self.weather_encoder is not None:
            try:
                weather_enc = int(
                    self.weather_encoder.transform([self.weather])[0]
                )
            except (ValueError, AttributeError):
                weather_enc = 0

            density = estimate_vehicle_density(self.time_of_day, self.day_of_week)
            features = np.array(
                [[self.time_of_day, self.day_of_week, weather_enc, density]]
            )
            score = float(self.ml_model.predict(features)[0])
            return min(max(score, 0.0), 1.0)

        # Heuristic fallback
        if 8 <= self.time_of_day <= 10:
            base = 0.7
        elif 17 <= self.time_of_day <= 20:
            base = 0.75
        elif 12 <= self.time_of_day <= 14:
            base = 0.45
        else:
            base = 0.2
        if self.day_of_week >= 5:
            base *= 0.65
        if self.weather == "rain":
            base = min(1.0, base * 1.35)
        elif self.weather == "fog":
            base = min(1.0, base * 1.2)
        return base

    # ---- step ----
    def step(self, action_idx: int):
        """
        Take an action (move to a neighbour).

        Parameters
        ----------
        action_idx : int – index into the neighbours list of current node

        Returns
        -------
        next_state : tuple – new state key
        reward     : float – negative travel time (+ bonus/penalty)
        done       : bool  – True if destination reached or max steps exceeded
        info       : dict  – extra debug info
        """
        neighbours = ADJACENCY[self.current_node]

        # Clamp invalid actions to valid range
        if action_idx >= len(neighbours):
            action_idx = action_idx % len(neighbours)

        next_node, seg_id, base_time = neighbours[action_idx]

        # Compute congestion and effective travel time
        congestion = self._predict_congestion(seg_id)
        eff_time = base_time * (1.0 + congestion)

        # Reward = negative travel time
        reward = -eff_time

        # Penalty for revisiting nodes (discourages loops)
        if next_node in self.visited:
            reward -= 5.0

        self.visited.add(next_node)
        self.current_node = next_node
        self.steps += 1

        done = False
        info = {
            "segment_id": seg_id,
            "congestion": congestion,
            "eff_time": eff_time,
            "node": IDX_TO_NODE[next_node],
        }

        # Check destination reached
        if self.current_node == self.destination:
            reward += 100.0  # Large bonus for reaching destination
            done = True

        # Check max steps exceeded
        elif self.steps >= self.max_steps:
            reward -= 50.0   # Penalty for failing to reach destination
            done = True

        return self._state_key(), reward, done, info

    # ---- helpers ----
    def get_valid_actions(self):
        """Return number of valid actions (neighbours) for the current node."""
        return len(ADJACENCY[self.current_node])

    def get_neighbours(self):
        """Return list of (neighbour_idx, seg_id, base_time)."""
        return ADJACENCY[self.current_node]
