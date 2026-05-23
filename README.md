# Dynamic Route Rationalization Model 

> AI-powered traffic route optimization using Machine Learning, Reinforcement Learning, and real-time road parameters for the Nashik city road network.

## Project Structure

```
SIH1617/
├── data/
│   ├── generate_data.py      # Traffic data simulation script
│   └── road_data.csv          # Generated traffic dataset (8000 rows)
├── ml/
│   ├── train_model.py         # XGBoost model training script
│   ├── model.pkl              # Saved trained XGBoost model
│   ├── weather_encoder.pkl    # Fitted LabelEncoder for weather
│   ├── rl_env.py              # Reinforcement Learning environment
│   ├── train_rl.py            # Q-Learning agent training script
│   └── q_table.pkl            # Saved trained Q-table (after RL training)
├── api/
│   └── main.py                # FastAPI backend server
├── frontend/
│   ├── map.html               # Interactive map dashboard
│   └── style.css              # Dashboard styles
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Tech Stack

| Component    | Technology                          |
|-------------|-------------------------------------|
| ML Model    | XGBoost, scikit-learn, joblib       |
| RL Agent    | Q-Learning (custom environment)     |
| Backend     | FastAPI, Uvicorn                    |
| Routing     | NetworkX (Dijkstra) + RL (Q-table) |
| Data        | Pandas, NumPy                       |
| Frontend    | Leaflet.js, HTML/CSS/JS             |
| Map Tiles   | CartoDB Dark Matter (free, no key)  |

## Setup & Run

### Step 1: Install Dependencies

```bash
cd SIH1617
pip install -r requirements.txt
```

### Step 2: Generate Traffic Data

```bash
python data/generate_data.py
```

This creates `data/road_data.csv` with 8000 simulated traffic records for Nashik roads.

### Step 3: Train the XGBoost ML Model

```bash
python ml/train_model.py
```

This trains an XGBoost model and saves it as `ml/model.pkl`. It will print R² score and RMSE.

### Step 4: Train the Reinforcement Learning Agent

```bash
python ml/train_rl.py
```

This trains a Q-learning agent over 50,000 episodes on the Nashik road network and saves the Q-table as `ml/q_table.pkl`. The agent learns optimal routing policies by exploring different routes under various traffic conditions.

### Step 5: Start the API Server

```bash
python api/main.py
```

or

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will run on `http://localhost:8000`. API docs available at `http://localhost:8000/docs`.

### Step 6: Open the Dashboard

Open `frontend/map.html` in your web browser (or navigate to `http://localhost:8000`). The dashboard connects to the API at `localhost:8000`.

Use the **Algorithm** dropdown in the top-right controls to switch between **Dijkstra** and **Q-Learning RL** routing methods.

## API Endpoints

| Method | Endpoint           | Description                                  |
|--------|-------------------|----------------------------------------------|
| GET    | `/`               | Serve the frontend dashboard                 |
| GET    | `/graph-data`     | Returns road graph nodes and edges           |
| POST   | `/optimize-route` | Find optimal route (Dijkstra or RL)          |
| GET    | `/traffic-status` | Real-time congestion for all road segments   |
| GET    | `/all-routes`     | Top 3 alternative routes with travel times   |
| GET    | `/rl-status`      | Check if RL agent is loaded and available    |
| GET    | `/nearest-node`   | Find closest intersection to coordinates     |

### Example: Optimize Route (Dijkstra)

```bash
curl -X POST http://localhost:8000/optimize-route \
  -H "Content-Type: application/json" \
  -d '{"source": "CBS Chowk", "destination": "Mhasrul", "time_of_day": 9, "day_of_week": 1, "weather": "clear", "method": "dijkstra"}'
```

### Example: Optimize Route (RL Agent)

```bash
curl -X POST http://localhost:8000/optimize-route \
  -H "Content-Type: application/json" \
  -d '{"source": "CBS Chowk", "destination": "Mhasrul", "time_of_day": 9, "day_of_week": 1, "weather": "clear", "method": "rl"}'
```

## Road Network

The system models **Nashik, Maharashtra** with:
- **20 intersections**: CBS Chowk, Gangapur Road, Dwarka Circle, College Road, Nashik Road, Indira Nagar, Panchavati, Mhasrul, Satpur, Ambad, Cidco, Nashik Phata, Canada Corner, Sharanpur Road, Mumbai Naka, Deolali, Trimbak Road, Pathardi Phata, Old Agra Road, Commissioner Office
- **30 road segments**: MG Road, Sharanpur Road, CBS-Canada Road, Old Pune Road, Panchavati Road, College Road, Gangapur Road, Indira Nagar Road, Commissioner Office Road, Nashik Phata Road, Trimbak Road, Mhasrul Road, Satpur Road, Cidco Road, Pathardi Road, Dwarka-CBS Road, Ambad Link Road, MIDC Road, Industrial Road, Ambad-Phata Road, Nashik-Mumbai Road, Deolali Road, Agra Road, Mhasrul Link

## Features

- **ML Congestion Prediction**: XGBoost model trained on simulated traffic data
- **RL Route Optimization**: Q-learning agent that learns optimal routing policies through 50,000 training episodes
- **Dijkstra Routing**: Classic shortest-path algorithm with ML-predicted congestion weights
- **Algorithm Comparison**: Switch between Dijkstra and RL in the dashboard to compare results
- **Real-time Route Optimization**: Dynamic edge weights based on time, day, and weather
- **Interactive Map Dashboard**: Dark-themed Leaflet.js map with congestion visualization
- **Alternative Routes**: Top 3 routes with time comparison
- **Weather & Time Factors**: Congestion varies by time, day, and weather conditions

## How RL Works in This Project

1. **Environment**: The Nashik road network is modeled as a Markov Decision Process (MDP)
2. **State**: `(current_intersection, destination, time_bucket, weather)`
3. **Actions**: Move to any adjacent intersection
4. **Reward**: Negative travel time (penalizes slow routes) + bonus for reaching destination
5. **Training**: The agent explores 50,000 random routes using epsilon-greedy exploration
6. **Result**: A Q-table that maps (state, action) → expected future reward, enabling fast route lookup

## License

This project was built for Smart India Hackathon 2024 (SIH1617).
