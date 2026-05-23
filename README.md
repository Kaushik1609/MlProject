# Dynamic Route Rationalization Model 

> AI-powered traffic route optimization using Machine Learning and real-time road parameters for Indian city road networks.

## Project Structure

```
SIH1617/
├── data/
│   ├── generate_data.py      # Traffic data simulation script
│   └── road_data.csv          # Generated traffic dataset (5000 rows)
├── ml/
│   ├── train_model.py         # XGBoost model training script
│   └── model.pkl              # Saved trained model
├── api/
│   └── main.py                # FastAPI backend server
├── frontend/
│   └── map.html               # Interactive map dashboard
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Tech Stack

| Component  | Technology                          |
|-----------|-------------------------------------|
| ML Model  | XGBoost, scikit-learn, joblib       |
| Backend   | FastAPI, Uvicorn                    |
| Routing   | NetworkX (Dijkstra's algorithm)     |
| Data      | Pandas, NumPy                       |
| Frontend  | Leaflet.js, HTML/CSS/JS             |
| Map Tiles | CartoDB Dark Matter (free, no key)  |

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

This creates `data/road_data.csv` with 5000 simulated traffic records.

### Step 3: Train the ML Model

```bash
python ml/train_model.py
```

This trains an XGBoost model and saves it as `ml/model.pkl`. It will print R² score and RMSE.

### Step 4: Start the API Server

```bash
python api/main.py
```

or

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

The API will run on `http://localhost:8000`. API docs available at `http://localhost:8000/docs`.

### Step 5: Open the Dashboard

Open `frontend/map.html` in your web browser. The dashboard connects to the API at `localhost:8000`.

## API Endpoints

| Method | Endpoint           | Description                                  |
|--------|-------------------|----------------------------------------------|
| GET    | `/`               | Health check and API status                  |
| GET    | `/graph-data`     | Returns road graph nodes and edges           |
| POST   | `/optimize-route` | Find optimal route between two intersections |
| GET    | `/traffic-status` | Real-time congestion for all road segments   |
| GET    | `/all-routes`     | Top 3 alternative routes with travel times   |

### Example: Optimize Route

```bash
curl -X POST http://localhost:8000/optimize-route \
  -H "Content-Type: application/json" \
  -d '{"source": "Shivajinagar", "destination": "Hadapsar", "time_of_day": 9, "day_of_week": 1, "weather": "clear"}'
```

## Road Network

The system models **Pune, India** with:
- **10 intersections**: Shivajinagar, Deccan Gymkhana, Swargate, Hinjewadi, Kothrud, Hadapsar, Viman Nagar, Aundh, Pimpri, Katraj
- **15 road segments**: FC Road, Tilak Road, Karve Road, Paud Road, Mumbai-Bangalore Highway, Nagar Road, Solapur Road, Satara Road, Airport Road, University Road, Hinjewadi Road, Aundh-Ravet Road, Old Mumbai Highway, Sinhagad Road, NIBM Road

## Features

- **ML Congestion Prediction**: XGBoost model trained on simulated traffic data
- **Real-time Route Optimization**: Dijkstra's algorithm with ML-predicted edge weights
- **Interactive Map Dashboard**: Dark-themed Leaflet.js map with congestion visualization
- **Alternative Routes**: Top 3 routes with time comparison
- **Weather & Time Factors**: Congestion varies by time, day, and weather conditions

## License

This project was built for Smart India Hackathon 2024 (SIH1617).
