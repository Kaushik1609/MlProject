"""
Data Simulation Script for Dynamic Route Rationalization Model
Generates road_data.csv with 5000 rows of simulated traffic data.
Congestion scores realistically reflect peak hours, weekends, and weather.
"""

import pandas as pd
import numpy as np
import os

# --- Road Segment Definitions ---
# Each segment corresponds to an edge in the Pune road network graph
ROAD_SEGMENTS = {
    1: 'MG Road', 2: 'Sharanpur Road', 3: 'CBS-Canada Road', 4: 'Old Pune Road',
    5: 'Panchavati Road', 6: 'College Road', 7: 'Gangapur Road', 8: 'Indira Nagar Road',
    9: 'Commissioner Office Road', 10: 'Nashik Phata Road', 11: 'Trimbak Road',
    12: 'Mhasrul Road', 13: 'Nashik Phata Road', 14: 'Satpur Road', 15: 'Cidco Road',
    16: 'Pathardi Road', 17: 'Dwarka-CBS Road', 18: 'Ambad Link Road', 19: 'MIDC Road',
    20: 'Industrial Road', 21: 'Ambad-Phata Road', 22: 'Nashik-Mumbai Road',
    23: 'Deolali Road', 24: 'Pathardi Road', 25: 'Agra Road', 26: 'Mhasrul Link',
    27: 'Sharanpur Road', 28: 'Trimbak Road', 29: 'Pathardi Road', 30: 'Agra Road'
}

WEATHER_OPTIONS = ["clear", "rain", "fog"]


def calculate_congestion(time_of_day, day_of_week, weather, vehicle_density):
    """
    Calculate a realistic congestion score based on multiple factors.
    
    Peak hours (8-10 AM, 5-8 PM) have higher base congestion.
    Weekends (5=Sat, 6=Sun) have lower congestion.
    Bad weather (rain, fog) increases congestion.
    Vehicle density directly correlates with congestion.
    
    Returns a float between 0.0 and 1.0.
    """
    # Base congestion from time of day (morning and evening peaks)
    if 8 <= time_of_day <= 10:
        time_factor = 0.7 + np.random.uniform(0, 0.2)  # Morning rush
    elif 17 <= time_of_day <= 20:
        time_factor = 0.75 + np.random.uniform(0, 0.2)  # Evening rush
    elif 12 <= time_of_day <= 14:
        time_factor = 0.4 + np.random.uniform(0, 0.15)  # Lunch hour moderate
    elif 6 <= time_of_day <= 7:
        time_factor = 0.3 + np.random.uniform(0, 0.15)  # Early morning buildup
    elif 21 <= time_of_day <= 23:
        time_factor = 0.2 + np.random.uniform(0, 0.1)   # Late evening
    else:
        time_factor = 0.1 + np.random.uniform(0, 0.1)   # Night / early morning

    # Weekend reduction factor
    if day_of_week >= 5:  # Saturday or Sunday
        weekend_factor = 0.6  # 40% reduction in congestion
    else:
        weekend_factor = 1.0

    # Weather impact factor
    weather_factors = {"clear": 1.0, "rain": 1.35, "fog": 1.2}
    weather_factor = weather_factors.get(weather, 1.0)

    # Vehicle density contribution (normalized 0-1)
    density_factor = vehicle_density / 100.0

    # Combine all factors with weighted formula
    congestion = (
        0.35 * time_factor * weekend_factor * weather_factor
        + 0.45 * density_factor * weather_factor
        + 0.20 * np.random.uniform(0.05, 0.3)  # Random noise for realism
    )

    # Clamp to [0.0, 1.0]
    return round(min(max(congestion, 0.0), 1.0), 4)


def generate_vehicle_density(time_of_day, day_of_week):
    """
    Generate realistic vehicle density based on time and day.
    Higher density during peak hours, lower on weekends.
    """
    if 8 <= time_of_day <= 10:
        base = np.random.randint(60, 100)
    elif 17 <= time_of_day <= 20:
        base = np.random.randint(65, 100)
    elif 12 <= time_of_day <= 14:
        base = np.random.randint(40, 70)
    elif 6 <= time_of_day <= 7:
        base = np.random.randint(30, 55)
    elif 21 <= time_of_day <= 23:
        base = np.random.randint(15, 40)
    else:
        base = np.random.randint(5, 25)

    # Reduce density on weekends
    if day_of_week >= 5:
        base = int(base * 0.65)

    return min(max(base, 0), 100)


def generate_dataset(num_rows=8000):
    """
    Generate the complete simulated traffic dataset.
    Each row represents a traffic observation on a road segment
    at a specific time, day, and weather condition.
    """
    np.random.seed(42)  # For reproducibility

    data = []
    segment_ids = list(ROAD_SEGMENTS.keys())

    for _ in range(num_rows):
        seg_id = np.random.choice(segment_ids)
        road_name = ROAD_SEGMENTS[seg_id]
        time_of_day = np.random.randint(0, 24)
        day_of_week = np.random.randint(0, 7)
        
        # Weather distribution: 60% clear, 25% rain, 15% fog
        weather = np.random.choice(
            WEATHER_OPTIONS, p=[0.60, 0.25, 0.15]
        )
        
        vehicle_density = generate_vehicle_density(time_of_day, day_of_week)
        congestion_score = calculate_congestion(
            time_of_day, day_of_week, weather, vehicle_density
        )

        data.append({
            "segment_id": seg_id,
            "road_name": road_name,
            "time_of_day": time_of_day,
            "day_of_week": day_of_week,
            "weather": weather,
            "vehicle_density": vehicle_density,
            "congestion_score": congestion_score,
        })

    return pd.DataFrame(data)


if __name__ == "__main__":
    print("=" * 60)
    print("  Dynamic Route Rationalization - Data Generator")
    print("=" * 60)
    
    df = generate_dataset(8000)
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(__file__), "road_data.csv")
    df.to_csv(output_path, index=False)
    
    print(f"\n[OK] Dataset generated successfully!")
    print(f"   Rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    print(f"   Saved to: {output_path}")
    print(f"\n[DATA] Sample data:")
    print(df.head(10).to_string(index=False))
    print(f"\n[STATS] Congestion score statistics:")
    print(df['congestion_score'].describe().to_string())
    print(f"\n[ROAD] Road segments covered: {df['segment_id'].nunique()}")
    print(f"[WEATHER] Weather distribution:")
    print(df['weather'].value_counts().to_string())
