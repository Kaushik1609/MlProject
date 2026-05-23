"""
ML Model Training Script for Dynamic Route Rationalization
===========================================================
Trains an XGBoost regression model to predict congestion_score
from traffic features: time_of_day, day_of_week, weather, vehicle_density.

Outputs:
  - model.pkl           (trained XGBoost regressor)
  - weather_encoder.pkl (fitted LabelEncoder for weather column)
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor


def load_and_prepare_data(csv_path):
    """
    Load the road traffic dataset and prepare features for training.
    Encodes the 'weather' column using LabelEncoder.
    Returns the DataFrame, features X, target y, and the fitted encoder.
    """
    df = pd.read_csv(csv_path)
    print(f"  Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"  Columns: {list(df.columns)}")

    # Label encode the categorical weather column
    weather_encoder = LabelEncoder()
    df["weather_encoded"] = weather_encoder.fit_transform(df["weather"])
    print(f"  Weather classes: {dict(zip(weather_encoder.classes_, weather_encoder.transform(weather_encoder.classes_)))}")

    # Select features and target
    feature_cols = ["time_of_day", "day_of_week", "weather_encoded", "vehicle_density"]
    X = df[feature_cols]
    y = df["congestion_score"]

    return X, y, weather_encoder


def train_xgboost(X, y):
    """
    Train an XGBoost regression model with an 80/20 train-test split.
    Returns: trained model, X_test, y_test for evaluation.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"\n  Training XGBoost model...")
    print(f"    Training samples: {len(X_train)}")
    print(f"    Testing samples:  {len(X_test)}")

    # XGBoost regressor with tuned hyperparameters
    model = XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    return model, X_test, y_test


def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model and print R2 score and RMSE.
    Also displays feature importance ranking.
    """
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print(f"\n  +----------------------------------+")
    print(f"  |   Model Performance Metrics      |")
    print(f"  +----------------------------------+")
    print(f"  |   R2 Score :  {r2:.4f}             |")
    print(f"  |   RMSE     :  {rmse:.4f}             |")
    print(f"  +----------------------------------+")

    # Feature importance ranking
    feature_names = ["time_of_day", "day_of_week", "weather_encoded", "vehicle_density"]
    importances = model.feature_importances_
    print(f"\n  Feature Importance:")
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "#" * int(imp * 40)
        print(f"    {name:25s} {imp:.4f} {bar}")

    return r2, rmse


if __name__ == "__main__":
    print("=" * 60)
    print("  Dynamic Route Rationalization -- Model Training")
    print("=" * 60)

    # Resolve paths relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    csv_path = os.path.join(base_dir, "data", "road_data.csv")
    model_path = os.path.join(script_dir, "model.pkl")
    encoder_path = os.path.join(script_dir, "weather_encoder.pkl")

    # Check if dataset exists
    if not os.path.exists(csv_path):
        print(f"\n  [ERROR] {csv_path} not found.")
        print("  Run data/generate_data.py first!")
        exit(1)

    # Step 1: Load and prepare data
    X, y, weather_encoder = load_and_prepare_data(csv_path)

    # Step 2: Train model
    model, X_test, y_test = train_xgboost(X, y)

    # Step 3: Evaluate
    r2, rmse = evaluate_model(model, X_test, y_test)

    # Step 4: Save model and encoder SEPARATELY
    # The API loads these as independent files
    joblib.dump(model, model_path)
    joblib.dump(weather_encoder, encoder_path)

    print(f"\n  [OK] Model saved to: {model_path}")
    print(f"       Size: {os.path.getsize(model_path) / 1024:.1f} KB")
    print(f"  [OK] Weather encoder saved to: {encoder_path}")
    print(f"\n  Training complete! Model is ready for deployment.")
    print("=" * 60)
