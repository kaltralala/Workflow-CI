import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import os

# Aktifkan autolog (cukup, tidak perlu set_experiment dan tracking_uri di MLProject)
mlflow.sklearn.autolog()

def run_model():
    # Load dataset hasil preprocessing (harus relatif path)
    df = pd.read_csv("rumah_jakarta_preprocessing.csv")

    # Fitur dan target
    X = df.drop(columns=["price", "log_price"]).astype(np.float64)
    y = df["log_price"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Mulai MLflow run
    with mlflow.start_run(run_name="RandomForest_Baseline_AutoLog"):
        # Logging dataset eksplisit (opsional)
        mlflow.log_artifact("rumah_jakarta_preprocessing.csv", artifact_path="dataset")

        # Train model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        print("Model trained and autologged successfully.")

if __name__ == "__main__":
    run_model()