import pandas as pd
import mlflow
import mlflow.sklearn
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score

def load_data(path):
    """Fungsi untuk memuat data dari path CSV."""
    return pd.read_csv(path)

def train_model(data_path):
    """
    Fungsi ini fokus pada load data, training model,
    dan logging manual metrik tambahan selain autolog.
    """

    # Autolog aktif (akan log param & model secara otomatis)
    mlflow.sklearn.autolog()

    # --- Memuat Data ---
    print(f"Memuat data dari: {data_path}")
    X_train = load_data(os.path.join(data_path, 'train_features.csv'))
    y_train = load_data(os.path.join(data_path, 'train_labels.csv'))

    # --- Inisialisasi & Pelatihan Model ---
    print("Memulai pelatihan model RandomForest...")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train.values.ravel())
    print("Pelatihan model selesai.")

    # --- Metrik tambahan (manual logging) ---
    y_pred = model.predict(X_train)
    f1 = f1_score(y_train, y_pred, average='macro')
    precision = precision_score(y_train, y_pred, average='macro')

    # Manual logging ke MLflow
    mlflow.log_metric("custom_f1_score_macro", f1)
    mlflow.log_metric("custom_precision_macro", precision)

    print("Metrik tambahan berhasil dilog ke MLflow.")

# Eksekusi utama
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    train_model(args.data_path)