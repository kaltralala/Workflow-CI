# BLOK IMPOR PUSTAKA
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import argparse
from sklearn.ensemble import RandomForestClassifier

# FUNGSI-FUNGSI PEMBANTU
def load_data(path):
    """Fungsi untuk memuat data dari path CSV."""
    return pd.read_csv(path)

# FUNGSI UTAMA PELATIHAN MODEL
def train_model(data_path):
    """
    Fungsi utama untuk melatih model RandomForest dan menggunakan autolog.
    Menerima path data sebagai argumen untuk fleksibilitas.
    """

    mlflow.sklearn.autolog()
    
    # Mengatur nama eksperimen di MLflow.
    # Semua 'run' dari workflow ini akan dikelompokkan di sini.
    mlflow.set_experiment("Automated CI Training")

    # Memuat data menggunakan path yang diberikan sebagai argumen.
    print(f"Memuat data dari: {data_path}")
    try:
        X_train = load_data(os.path.join(data_path, 'train_features.csv'))
        y_train = load_data(os.path.join(data_path, 'train_labels.csv'))
    except FileNotFoundError as e:
        print(f"Error memuat data: {e}. Pastikan path '{data_path}' benar.")
        return
    
    print("Memulai pelatihan model RandomForest...")

    # Inisialisasi model RandomForest
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Melatih model
    model.fit(X_train, y_train.values.ravel())
    
    print("Pelatihan model selesai.")
    print("Parameter, metrik, dan model telah dicatat secara otomatis oleh MLflow.")

# BLOK EKSEKUSI UTAMA
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script pelatihan model untuk MLflow Project.")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="MLProject/hasil_preprocessing/"
    )
    args = parser.parse_args()
    train_model(args.data_path)