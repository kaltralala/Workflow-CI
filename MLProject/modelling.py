import pandas as pd
import mlflow
import mlflow.sklearn
import os
import argparse
from sklearn.ensemble import RandomForestClassifier

def load_data(path):
    """Fungsi untuk memuat data dari path CSV."""
    return pd.read_csv(path)

def train_model(data_path):
    """
    Fungsi ini hanya fokus pada logika inti: load data dan train model.
    Pengaturan eksperimen dan run dikendalikan dari luar.
    """
    
    # --- Autologging diaktifkan di awal ---
    # Ia akan otomatis melapor ke run aktif yang dibuat oleh 'mlflow run'.
    mlflow.sklearn.autolog()

    # --- Memuat Data ---
    print(f"Memuat data dari: {data_path}")
    X_train = load_data(os.path.join(data_path, 'train_features.csv'))
    y_train = load_data(os.path.join(data_path, 'train_labels.csv'))
    
    # --- Inisialisasi dan Pelatihan Model ---
    print("Memulai pelatihan model RandomForest...")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # Melatih model
    model.fit(X_train, y_train.values.ravel())
    
    print("Pelatihan model selesai. Autologging seharusnya sudah mencatat hasilnya.")

# --- BLOK EKSEKUSI UTAMA (TETAP SAMA) ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    
    train_model(args.data_path)