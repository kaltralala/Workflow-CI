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

    # Memulai sesi logging MLflow.
    with mlflow.start_run(run_name="Automated_RF_Training"):
        
        # Mengaktifkan autologging untuk scikit-learn.
        # Ini akan secara otomatis mencatat parameter, metrik, dan artefak model.
        mlflow.sklearn.autolog()
        
        print("Memulai pelatihan model RandomForest...")
        # Inisialisasi model "juara" kita dari Kriteria 2.
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
    # Bagian ini membuat skrip kita bisa menerima argumen dari luar.
    
    # Membuat parser
    parser = argparse.ArgumentParser(description="Script pelatihan model untuk MLflow Project.")
    
    # Mendefinisikan argumen yang kita harapkan, yaitu --data_path.
    # Ini harus cocok dengan parameter yang kita definisikan di file MLProject.
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="MLProject/hasil_preprocessing/"
    )
    
    args = parser.parse_args()
    
    # Memanggil fungsi pelatihan utama dengan path data yang diterima dari argumen.
    train_model(args.data_path)