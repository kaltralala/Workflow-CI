import pandas as pd
import mlflow
import mlflow.sklearn
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def load_data(path):
    """Fungsi untuk memuat data dari path CSV."""
    return pd.read_csv(path)

def train_model(data_path):
    """
    Menggunakan pendekatan hybrid: autolog untuk hal standar, manual log untuk metrik uji dan visual.
    Script ini 'pasif' dan tidak memulai run-nya sendiri.
    """
    
    # Mengaktifkan Autologging. Ia akan melapor ke run aktif yang dibuat oleh 'mlflow run'.
    mlflow.sklearn.autolog()

    # Memuat Data
    print(f"Memuat data dari: {data_path}")
    X_train = load_data(os.path.join(data_path, 'train_features.csv'))
    y_train = load_data(os.path.join(data_path, 'train_labels.csv')).values.ravel()
    X_test = load_data(os.path.join(data_path, 'test_features.csv'))
    y_test = load_data(os.path.join(data_path, 'test_labels.csv')).values.ravel()
    
    # Inisialisasi dan Pelatihan Model
    print("Memulai pelatihan model RandomForest...")
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    # .fit() akan secara otomatis dicatat oleh autolog (parameter, metrik training, model dasar)
    model.fit(X_train, y_train)
    print("Model selesai dilatih.")

    # --- BAGIAN MANUAL LOGGING UNTUK HAL-HAL KUSTOM ---
    print("Memulai pencatatan manual untuk metrik uji dan artefak visual...")
    
    y_pred_test = model.predict(X_test)

    # Log Metrik pada Data Uji secara Manual
    mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
    mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
    mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred_test))
    print("Metrik pada data uji telah dicatat.")

    # Log Confusion Matrix sebagai Artefak Visual
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred_test, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title("Confusion Matrix (Test Data)")
    mlflow.log_figure(fig, "visuals/confusion_matrix.png")
    plt.close(fig) # Penting untuk menutup plot
    print("Artefak Confusion Matrix dicatat.")

# BLOK EKSEKUSI UTAMA
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    
    train_model(args.data_path)