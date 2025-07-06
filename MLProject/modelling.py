# 1. BLOK IMPOR PUSTAKA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc
import mlflow
import mlflow.sklearn
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# 2. FUNGSI-FUNGSI PEMBANTU
def load_data(path):
    """Fungsi untuk memuat data dari path CSV."""
    return pd.read_csv(path)

# 3. FUNGSI UTAMA PELATIHAN MODEL
def train_model(data_path):
    """
    Fungsi ini menggunakan pendekatan hybrid:
    1. Autolog untuk mencatat parameter, metrik training, dan model dasar.
    2. Manual log untuk menambahkan metrik pada data uji dan artefak visual kustom.
    """
    
    # Mengaktifkan Autologging di awal.
    # MLflow akan mencatat secara otomatis saat .fit() dipanggil.
    mlflow.sklearn.autolog()

    # Memuat Data
    print(f"Memuat data dari: {data_path}")
    X_train = load_data(os.path.join(data_path, 'train_features.csv'))
    y_train = load_data(os.path.join(data_path, 'train_labels.csv')).values.ravel()
    X_test = load_data(os.path.join(data_path, 'test_features.csv'))
    y_test = load_data(os.path.join(data_path, 'test_labels.csv')).values.ravel()
    
    # Dengan 'autolog' aktif, kita TIDAK perlu 'with mlflow.start_run()' di sini
    # saat dijalankan via 'mlflow run'. MLflow akan menanganinya.
    
    # Inisialisasi dan Pelatihan Model
    print("Memulai pelatihan model RandomForest...")
    # Kita tidak perlu log parameter secara manual, autolog akan melakukannya.
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    print("Model selesai dilatih. Autolog telah mencatat parameter dan metrik training.")

    # --- BAGIAN MANUAL LOGGING UNTUK METRIK UJI & VISUAL ---
    print("Memulai pencatatan manual untuk metrik uji dan artefak visual...")
    
    # Evaluasi Model pada Data Uji
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    # Log Metrik pada Data Uji secara Manual
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
    mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
    mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
    mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred_test))
    mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_pred_proba_test))
    print("Metrik pada data uji telah dicatat.")

    # Log Confusion Matrix sebagai Artefak Visual
    fig_cm, ax_cm = plt.subplots()
    cm = confusion_matrix(y_test, y_pred_test, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax_cm, cmap=plt.cm.Blues, colorbar=False)
    ax_cm.set_title("Confusion Matrix (Test Data)")
    mlflow.log_figure(fig_cm, "visuals/confusion_matrix.png")
    plt.close(fig_cm)
    print("Artefak Confusion Matrix dicatat.")

    # Log Kurva Precision-Recall sebagai Artefak Visual
    fig_pr, ax_pr = plt.subplots()
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
    pr_auc = auc(recall, precision)
    ax_pr.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend()
    mlflow.log_figure(fig_pr, "visuals/precision_recall_curve.png")
    plt.close(fig_pr)
    print("Artefak Kurva Precision-Recall dicatat.")


# BLOK EKSEKUSI UTAMA
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    
    train_model(args.data_path)