# 1. BLOK IMPOR PUSTAKA
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
    Fungsi ini HANYA fokus pada logika inti: load data, train model,
    dan melaporkan hasilnya ke run MLflow yang SUDAH AKTIF.
    """

    # Memuat Data
    print(f"Memuat data dari: {data_path}")
    X_train = load_data(os.path.join(data_path, 'train_features.csv'))
    y_train = load_data(os.path.join(data_path, 'train_labels.csv')).values.ravel()
    X_test = load_data(os.path.join(data_path, 'test_features.csv'))
    y_test = load_data(os.path.join(data_path, 'test_labels.csv')).values.ravel()
    
    # Inisialisasi dan Pelatihan Model
    print("Memulai pelatihan model RandomForest...")
    params = {
        'n_estimators': 100,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    print("Model selesai dilatih.")

    # Evaluasi Model pada Data Uji
    print("Evaluasi model pada data uji...")
    y_pred_test = model.predict(X_test)

    # MANUAL LOGGING (Akan otomatis log ke run yang aktif)
    print("Mencatat parameter, metrik, dan artefak ke MLflow...")
    
    mlflow.log_params(params)
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
    mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
    mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
    mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred_test))
    
    # Membuat dan menyimpan Confusion Matrix
    fig, ax = plt.subplots()
    cm = confusion_matrix(y_test, y_pred_test, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
    ax.set_title("Confusion Matrix (Test Data)")
    mlflow.log_figure(fig, "visuals/confusion_matrix.png")
    plt.close(fig)
    print("Artefak Confusion Matrix dicatat.")
    
    # Log Model itu sendiri
    mlflow.sklearn.log_model(model, "model")
    print("Model dicatat sebagai artefak.")
    print("Logging selesai.")

# 4. BLOK EKSEKUSI UTAMA
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    train_model(args.data_path)