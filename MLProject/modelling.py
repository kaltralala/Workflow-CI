# BLOK IMPOR PUSTAKA
import pandas as pd
import mlflow
import mlflow.sklearn
import os
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, auc
)

# FUNGSI PEMBANTU
def load_data(path):
    return pd.read_csv(path)

# FUNGSI UTAMA
def train_advanced_champion_model(train_feat_path, train_label_path, test_feat_path, test_label_path):
    print("Melacak eksperimen dengan MLflow (lokal)...")

    # Konfigurasi tracking lokal (bisa diganti ke folder permanen di proyek)
    tracking_dir = os.path.abspath("mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    mlflow.set_experiment("Fraud Detection - Champion Model")

    # Memuat data
    print("Memuat data...")
    X_train = load_data(train_feat_path)
    y_train = load_data(train_label_path).values.ravel()
    X_test = load_data(test_feat_path)
    y_test = load_data(test_label_path).values.ravel()

    # Mulai tracking eksperimen
    with mlflow.start_run(run_name="RandomForest_Champion_FinalReport"):
        print("Melatih model RandomForest...")

        params = {
            'n_estimators': 100,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        print("Evaluasi model...")
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]

        # Logging parameter dan metrik
        mlflow.log_params(params)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
        mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred_test))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_pred_proba_test))

        # Confusion Matrix
        fig_cm, ax_cm = plt.subplots()
        cm = confusion_matrix(y_test, y_pred_test, labels=model.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
        disp.plot(ax=ax_cm, cmap=plt.cm.Blues, colorbar=False)
        ax_cm.set_title("Confusion Matrix")
        mlflow.log_figure(fig_cm, "visuals/confusion_matrix.png")

        # Precision-Recall Curve
        fig_pr, ax_pr = plt.subplots()
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_test)
        pr_auc = auc(recall, precision)
        ax_pr.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
        ax_pr.set_title("Precision-Recall Curve")
        ax_pr.set_xlabel("Recall")
        ax_pr.set_ylabel("Precision")
        ax_pr.legend()
        mlflow.log_figure(fig_pr, "visuals/precision_recall_curve.png")

        # Simpan dan log model
        local_model_path = "champion_rf_model_local"
        mlflow.sklearn.save_model(sk_model=model, path=local_model_path)
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

        # Simpan path model untuk workflow (opsional)
        with open("model_path.txt", "w") as f:
            f.write(local_model_path)

        print("\n" + "="*60)
        print("Run selesai! Artefak & metrik telah dicatat (lokal).")
        print("="*60)

# EKSEKUSI UTAMA
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_features", type=str, required=True)
    parser.add_argument("--train_labels", type=str, required=True)
    parser.add_argument("--test_features", type=str, required=True)
    parser.add_argument("--test_labels", type=str, required=True)
    args = parser.parse_args()

    train_advanced_champion_model(
        args.train_features,
        args.train_labels,
        args.test_features,
        args.test_labels
    )