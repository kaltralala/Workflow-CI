import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import mlflow
import mlflow.sklearn
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data(path):
    return pd.read_csv(path)

def train_model(data_path):
    mlflow.set_experiment("Automated CI Training")

    print("Memuat data dari:", data_path)
    X_train = load_data(os.path.join(data_path, 'train_features.csv'))
    y_train = load_data(os.path.join(data_path, 'train_labels.csv')).values.ravel()
    X_test = load_data(os.path.join(data_path, 'test_features.csv'))
    y_test = load_data(os.path.join(data_path, 'test_labels.csv')).values.ravel()
    
    # Cek apakah run sudah aktif (karena dipanggil dari `mlflow run .`)
    if mlflow.active_run() is None:
        mlflow.start_run(run_name="Automated_RF_ManualLog")

    print("Melatih model RandomForest...")
    params = {
        'n_estimators': 100,
        'class_weight': 'balanced',
        'random_state': 42,
        'n_jobs': -1
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    print("Model selesai dilatih.")

    print("Evaluasi model pada data uji...")
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    print("Logging parameter dan metrik ke MLflow...")
    mlflow.log_params(params)
    mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
    mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
    mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
    mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred_test))
    mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_pred_proba_test))

    mlflow.sklearn.log_model(model, "model")

    cm = confusion_matrix(y_test, y_pred_test)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Test Data)")

    conf_matrix_path = "training_confusion_matrix.png"
    plt.savefig(conf_matrix_path)
    mlflow.log_artifact(conf_matrix_path)
    plt.close()

    print("Logging selesai.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    train_model(args.data_path)