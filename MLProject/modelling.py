import pandas as pd
import mlflow
import mlflow.sklearn
import os
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def load_data(path):
    """Memuat data dari file CSV."""
    return pd.read_csv(path)

def train_model(data_path):
    """Melatih model RandomForest dan mencatat hasilnya secara manual ke MLflow."""
    
    mlflow.set_experiment("Automated CI Training")

    print("Memuat data dari:", data_path)
    X_train = load_data(os.path.join(data_path, 'train_features.csv'))
    y_train = load_data(os.path.join(data_path, 'train_labels.csv')).values.ravel()
    X_test = load_data(os.path.join(data_path, 'test_features.csv'))
    y_test = load_data(os.path.join(data_path, 'test_labels.csv')).values.ravel()
    
    with mlflow.start_run(run_name="Automated_RF_ManualLog"):
        
        # Inisialisasi dan latih model
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

        # Evaluasi model
        print("Evaluasi model pada data uji...")
        y_pred_test = model.predict(X_test)
        y_pred_proba_test = model.predict_proba(X_test)[:, 1]

        # Logging manual ke MLflow
        print("Logging parameter dan metrik ke MLflow...")
        mlflow.log_params(params)
        mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_pred_test))
        mlflow.log_metric("test_precision", precision_score(y_test, y_pred_test))
        mlflow.log_metric("test_recall", recall_score(y_test, y_pred_test))
        mlflow.log_metric("test_f1_score", f1_score(y_test, y_pred_test))
        mlflow.log_metric("test_roc_auc", roc_auc_score(y_test, y_pred_proba_test))

        # Simpan model ke MLflow
        mlflow.sklearn.log_model(model, "model")
        print("Logging selesai.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    train_model(args.data_path)