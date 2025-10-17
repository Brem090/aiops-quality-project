import pickle
import numpy as np
import os
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pathlib
import warnings
warnings.filterwarnings("ignore")


def train_model():
    print(f"[{datetime.now()}] Початок тренування ансамблевої моделі...")

    # ---- Параметри ----
    rng = np.random.default_rng(42)
    n_features = 20
    n_samples = 2000

    # ---- Створення датасету ----
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=8,
        n_redundant=8,
        n_clusters_per_class=3,
        class_sep=0.8,   
        flip_y=0.1,      
        random_state=42
    )

    # ---- Поділ на train/test ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---- Окремі моделі ----
    rf = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=3,
        random_state=42
    )

    gb = GradientBoostingClassifier(
        n_estimators=120,
        learning_rate=0.07,
        max_depth=3,
        subsample=0.9,
        random_state=42
    )

    lr = LogisticRegression(
        max_iter=1000,
        solver="saga",
        C=0.8,
        random_state=42
    )

    # ---- Ансамбль ----
    estimators = [
        ('rf', rf),
        ('gb', gb),
        ('lr', lr)
    ]

    model_name = "VotingEnsemble"
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', VotingClassifier(
            estimators=estimators,
            voting='soft',
            weights=[2, 2, 1]  # сильніші ваги для RF і GB
        ))
    ])

    print(f"[{datetime.now()}] Модель: {model_name}")
    model.fit(X_train, y_train)

    # ---- Оцінка ----
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)

    print(f"[{datetime.now()}] Train accuracy: {train_acc:.4f}")
    print(f"[{datetime.now()}] Test  accuracy: {test_acc:.4f}")
    print(f"[{datetime.now()}] Фіч: {n_features}, зразків: {n_samples}")

    # ---- Збереження ----
    base_dir = pathlib.Path(__file__).resolve().parent.parent / "models"
    base_dir.mkdir(parents=True, exist_ok=True)

    model_path = base_dir / "model.pkl"
    metadata_path = base_dir / "metadata.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model_type": model_name,
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "trained_at": datetime.now().isoformat(),
        "random_state": 42
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[{datetime.now()}] Модель ({model_name}) збережена у {model_path}")
    print(f"[{datetime.now()}] Метадані: {metadata_path}")

    return model_path, test_acc


if __name__ == "__main__":
    train_model()


