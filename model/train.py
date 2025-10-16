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

    # Фіксуємо seed для стабільності результатів
    rng = np.random.default_rng(42)

    # Параметри датасету
    n_features = 20
    n_samples = 2000

    # Створюємо більш складний датасет
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_redundant=4,
        n_clusters_per_class=2,
        class_sep=0.9,     
        flip_y=0.07,     
        random_state=42
    )

    # Поділ на тренувальну й тестову вибірку
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Створюємо ансамбль моделей
    estimators = [
        ('rf', RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            random_state=42
        )),
        ('gb', GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.08,
            max_depth=3,
            random_state=42
        )),
        ('lr', LogisticRegression(
            max_iter=1000,
            solver="saga",
            C=0.8,
            random_state=42
        ))
    ]

    model_name = "VotingEnsemble"
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', VotingClassifier(estimators=estimators, voting='soft'))
    ])

    print(f"[{datetime.now()}] Обрана модель: {model_name}")
    model.fit(X_train, y_train)

    # Оцінка точності
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"[{datetime.now()}] Точність моделі: {accuracy:.4f}")
    print(f"[{datetime.now()}] Параметри: {n_features} фіч, {n_samples} зразків")

    # Збереження
    base_dir = pathlib.Path(__file__).resolve().parent.parent / "models"
    base_dir.mkdir(parents=True, exist_ok=True)

    model_path = base_dir / "model.pkl"
    metadata_path = base_dir / "metadata.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model_type": model_name,
        "accuracy": float(accuracy),
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "trained_at": datetime.now().isoformat()
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[{datetime.now()}] Модель ({model_name}) збережена у {model_path}")
    print(f"[{datetime.now()}] Метадані: {metadata_path}")

    return model_path, accuracy


if __name__ == "__main__":
    train_model()

