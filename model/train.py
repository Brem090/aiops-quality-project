import pickle
import numpy as np
import os
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pathlib
import warnings
warnings.filterwarnings("ignore")


def train_model():
    print(f"[{datetime.now()}] Початок тренування моделі...")

    # Фіксуємо seed для відтворюваності
    rng = np.random.default_rng(42)

    # Параметри датасету
    n_features = 20
    n_samples = 1500

    # Створюємо стабільний датасет
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=12,
        n_redundant=4,
        flip_y=0.02,
        random_state=42
    )

    # Тренувальний і тестовий спліт
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Фіксована модель RandomForest
    model_name = "RandomForest"
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        random_state=42
    )

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
