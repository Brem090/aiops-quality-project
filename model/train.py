import pickle
import numpy as np
import os
from datetime import datetime
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

def train_model():
    print(f"[{datetime.now()}] Початок тренування моделі...")

    rng = np.random.default_rng(int(datetime.now().timestamp()) % 2**32)
    
    n_features = 20
    n_samples = 1000 + rng.integers(0, 500) 

    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,  # ✅ Завжди 20
        n_informative=rng.integers(10, 16),
        n_redundant=rng.integers(2, 6),
        flip_y=rng.uniform(0.01, 0.05), 
        random_state=rng.integers(0, 9999)
    )

    # Варіюємо трансформації фіч (опціонально)
    if rng.random() > 0.5:
        X[:, 0] = np.sin(X[:, 0]) * rng.uniform(0.8, 1.2)
    if rng.random() > 0.7:
        X[:, 1] = np.log(np.abs(X[:, 1]) + 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=rng.integers(0, 9999)
    )

    # Обираємо рандомну модель
    model_choices = {
        "RandomForest": RandomForestClassifier(
            n_estimators=rng.integers(80, 150),
            max_depth=rng.choice([None, 8, 10, 12]),
            random_state=rng.integers(0, 9999)
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=rng.integers(80, 150),
            learning_rate=rng.uniform(0.03, 0.1),
            max_depth=rng.integers(2, 5),
            random_state=rng.integers(0, 9999)
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=500,
            C=rng.uniform(0.5, 3.0),
            solver="lbfgs"
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(rng.integers(8, 32), rng.integers(8, 32)),
            activation=rng.choice(["relu", "tanh"]),
            learning_rate_init=rng.uniform(0.001, 0.01),
            max_iter=300,
            random_state=rng.integers(0, 9999)
        )
    }

    chosen_name = rng.choice(list(model_choices.keys()))
    model = model_choices[chosen_name]
    print(f"[{datetime.now()}] Обрана модель: {chosen_name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"[{datetime.now()}] Точність моделі: {accuracy:.4f}")
    print(f"[{datetime.now()}] Параметри: {n_features} фіч, {n_samples} зразків")

    os.makedirs("../models", exist_ok=True)
    model_path = "models/model.pkl"
    metadata_path = "models/metadata.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    metadata = {
        "model_type": chosen_name,
        "accuracy": float(accuracy),
        "n_samples": int(n_samples),
        "n_features": int(n_features),  # ✅ Завжди 20
        "trained_at": datetime.now().isoformat()
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"[{datetime.now()}] Модель ({chosen_name}) збережена у {model_path}")
    print(f"[{datetime.now()}] Метадані: {metadata_path}")

    return model_path, accuracy


if __name__ == "__main__":
    train_model()