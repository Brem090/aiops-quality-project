import pickle
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from datetime import datetime

def train_model():
    """Тренування простої моделі класифікації"""
    print(f"[{datetime.now()}] Starting model training...")
    
    # Генерація даних
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Тренування моделі
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Оцінка
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy: {accuracy:.4f}")
    
    # Збереження моделі
    os.makedirs("../models", exist_ok=True)
    model_path = "../models/model.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    print(f"Model saved to {model_path}")
    
    # Збереження метаданих
    metadata = {
        "accuracy": accuracy,
        "trained_at": datetime.now().isoformat(),
        "n_features": 20
    }
    
    with open("../models/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    
    return model_path, accuracy

if __name__ == "__main__":
    train_model()