#!/usr/bin/env python3
"""Скрипт для тестування Drift Detection у FastAPI ML Inference Service"""

import requests
import numpy as np
import time
from datetime import datetime

API_URL = "http://localhost:8000"
TIMEOUT = 5  # секунд для HTTP-запитів
SEED = 42    # для відтворюваності результатів

np.random.seed(SEED)

def send_prediction_request(features):
    """Відправка запиту на передбачення"""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            timeout=TIMEOUT
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}


def generate_normal_data(n_features=20):
    """Генерація нормальних (стандартних) даних"""
    return np.random.randn(n_features).tolist()


def generate_drift_data(n_features=20, shift=10):
    """Генерація даних з дрейфом (зміщення середнього)"""
    return (np.random.randn(n_features) + shift).tolist()


def test_normal_requests(n_requests=30):
    """Тестування звичайних (нормальних) запитів"""
    print(f"\n{'='*60}")
    print(f"Відправка {n_requests} нормальних запитів...")
    print(f"{'='*60}\n")

    drift_count = 0
    start = time.time()

    for i in range(n_requests):
        features = generate_normal_data()
        result = send_prediction_request(features)

        if "error" in result:
            print(f"[{i+1:3d}] ✗ Error: {result['error']}")
            continue

        if result.get("drift_detected"):
            drift_count += 1
            print(f"[{i+1:3d}] ⚠️  DRIFT - Prediction: {result['prediction']}")
        else:
            print(f"[{i+1:3d}] ✓  Normal - Prediction: {result['prediction']}")

        time.sleep(0.1)

    duration = time.time() - start
    print(f"\nDrift detections: {drift_count}/{n_requests}")
    print(f"⏱ Тривалість тесту: {duration:.2f} с")


def test_drift_requests(n_requests=20):
    """Тестування аномальних (з дрейфом) запитів"""
    print(f"\n{'='*60}")
    print(f"Відправка {n_requests} аномальних запитів...")
    print(f"{'='*60}\n")

    drift_count = 0
    start = time.time()

    for i in range(n_requests):
        features = generate_drift_data(shift=10)
        result = send_prediction_request(features)

        if "error" in result:
            print(f"[{i+1:3d}] ✗ Error: {result['error']}")
            continue

        if result.get("drift_detected"):
            drift_count += 1
            print(f"[{i+1:3d}] 🔴 DRIFT DETECTED")
        else:
            print(f"[{i+1:3d}] ✓  Normal")

        time.sleep(0.1)

    duration = time.time() - start
    print(f"\nDrift detections: {drift_count}/{n_requests}")
    print(f"⏱ Тривалість тесту: {duration:.2f} с")


def main():
    """Головна функція"""
    print(f"\n{'#'*60}")
    print(f"# ML Inference Service - Drift Detection Test")
    print(f"{'#'*60}\n")

    # Перевірка доступності API
    try:
        response = requests.get(f"{API_URL}/health", timeout=TIMEOUT)
        response.raise_for_status()
        print(f"✓ API is healthy\n")
    except Exception as e:
        print(f"✗ API is not available: {e}")
        print("👉 Запусти сервіс або виконай:")
        print("   kubectl port-forward -n ml-service svc/ml-service-ml-inference-service 8000:8000\n")
        return

    # Основні тести
    test_normal_requests(n_requests=30)
    time.sleep(2)
    test_drift_requests(n_requests=20)

    print(f"\n{'#'*60}")
    print("# ✅ Тестування завершено успішно!")
    print(f"{'#'*60}\n")


if __name__ == "__main__":
    main()
