from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest

import numpy as np
import pickle
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import List
from threading import Lock

# ---------------- Налаштування логування ----------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ml-inference")

# ---------------- Конфігурація через змінні оточення ----------------
MODEL_PATH = os.getenv("MODEL_PATH")  
FEATURES_N = int(os.getenv("FEATURES_N", "20"))
ZSCORE_THRESHOLD = float(os.getenv("ZSCORE_THRESHOLD", "3.0"))
PROBA_THRESHOLD = float(os.getenv("PROBA_THRESHOLD", "0.5"))

# ---------------- Прометеус-метрики ----------------
prediction_counter = Counter("predictions_total", "Загальна кількість передбачень")
prediction_latency = Histogram("prediction_latency_seconds", "Латентність передбачень, с")
drift_counter = Counter("drift_detected_total", "Загальна кількість детекцій дрейфу")

# ---------------- Ініціалізація застосунку ----------------
app = FastAPI(title="ML Inference Service")

# ---------------- Глобальний стан ----------------
model = None
feature_stats = None  
_state_lock = Lock()
_drift_events = 0

# ---------------- Моделі запиту/відповіді ----------------
class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    drift_detected: bool
    timestamp: str

# ---------------- Завантаження моделі ----------------
def load_model() -> None:
    """Завантажити модель з диска та скинути статистики ознак."""
    global model, feature_stats

    base_dir = Path(__file__).resolve().parent
    default_model_path = base_dir / "models" / "model.pkl"
    model_path = Path(MODEL_PATH) if MODEL_PATH else default_model_path

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Модель завантажено з: {model_path}")

        with _state_lock:
            feature_stats = {
                "mean": np.zeros(FEATURES_N, dtype=np.float32),
                "std": np.ones(FEATURES_N, dtype=np.float32),
                "count": 0,
            }

        logger.info(
            f"Увімкнено дрейф за Z-score | FEATURES_N={FEATURES_N} | поріг_z={ZSCORE_THRESHOLD}"
        )
    except Exception as e:
        logger.error(f"Не вдалося завантажити модель: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    load_model()

# ---------------- Внутрішні перевірки вхідних даних ----------------
def _quick_sanity(x: np.ndarray) -> None:
    """Кидає помилку, якщо дані невалідні."""
    if x.ndim != 2 or x.shape[1] != FEATURES_N:
        raise ValueError(f"Очікується {FEATURES_N} ознак, отримано {x.shape[1] if x.ndim == 2 else 'не 2D'}")
    if not np.all(np.isfinite(x)):
        raise ValueError("Вхідні ознаки містять NaN/Inf.")

# ---------------- Детекція дрейфу: Z-score ----------------
def detect_drift(features: np.ndarray) -> bool:

    global feature_stats, _drift_events

    with _state_lock:
        # ініціалізація
        if feature_stats["count"] == 0:
            feature_stats["mean"] = features.astype(np.float32).copy()
            feature_stats["std"] = np.ones_like(features, dtype=np.float32)
        else:
            alpha = 0.1  
            feature_stats["mean"] = (1 - alpha) * feature_stats["mean"] + alpha * features
            feature_stats["std"] = (1 - alpha) * feature_stats["std"] + alpha * np.abs(features - feature_stats["mean"])

        feature_stats["count"] += 1
        z_scores = np.abs((features - feature_stats["mean"]) / (feature_stats["std"] + 1e-6))
        zscore_drift = bool(np.any(z_scores > ZSCORE_THRESHOLD))

    if zscore_drift:
        with _state_lock:
            _drift_events += 1
        drift_counter.inc()
        logger.warning(f"Виявлено дрейф: max_z={float(np.max(z_scores)):.2f}, поріг_z={ZSCORE_THRESHOLD}")

    return zscore_drift

# ---------------- Передбачення ----------------
def _predict_impl(features: List[float]) -> dict:
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    _quick_sanity(X)

    # Класифікація
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
        y_pred = int(proba > PROBA_THRESHOLD)
    else:
        y_pred = int(model.predict(X)[0])
        proba = 1.0 

    drift = detect_drift(X[0])
    return {"prediction": y_pred, "probability": proba, "drift_detected": drift}

# ---------------- Маршрути ----------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    logger.info(f"Отримано запит: {len(request.features)} ознак")
    with prediction_latency.time():
        try:
            result = _predict_impl(request.features)
            prediction_counter.inc()
        except Exception as e:
            logger.error(f"Помилка під час передбачення: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    resp = PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        drift_detected=result["drift_detected"],
        timestamp=datetime.now().isoformat(),
    )
    logger.info(
        f"Відповідь: клас={resp.prediction}, ймовірність={resp.probability:.3f}, дрейф={resp.drift_detected}"
    )
    return resp

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    with _state_lock:
        stats = None if feature_stats is None else {"count": feature_stats["count"]}
    return {
        "status": "працює",
        "model_loaded": bool(model is not None),
        "features_n": FEATURES_N,
        "zscore_threshold": ZSCORE_THRESHOLD,
        "proba_threshold": PROBA_THRESHOLD,
        "drift_events": _drift_events,
        "feature_stats": stats,
    }

@app.get("/")
async def root():
    return {
        "service": "ML Inference API",
        "version": "2.4.0",
        "description": "Сервіс інференсу з детекцією дрейфу за Z-score (EMA + 3σ).",
        "endpoints": ["/predict", "/metrics", "/health"],
    }

