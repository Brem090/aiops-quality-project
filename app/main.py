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

# ---------------- Опційний шар якості даних (Great Expectations) ----------------
try:
    import pandas as pd
    import great_expectations as ge
    _GE_AVAILABLE = True
except Exception:
    _GE_AVAILABLE = False
    ge = None
    pd = None

# ---------------- Конфіг через змінні оточення ----------------
MODEL_PATH = os.getenv("MODEL_PATH")  # якщо не задано — /app/models/model.pkl
FEATURES_N = int(os.getenv("FEATURES_N", "20"))
ZSCORE_THRESHOLD = float(os.getenv("ZSCORE_THRESHOLD", "3.0"))
GE_ENABLE = os.getenv("GE_ENABLE", "true").lower() == "true"

# ---------------- Прометеус-метрики ----------------
prediction_counter = Counter("predictions_total", "Загальна кількість передбачень")
prediction_latency = Histogram("prediction_latency_seconds", "Латентність передбачень, с")
drift_counter = Counter("drift_detected_total", "Загальна кількість детекцій дрейфу")

# ---------------- Ініціалізація застосунку ----------------
app = FastAPI(title="ML Inference Service")

# ---------------- Глобальний стан ----------------
model = None
feature_stats = None  # онлайн-оцінки середнього/розкиду для Z-score
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

# ---------------- GE-валідація (м'яка перевірка якості) ----------------
def _validate_with_ge(features: np.ndarray) -> bool:
    """Повертає True, якщо GE-валідація пройдена або вимкнена."""
    if not (_GE_AVAILABLE and GE_ENABLE):
        return True
    try:
        import great_expectations as gx
        import pandas as pd
        from great_expectations.validator.validator import Validator
        from great_expectations.execution_engine import PandasExecutionEngine

        cols = [f"f{i}" for i in range(features.shape[0])]
        df = pd.DataFrame([features], columns=cols)

        # Створюємо Validator із PandasExecutionEngine (новий API)
        validator = Validator(execution_engine=PandasExecutionEngine())

        # Додаємо батч вручну
        validator.execution_engine.load_batch_data("tmp_batch", df)

        # Очікування
        validator.expect_table_row_count_to_equal(1)
        for c in cols:
            validator.expect_column_values_to_not_be_null(c)

        # Межі значень на основі статистики
        with _state_lock:
            fs = feature_stats
            have_stats = fs is not None and fs["count"] >= 5
            if have_stats:
                mean, std = fs["mean"], fs["std"]

        if have_stats:
            low = (mean - ZSCORE_THRESHOLD * (std + 1e-6)).tolist()
            high = (mean + ZSCORE_THRESHOLD * (std + 1e-6)).tolist()
            for i, c in enumerate(cols):
                validator.expect_column_values_to_be_between(c, low[i], high[i], mostly=1.0)

        result = validator.validate()
        return bool(result.success)

    except Exception as e:
        logger.error(f"GE validation error (ignored): {e}")
        return True

# ---------------- Детекція дрейфу: Z-score (+ опційний GE) ----------------
def detect_drift(features: np.ndarray) -> bool:
    """
    Онлайн-оновлення статистик ознак та перевірка на дрейф за Z-score.
    Додатково застосовується м'яка GE-перевірка якості.
    """
    global feature_stats, _drift_events

    with _state_lock:
        if feature_stats["count"] == 0:
            feature_stats["mean"] = features.astype(np.float32).copy()
            feature_stats["std"] = np.ones_like(features, dtype=np.float32)
        else:
            alpha = 0.1  # швидкість згладжування
            feature_stats["mean"] = (
                (1 - alpha) * feature_stats["mean"] + alpha * features
            )
            feature_stats["std"] = (
                (1 - alpha) * feature_stats["std"]
                + alpha * np.abs(features - feature_stats["mean"])
            )
        feature_stats["count"] += 1

        z_scores = np.abs((features - feature_stats["mean"]) / (feature_stats["std"] + 1e-6))
        zscore_drift = bool(np.any(z_scores > ZSCORE_THRESHOLD))

    ge_ok = _validate_with_ge(features)
    drift_detected = zscore_drift or (not ge_ok)

    if drift_detected:
        with _state_lock:
            _drift_events += 1
        drift_counter.inc()
        logger.warning(
            f"Виявлено дрейф: max_z={float(np.max(z_scores)):.2f}, поріг_z={ZSCORE_THRESHOLD}, ge_ok={ge_ok}"
        )

    return drift_detected

# ---------------- Передбачення ----------------
def _predict_impl(features: List[float]) -> dict:
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    if X.shape[1] != FEATURES_N:
        raise ValueError(f"Очікується {FEATURES_N} ознак, отримано {X.shape[1]}")

    # Класифікація
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
        y_pred = int(proba > 0.5)
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
        "ge_enabled": bool(_GE_AVAILABLE and GE_ENABLE),
        "drift_events": _drift_events,
        "feature_stats": stats,
    }

@app.get("/")
async def root():
    return {
        "service": "ML Inference API",
        "version": "2.1.0",
        "description": "Сервіс інференсу з детекцією дрейфу за Z-score та опційною GE-валідацією.",
        "endpoints": ["/predict", "/metrics", "/health"],
    }
