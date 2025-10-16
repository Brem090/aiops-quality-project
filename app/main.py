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
from threading import Lock, Thread

# -------- Опційний бекенд дрейфу (alibi-detect) --------
try:
    from alibi_detect.cd import TabularDrift
    _ALIBI_AVAILABLE = True
except ImportError:
    _ALIBI_AVAILABLE = False
    TabularDrift = None

# -------- Логування --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ml-inference")

# -------- Конфіг --------
MODEL_PATH = os.getenv("MODEL_PATH")  # якщо не задано — візьмемо з /app/models/model.pkl
DRIFT_BACKEND = os.getenv("DRIFT_BACKEND", "tabular")  # "tabular" або "zscore"
FEATURES_N = int(os.getenv("FEATURES_N", "20"))
MIN_REFERENCE_SAMPLES = int(os.getenv("MIN_REFERENCE_SAMPLES", "50"))
MAX_REFERENCE_SAMPLES = int(os.getenv("MAX_REFERENCE_SAMPLES", "200"))
ALIBI_P_VALUE = float(os.getenv("ALIBI_P_VALUE", "0.1"))  # чутливість тесту (0.05..0.2)

# -------- Прометеус-метрики --------
prediction_counter = Counter("predictions_total", "Total number of predictions")
prediction_latency = Histogram("prediction_latency_seconds", "Prediction latency")
drift_counter = Counter("drift_detected_total", "Total number of drift detections")

# -------- App --------
app = FastAPI(title="ML Inference Service")

# -------- Глобальний стан --------
model = None
reference_data: list = []          # буфер для X_ref
drift_detector = None              # об'єкт TabularDrift або None
feature_stats = None               # для Z-score fallback
_state_lock = Lock()
_init_in_progress = False
_drift_events = 0                  # власний лічильник детекцій

# -------- Pydantic --------
class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    drift_detected: bool
    timestamp: str

# -------- Завантаження моделі --------
def load_model():
    global model, feature_stats, drift_detector, reference_data, _init_in_progress

    base_dir = Path(__file__).resolve().parent
    default_model_path = base_dir / "models" / "model.pkl"
    model_path = Path(MODEL_PATH) if MODEL_PATH else default_model_path

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"✅ Model loaded from: {model_path}")

        with _state_lock:
            reference_data.clear()
            feature_stats = {
                "mean": np.zeros(FEATURES_N),
                "std": np.ones(FEATURES_N),
                "count": 0,
            }
            # Скидаємо детектор і прапор ініціалізації
            global drift_detector
            drift_detector = None
            global _init_in_progress
            _init_in_progress = False

        if DRIFT_BACKEND == "tabular" and not _ALIBI_AVAILABLE:
            logger.warning("⚠️ alibi-detect is not installed. Falling back to Z-score backend.")
        logger.info(f"🧭 Drift backend: {DRIFT_BACKEND} | FEATURES_N={FEATURES_N}")

    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    load_model()

# -------- Ініціалізація TabularDrift (синхронна) --------
def initialize_drift_detector() -> bool:
    """Ініціалізує TabularDrift з reference_data. Викликається у фоні."""
    global drift_detector
    if not _ALIBI_AVAILABLE or DRIFT_BACKEND != "tabular":
        return False

    with _state_lock:
        if len(reference_data) < MIN_REFERENCE_SAMPLES:
            logger.info(
                f"Collecting reference data: {len(reference_data)}/{MIN_REFERENCE_SAMPLES}"
            )
            return False

        try:
            X_ref = np.array(reference_data[:MAX_REFERENCE_SAMPLES], dtype=np.float32)
        except Exception as e:
            logger.error(f"Cannot convert reference_data to array: {e}")
            return False

    try:
        dd = TabularDrift(
            x_ref=X_ref,
            p_val=ALIBI_P_VALUE,
            categories_per_feature=None,
            preprocess_at_init=True,
        )
        with _state_lock:
            drift_detector = dd
        logger.info(f"✅ TabularDrift initialized with {X_ref.shape[0]} ref samples (p_val={ALIBI_P_VALUE})")
        return True
    except Exception as e:
        logger.error(f"Failed to initialize TabularDrift: {e}")
        return False

# -------- Фоновий воркер ініціалізації --------
def _init_worker():
    global _init_in_progress
    try:
        ok = initialize_drift_detector()
        if ok:
            logger.info("🎯 TabularDrift is ready")
        else:
            logger.error("❌ TabularDrift background init failed")
    finally:
        with _state_lock:
            _init_in_progress = False

# -------- Детекція дрейфу: Z-score fallback --------
def detect_drift_zscore(features: np.ndarray) -> bool:
    global feature_stats, _drift_events

    with _state_lock:
        if feature_stats["count"] == 0:
            feature_stats["mean"] = features.copy()
            feature_stats["std"] = np.ones_like(features)
        else:
            alpha = 0.1
            feature_stats["mean"] = (1 - alpha) * feature_stats["mean"] + alpha * features
            feature_stats["std"] = (1 - alpha) * feature_stats["std"] + alpha * np.abs(features - feature_stats["mean"])
        feature_stats["count"] += 1

        z_scores = np.abs((features - feature_stats["mean"]) / (feature_stats["std"] + 1e-6))
        drift_detected = bool(np.any(z_scores > 3.0))

    if drift_detected:
        with _state_lock:
            _drift_events += 1
        drift_counter.inc()
        logger.warning(f"🚨 Drift detected (Z-score)! max_z={float(np.max(z_scores)):.2f}")

    return drift_detected

# -------- Детекція дрейфу: TabularDrift --------
def detect_drift_tabular(features: np.ndarray) -> bool:
    global drift_detector, _drift_events, _init_in_progress

    with _state_lock:
        # Буферизуємо reference, поки детектор не готовий
        if drift_detector is None:
            if len(reference_data) < MAX_REFERENCE_SAMPLES:
                reference_data.append(features.tolist())

            # Досягли мінімуму — стартуємо одноразову фонову ініціалізацію
            if len(reference_data) >= MIN_REFERENCE_SAMPLES and not _init_in_progress:
                _init_in_progress = True
                Thread(target=_init_worker, daemon=True).start()

            # Поки ініт не завершено — не детектимо дрейф
            return False

    # Детектор готовий — перевіряємо дрейф
    try:
        X_test = features.reshape(1, -1).astype(np.float32)
        preds = drift_detector.predict(X_test)
        is_drift = bool(preds["data"]["is_drift"] == 1)
        if is_drift:
            with _state_lock:
                _drift_events += 1
            drift_counter.inc()
            logger.warning("🚨 Drift detected by TabularDrift!")
        return is_drift
    except Exception as e:
        logger.error(f"TabularDrift prediction failed: {e}, falling back to Z-score")
        return detect_drift_zscore(features)

# -------- Роутерний вибір бекенду --------
def detect_drift(features: np.ndarray) -> bool:
    if DRIFT_BACKEND == "tabular" and _ALIBI_AVAILABLE:
        return detect_drift_tabular(features)
    return detect_drift_zscore(features)

# -------- Інференс --------
def _predict_impl(features: List[float]) -> dict:
    X = np.array(features, dtype=np.float32).reshape(1, -1)
    if X.shape[1] != FEATURES_N:
        raise ValueError(f"Expected {FEATURES_N} features, got {X.shape[1]}")

    # Класифікація
    if hasattr(model, "predict_proba"):
        proba = float(model.predict_proba(X)[0][1])
        y_pred = int(proba > 0.5)
    else:
        # Fallback (без імовірності)
        y_pred = int(model.predict(X)[0])
        proba = 1.0

    drift = detect_drift(X[0])
    return {"prediction": y_pred, "probability": proba, "drift_detected": drift}

# -------- Endpoints --------
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    logger.info(f"Request: {len(request.features)} features")

    with prediction_latency.time():
        try:
            result = _predict_impl(request.features)
            prediction_counter.inc()
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    resp = PredictionResponse(
        prediction=result["prediction"],
        probability=result["probability"],
        drift_detected=result["drift_detected"],
        timestamp=datetime.now().isoformat(),
    )
    logger.info(f"Response: pred={resp.prediction}, prob={resp.probability:.3f}, drift={resp.drift_detected}")
    return resp

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    with _state_lock:
        detector_status = "ready" if drift_detector is not None else "not_initialized"
        ref_samples = len(reference_data) if drift_detector is None else "bound_to_detector"
        init_flag = _init_in_progress
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "drift_backend": DRIFT_BACKEND,
        "drift_detector": detector_status,
        "init_in_progress": init_flag,
        "reference_samples": ref_samples,
        "features_n": FEATURES_N,
    }

@app.get("/drift-stats")
async def drift_stats():
    with _state_lock:
        ref_count = len(reference_data)
        init_flag = _init_in_progress
        detector_ready = drift_detector is not None
        drift_events = _drift_events
    return {
        "backend": DRIFT_BACKEND,
        "alibi_available": _ALIBI_AVAILABLE,
        "detector_initialized": detector_ready,
        "init_in_progress": init_flag,
        "reference_samples_collected": ref_count,
        "min_samples_required": MIN_REFERENCE_SAMPLES,
        "x_ref_ready": min(ref_count, MAX_REFERENCE_SAMPLES) if not detector_ready else "bound to detector",
        "total_drift_detections": drift_events,
    }

@app.post("/reference/reset")
async def reference_reset():
    global drift_detector, _init_in_progress, _drift_events
    with _state_lock:
        reference_data.clear()
        drift_detector = None
        _init_in_progress = False
        _drift_events = 0
    return {"ok": True, "message": "reference buffer cleared"}

@app.post("/reference/init")
async def reference_init():
    """Форс-ініт з наявного буфера (для тестів/стендів)."""
    if not _ALIBI_AVAILABLE or DRIFT_BACKEND != "tabular":
        return {"ok": False, "reason": "tabular/alibi not active"}
    if len(reference_data) < max(5, MIN_REFERENCE_SAMPLES // 2):
        return {"ok": False, "reason": "not enough samples in buffer"}
    ok = initialize_drift_detector()
    return {"ok": bool(ok), "detector_initialized": drift_detector is not None}

@app.get("/")
async def root():
    return {
        "service": "ML Inference API",
        "version": "3.0.0",
        "drift_backend": DRIFT_BACKEND,
        "endpoints": ["/predict", "/metrics", "/health", "/drift-stats", "/reference/reset", "/reference/init"],
    }
