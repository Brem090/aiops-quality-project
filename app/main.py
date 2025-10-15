from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import logging
from pathlib import Path
from datetime import datetime
from typing import List
import os
import requests
from threading import Lock, Thread

# Спроба завантажити Alibi Detect
try:
    from alibi_detect.cd import TabularDrift
    _ALIBI_AVAILABLE = True
except ImportError:
    _ALIBI_AVAILABLE = False
    TabularDrift = None

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus метрики
prediction_counter = Counter('predictions_total', 'Total number of predictions')
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
drift_counter = Counter('drift_detected_total', 'Total number of drift detections')

app = FastAPI(title="ML Inference Service")

# Глобальні змінні
model = None
drift_detector = None
reference_data: list = []  # для збирання даних референсу
feature_stats = None       # fallback для Z-score

# Конкурентні оновлення
_state_lock = Lock()
_drift_events = 0  # власний лічильник дрейфів

# GitHub webhook (для retrain)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "Brem090/aiops-quality-project")
GITHUB_WORKFLOW = "ci-cd.yaml"

# Конфігурація дрейфу
DRIFT_BACKEND = os.getenv("DRIFT_BACKEND", "tabular")  # tabular або zscore
MIN_REFERENCE_SAMPLES = 100
MAX_REFERENCE_SAMPLES = 500

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    drift_detected: bool
    timestamp: str

def load_model():
    """Завантаження моделі при старті"""
    global model, feature_stats, drift_detector, reference_data

    base_dir = Path(__file__).resolve().parent
    default_path = base_dir / "models" / "model.pkl"
    model_path = Path(os.getenv("MODEL_PATH", str(default_path)))

    try:
        with open(model_path, "rb") as f:
            loaded = pickle.load(f)
        model = loaded
        logger.info(f"✓ Model loaded successfully from {model_path}")

        # Ініціалізація статистики для Z-score fallback
        feature_stats = {
            "mean": np.zeros(20),
            "std": np.ones(20),
            "count": 0,
        }

        # Скидаємо детектор і референсні дані
        with _state_lock:
            reference_data.clear()
            # ініціалізуємо знову
            global drift_detector
            drift_detector = None

        logger.info(f"Drift backend: {DRIFT_BACKEND}")
        if not _ALIBI_AVAILABLE and DRIFT_BACKEND == "tabular":
            logger.warning("⚠️  alibi-detect not available, falling back to Z-score")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

@app.on_event("startup")
async def startup_event():
    load_model()

def trigger_retrain_workflow():
    """Тригер GitHub Actions workflow для retrain"""
    if not GITHUB_TOKEN:
        logger.warning("GITHUB_TOKEN not set, skipping workflow trigger")
        return False

    url = f"https://api.github.com/repos/{GITHUB_REPO}/actions/workflows/{GITHUB_WORKFLOW}/dispatches"

    headers = {
        "Authorization": f"token {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json",
    }

    data = {
        "ref": "final-project",
        "inputs": {"retrain": "true"},
    }

    try:
        resp = requests.post(url, json=data, headers=headers, timeout=10)
        if resp.status_code == 204:
            logger.info("✓ GitHub Actions workflow triggered successfully")
            return True
        else:
            logger.error(f"✗ Failed to trigger workflow: {resp.status_code} {resp.text}")
            return False
    except Exception as e:
        logger.error(f"✗ Error triggering workflow: {e}")
        return False

def initialize_drift_detector():
    """Ініціалізація TabularDrift детектора"""
    global drift_detector
    if not _ALIBI_AVAILABLE or DRIFT_BACKEND != "tabular":
        return False

    with _state_lock:
        if len(reference_data) < MIN_REFERENCE_SAMPLES:
            logger.info(f"Collecting reference data: {len(reference_data)}/{MIN_REFERENCE_SAMPLES}")
            return False
        try:
            X_ref = np.array(reference_data[:MAX_REFERENCE_SAMPLES])
            drift_detector = TabularDrift(
                x_ref=X_ref,
                p_val=0.05,
                categories_per_feature=None,
                preprocess_at_init=True,
            )
            logger.info(f"✓ TabularDrift initialized with {X_ref.shape[0]} reference samples")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize TabularDrift: {e}")
            return False

def detect_drift_tabular(features: np.ndarray) -> bool:
    global drift_detector, _drift_events

    # Збір референсу / lazy init
    with _state_lock:
        if drift_detector is None:
            if len(reference_data) < MAX_REFERENCE_SAMPLES:
                reference_data.append(features.tolist())
            if len(reference_data) >= MIN_REFERENCE_SAMPLES:
                initialize_drift_detector()
            return False

    try:
        X_test = features.reshape(1, -1)
        preds = drift_detector.predict(X_test)  # головне виправлення: без drift_type
        is_drift = preds['data']['is_drift'] == 1
        if is_drift:
            logger.warning("🚨 Drift detected by TabularDrift!")
            with _state_lock:
                _drift_events += 1
                drift_counter.inc()
                if _drift_events % 5 == 0:
                    Thread(target=trigger_retrain_workflow, daemon=True).start()
        return is_drift
    except Exception as e:
        logger.error(f"TabularDrift prediction failed: {e}, falling back to Z-score")
        return detect_drift_zscore(features)

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
        logger.warning(f"🚨 Drift detected (Z-score)! Max Z: {float(np.max(z_scores)):.2f}")
        with _state_lock:
            _drift_events += 1
            drift_counter.inc()
            if _drift_events % 5 == 0:
                Thread(target=trigger_retrain_workflow, daemon=True).start()

    return drift_detected

def detect_drift(features: np.ndarray) -> bool:
    if DRIFT_BACKEND == "tabular" and _ALIBI_AVAILABLE:
        return detect_drift_tabular(features)
    else:
        return detect_drift_zscore(features)

def predict(features: List[float]) -> dict:
    try:
        X = np.array(features).reshape(1, -1)
        if X.shape[1] != 20:
            raise ValueError(f"Expected 20 features, got {X.shape[1]}")

        # Передбачення (працює з більшістю sklearn моделей)
        prediction = int(model.predict(X)[0])
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(X)[0]
            max_prob = float(np.max(probabilities))
        else:
            # fallback для моделей без predict_proba
            max_prob = 1.0

        drift = detect_drift(X[0])
        return {"prediction": prediction, "probability": max_prob, "drift_detected": drift}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    logger.info(f"Request: {len(request.features)} features")

    with prediction_latency.time():
        try:
            result = predict(request.features)
            prediction_counter.inc()
            response = PredictionResponse(
                prediction=result["prediction"],
                probability=result["probability"],
                drift_detected=result["drift_detected"],
                timestamp=datetime.now().isoformat(),
            )
            logger.info(
                f"Response: pred={response.prediction}, prob={response.probability:.4f}, drift={response.drift_detected}"
            )
            return response
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    detector_status = "not_initialized" if drift_detector is None else "ready"
    with _state_lock:
        ref_samples = len(reference_data) if drift_detector is None else "N/A"
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "drift_backend": DRIFT_BACKEND,
        "drift_detector": detector_status,
        "reference_samples": ref_samples,
    }

@app.get("/drift-stats")
async def drift_stats():
    with _state_lock:
        ref_count = len(reference_data)
        total_drifts = int(drift_counter._value.get()) if hasattr(drift_counter, "_value") else None
    return {
        "backend": DRIFT_BACKEND,
        "alibi_available": _ALIBI_AVAILABLE,
        "detector_initialized": drift_detector is not None,
        "reference_samples_collected": ref_count,
        "min_samples_required": MIN_REFERENCE_SAMPLES,
        "total_drift_detections": total_drifts,
    }

@app.get("/")
async def root():
    return {
        "service": "ML Inference API",
        "version": "2.0.1",
        "drift_backend": DRIFT_BACKEND,
        "endpoints": ["/predict", "/metrics", "/health", "/drift-stats"],
    }