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
reference_data = []  # Для накопичення референсних даних
feature_stats = None  # Fallback для Z-score

# GitHub webhook
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "YOUR_USERNAME/aiops-quality-project")
GITHUB_WORKFLOW = "ci-cd.yaml"

# Конфігурація
DRIFT_BACKEND = os.getenv("DRIFT_BACKEND", "tabular")  # tabular або zscore
MIN_REFERENCE_SAMPLES = 100  # Мінімум зразків для ініціалізації детектора
MAX_REFERENCE_SAMPLES = 500  # Максимум зразків для референсу

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
            model = pickle.load(f)
        logger.info(f"✓ Model loaded successfully from {model_path}")
        
        # Ініціалізація статистики для Z-score fallback
        feature_stats = {
            "mean": np.zeros(20),
            "std": np.ones(20),
            "count": 0
        }
        
        # Ініціалізація референсних даних
        reference_data = []
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
        "Accept": "application/vnd.github.v3+json"
    }
    
    data = {
        "ref": "final-project",
        "inputs": {"retrain": "true"}
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        if response.status_code == 204:
            logger.info("✓ GitHub Actions workflow triggered successfully")
            return True
        else:
            logger.error(f"✗ Failed to trigger workflow: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"✗ Error triggering workflow: {e}")
        return False

def initialize_drift_detector():
    """Ініціалізація TabularDrift детектора"""
    global drift_detector, reference_data
    
    if not _ALIBI_AVAILABLE or DRIFT_BACKEND != "tabular":
        return False
    
    if len(reference_data) < MIN_REFERENCE_SAMPLES:
        logger.info(f"Collecting reference data: {len(reference_data)}/{MIN_REFERENCE_SAMPLES}")
        return False
    
    try:
        # Конвертуємо в numpy array
        X_ref = np.array(reference_data[:MAX_REFERENCE_SAMPLES])
        
        # Створюємо TabularDrift детектор
        drift_detector = TabularDrift(
            x_ref=X_ref,
            p_val=0.05,  # Поріг p-value (5%)
            categories_per_feature=None,  # Усі фічі числові
            preprocess_at_init=True
        )
        
        logger.info(f"✓ TabularDrift detector initialized with {X_ref.shape[0]} reference samples")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize TabularDrift: {e}")
        return False

def detect_drift_tabular(features: np.ndarray) -> bool:
    """Перевірка дрейфу через TabularDrift"""
    global drift_detector, reference_data
    
    # Якщо детектор не ініціалізований - збираємо дані
    if drift_detector is None:
        if len(reference_data) < MAX_REFERENCE_SAMPLES:
            reference_data.append(features.tolist())
        
        # Спробуємо ініціалізувати детектор
        if len(reference_data) >= MIN_REFERENCE_SAMPLES:
            initialize_drift_detector()
        
        # Поки немає детектора - drift не виявляємо
        return False
    
    try:
        # Перевірка на drift
        X_test = features.reshape(1, -1)
        preds = drift_detector.predict(X_test, drift_type='batch')
        
        # Перевірка результату
        is_drift = preds['data']['is_drift'] == 1
        
        if is_drift:
            p_val = preds['data'].get('p_val', 0.0)
            logger.warning(f"🚨 Drift detected by TabularDrift! p-value: {p_val:.6f}")
            drift_counter.inc()
            
            # Тригер retrain кожні 5 drift детекцій
            if int(drift_counter._value.get()) % 5 == 0:
                logger.info("⚠️  Triggering retrain due to repeated drift...")
                trigger_retrain_workflow()
        
        return is_drift
        
    except Exception as e:
        logger.error(f"TabularDrift prediction failed: {e}, falling back to Z-score")
        # При помилці переходимо на Z-score
        return detect_drift_zscore(features)

def detect_drift_zscore(features: np.ndarray) -> bool:
    """Z-score drift detection (fallback)"""
    global feature_stats
    
    # Оновлюємо статистику
    if feature_stats["count"] == 0:
        feature_stats["mean"] = features
        feature_stats["std"] = np.ones_like(features)
    else:
        # Exponential moving average
        alpha = 0.1
        feature_stats["mean"] = (1 - alpha) * feature_stats["mean"] + alpha * features
        feature_stats["std"] = (1 - alpha) * feature_stats["std"] + alpha * np.abs(features - feature_stats["mean"])
    
    feature_stats["count"] += 1
    
    # Z-score перевірка (> 3 sigma = аномалія)
    z_scores = np.abs((features - feature_stats["mean"]) / (feature_stats["std"] + 1e-6))
    drift_detected = np.any(z_scores > 3.0)
    
    if drift_detected:
        logger.warning(f"🚨 Drift detected (Z-score)! Max Z: {np.max(z_scores):.2f}")
        drift_counter.inc()
        
        if int(drift_counter._value.get()) % 5 == 0:
            logger.info("⚠️  Triggering retrain due to repeated drift...")
            trigger_retrain_workflow()
    
    return drift_detected

def detect_drift(features: np.ndarray) -> bool:
    """Головна функція виявлення дрейфу"""
    if DRIFT_BACKEND == "tabular" and _ALIBI_AVAILABLE:
        return detect_drift_tabular(features)
    else:
        return detect_drift_zscore(features)

def predict(features: List[float]) -> dict:
    """Основна функція передбачення"""
    try:
        X = np.array(features).reshape(1, -1)
        
        if X.shape[1] != 20:
            raise ValueError(f"Expected 20 features, got {X.shape[1]}")
        
        # Передбачення (працює з усіма 4 типами моделей)
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        max_prob = float(np.max(probabilities))
        
        # Drift detection
        drift = detect_drift(X[0])
        
        return {
            "prediction": int(prediction),
            "probability": max_prob,
            "drift_detected": drift
        }
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise

@app.post("/predict", response_model=PredictionResponse)
async def make_prediction(request: PredictionRequest):
    """Endpoint для передбачення"""
    logger.info(f"Request: {len(request.features)} features")
    
    with prediction_latency.time():
        try:
            result = predict(request.features)
            prediction_counter.inc()
            
            response = PredictionResponse(
                prediction=result["prediction"],
                probability=result["probability"],
                drift_detected=result["drift_detected"],
                timestamp=datetime.now().isoformat()
            )
            
            logger.info(f"Response: pred={response.prediction}, prob={response.probability:.4f}, drift={response.drift_detected}")
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
async def metrics():
    """Prometheus metrics"""
    return Response(content=generate_latest(), media_type="text/plain")

@app.get("/health")
async def health():
    """Health check"""
    detector_status = "not_initialized" if drift_detector is None else "ready"
    ref_samples = len(reference_data) if drift_detector is None else "N/A"
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "drift_backend": DRIFT_BACKEND,
        "drift_detector": detector_status,
        "reference_samples": ref_samples
    }

@app.get("/drift-stats")
async def drift_stats():
    """Статистика drift детектора"""
    return {
        "backend": DRIFT_BACKEND,
        "alibi_available": _ALIBI_AVAILABLE,
        "detector_initialized": drift_detector is not None,
        "reference_samples_collected": len(reference_data),
        "min_samples_required": MIN_REFERENCE_SAMPLES,
        "total_drift_detections": int(drift_counter._value.get())
    }

@app.get("/")
async def root():
    return {
        "service": "ML Inference API",
        "version": "2.0.0",
        "drift_backend": DRIFT_BACKEND,
        "endpoints": ["/predict", "/metrics", "/health", "/drift-stats"]
    }