from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest
from fastapi.responses import Response
import logging
from datetime import datetime
from typing import List
import os
import requests

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
feature_stats = None

# GitHub webhook
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GITHUB_REPO = os.getenv("GITHUB_REPO", "YOUR_USERNAME/aiops-quality-project")
GITHUB_WORKFLOW = "ci-cd.yaml"

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    drift_detected: bool
    timestamp: str

def load_model():
    """Завантаження моделі при старті"""
    global model, feature_stats
    
    model_path = os.getenv("MODEL_PATH", "/app/models/model.pkl")
    
    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"✓ Model loaded successfully from {model_path}")
        
        # Ініціалізація статистики для drift detection
        feature_stats = {
            "mean": np.zeros(20),
            "std": np.ones(20),
            "count": 0
        }
        
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

def detect_drift(features: np.ndarray) -> bool:
    """Z-score drift detection"""
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
        logger.warning(f"🚨 Drift detected! Max Z-score: {np.max(z_scores):.2f}")
        drift_counter.inc()
        
        # Тригер retrain кожні 5 drift детекцій
        if int(drift_counter._value.get()) % 5 == 0:
            logger.info("⚠️  Triggering retrain due to repeated drift...")
            trigger_retrain_workflow()
    
    return drift_detected

def predict(features: List[float]) -> dict:
    """Основна функція передбачення"""
    try:
        X = np.array(features).reshape(1, -1)
        
        if X.shape[1] != 20:
            raise ValueError(f"Expected 20 features, got {X.shape[1]}")
        
        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]
        max_prob = float(np.max(probabilities))
        
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
    return {"status": "healthy", "model_loaded": model is not None}

@app.get("/")
async def root():
    return {
        "service": "ML Inference API",
        "version": "1.0.0",
        "endpoints": ["/predict", "/metrics", "/health"]
    }
