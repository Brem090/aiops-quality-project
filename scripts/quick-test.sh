#!/bin/bash
# Quick Test Script for ML Inference Service
# Використання: ./scripts/quick-test.sh
# Перевіряє основні ендпоїнти: /health, /predict, /metrics, drift detection.

set -o pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

API_URL="http://localhost:8000"

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}   Quick Test: ML Inference Service  ${NC}"
echo -e "${GREEN}=====================================${NC}\n"
echo "Started at: $(date)"
echo

# -------------------- TEST 1: Health --------------------
echo -e "${YELLOW}[1/5] Testing health endpoint...${NC}"
sleep 2
HEALTH_HTTP=$(curl -s -o /tmp/health.json -w "%{http_code}" ${API_URL}/health || echo 000)

if [ "$HEALTH_HTTP" == "200" ] && grep -q "healthy" /tmp/health.json; then
    echo -e "${GREEN}✓ Health check passed${NC}"
else
    echo -e "${RED}✗ Health check failed (HTTP $HEALTH_HTTP)${NC}"
    cat /tmp/health.json || true
    exit 1
fi

# -------------------- TEST 2: Prediction --------------------
echo -e "\n${YELLOW}[2/5] Testing prediction endpoint...${NC}"
PAYLOAD='{"features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]}'

HTTP_CODE=$(curl -s -o /tmp/predict.json -w "%{http_code}" -X POST ${API_URL}/predict \
  -H "Content-Type: application/json" -d "${PAYLOAD}" || echo 000)

if [ "$HTTP_CODE" == "200" ] && grep -q "prediction" /tmp/predict.json; then
    echo -e "${GREEN}✓ Prediction successful${NC}"
    python -m json.tool < /tmp/predict.json
else
    echo -e "${RED}✗ Prediction failed (HTTP $HTTP_CODE)${NC}"
    cat /tmp/predict.json || true
    exit 1
fi

# -------------------- TEST 3: Metrics --------------------
echo -e "\n${YELLOW}[3/5] Testing metrics endpoint...${NC}"
METRICS_HTTP=$(curl -s -o /tmp/metrics.txt -w "%{http_code}" ${API_URL}/metrics || echo 000)

if [ "$METRICS_HTTP" == "200" ] && grep -q "predictions_total" /tmp/metrics.txt; then
    echo -e "${GREEN}✓ Metrics endpoint working${NC}"
    echo -e "${YELLOW}Sample metrics:${NC}"
    grep -E "(predictions_total|drift_detected_total)" /tmp/metrics.txt | grep -v "^#" | head -5
else
    echo -e "${RED}✗ Metrics endpoint failed (HTTP $METRICS_HTTP)${NC}"
    exit 1
fi

# -------------------- TEST 4: Multiple Predictions --------------------
echo -e "\n${YELLOW}[4/5] Testing multiple predictions...${NC}"
SUCCESS_COUNT=0

for i in {1..10}; do
    RESULT_HTTP=$(curl -s -o /tmp/multi_predict.json -w "%{http_code}" -X POST ${API_URL}/predict \
      -H "Content-Type: application/json" -d "${PAYLOAD}" || echo 000)
    if [ "$RESULT_HTTP" == "200" ] && grep -q "prediction" /tmp/multi_predict.json; then
        ((SUCCESS_COUNT++))
    fi
    sleep 0.1
done

if [ $SUCCESS_COUNT -eq 10 ]; then
    echo -e "${GREEN}✓ All 10 predictions successful${NC}"
else
    echo -e "${YELLOW}⚠ Only $SUCCESS_COUNT/10 predictions successful${NC}"
fi

# -------------------- TEST 5: Drift Detection --------------------
echo -e "\n${YELLOW}[5/5] Testing drift detection with anomalous data...${NC}"
DRIFT_PAYLOAD='{"features": [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]}'
DRIFT_HTTP=$(curl -s -o /tmp/drift.json -w "%{http_code}" -X POST ${API_URL}/predict \
  -H "Content-Type: application/json" -d "${DRIFT_PAYLOAD}" || echo 000)

python -m json.tool < /tmp/drift.json || true

if [ "$DRIFT_HTTP" == "200" ] && grep -q '"drift_detected": true' /tmp/drift.json; then
    echo -e "${GREEN}✓ Drift detection is working${NC}"
else
    echo -e "${YELLOW}⚠ Drift not detected or field missing${NC}"
fi

# -------------------- Summary --------------------
echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN}   All basic tests completed!        ${NC}"
echo -e "${GREEN}=====================================${NC}\n"

echo -e "${YELLOW}Current metrics summary:${NC}"
grep -E "(predictions_total|drift_detected_total)" /tmp/metrics.txt | grep -v "^#" || true

echo -e "\n${YELLOW}To run full drift test:${NC}"
echo -e "  python tests/test_drift.py"

echo -e "\n${YELLOW}To view logs:${NC}"
echo -e "  kubectl logs -n ml-service -l app.kubernetes.io/name=ml-inference-service -f"

echo -e "\nFinished at: $(date)"
