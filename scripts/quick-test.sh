# Швидкий тест ML Inference Service

set -o pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

API_URL="http://localhost:8000"

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}   Швидкий тест: ML Inference Service ${NC}"
echo -e "${GREEN}=====================================${NC}\n"
echo "Початок роботи: $(date)"
echo

# -------------------- Тест 1: Health --------------------
echo -e "${YELLOW}[1/5] Перевірка ендпоїнту /health...${NC}"
sleep 2
HEALTH_HTTP=$(curl -s -o /tmp/health.json -w "%{http_code}" ${API_URL}/health || echo 000)

if [ "$HEALTH_HTTP" == "200" ] && grep -q "healthy" /tmp/health.json; then
    echo -e "${GREEN}✓ Перевірка здоров’я успішна${NC}"
else
    echo -e "${RED}✗ Перевірка здоров’я неуспішна (HTTP $HEALTH_HTTP)${NC}"
    cat /tmp/health.json || true
    exit 1
fi

# -------------------- Тест 2: Prediction --------------------
echo -e "\n${YELLOW}[2/5] Перевірка ендпоїнту /predict...${NC}"
PAYLOAD='{"features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]}'

HTTP_CODE=$(curl -s -o /tmp/predict.json -w "%{http_code}" -X POST ${API_URL}/predict \
  -H "Content-Type: application/json" -d "${PAYLOAD}" || echo 000)

if [ "$HTTP_CODE" == "200" ] && grep -q "prediction" /tmp/predict.json; then
    echo -e "${GREEN}✓ Передбачення виконано успішно${NC}"
    python -m json.tool < /tmp/predict.json
else
    echo -e "${RED}✗ Помилка під час передбачення (HTTP $HTTP_CODE)${NC}"
    cat /tmp/predict.json || true
    exit 1
fi

# -------------------- Тест 3: Metrics --------------------
echo -e "\n${YELLOW}[3/5] Перевірка ендпоїнту /metrics...${NC}"
METRICS_HTTP=$(curl -s -o /tmp/metrics.txt -w "%{http_code}" ${API_URL}/metrics || echo 000)

if [ "$METRICS_HTTP" == "200" ] && grep -q "predictions_total" /tmp/metrics.txt; then
    echo -e "${GREEN}✓ Ендпоїнт метрик працює${NC}"
    echo -e "${YELLOW}Приклад метрик:${NC}"
    grep -E "(predictions_total|drift_detected_total)" /tmp/metrics.txt | grep -v "^#" | head -5
else
    echo -e "${RED}✗ Не вдалося отримати метрики (HTTP $METRICS_HTTP)${NC}"
    exit 1
fi

# -------------------- Тест 4: Кілька передбачень --------------------
echo -e "\n${YELLOW}[4/5] Тестування кількох передбачень...${NC}"
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
    echo -e "${GREEN}✓ Усі 10 передбачень успішні${NC}"
else
    echo -e "${YELLOW}⚠ Успішно лише $SUCCESS_COUNT із 10 передбачень${NC}"
fi

# -------------------- Тест 5: Drift Detection --------------------
echo -e "\n${YELLOW}[5/5] Перевірка виявлення дрейфу...${NC}"
DRIFT_PAYLOAD='{"features": [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10]}'
DRIFT_HTTP=$(curl -s -o /tmp/drift.json -w "%{http_code}" -X POST ${API_URL}/predict \
  -H "Content-Type: application/json" -d "${DRIFT_PAYLOAD}" || echo 000)

python -m json.tool < /tmp/drift.json || true

if [ "$DRIFT_HTTP" == "200" ] && grep -q '"drift_detected": true' /tmp/drift.json; then
    echo -e "${GREEN}✓ Механізм виявлення дрейфу працює${NC}"
else
    echo -e "${YELLOW}⚠ Дрейф не виявлено або поле відсутнє${NC}"
fi

# -------------------- Підсумок --------------------
echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN}   Усі базові тести завершено!       ${NC}"
echo -e "${GREEN}=====================================${NC}\n"

echo -e "${YELLOW}Поточні метрики:${NC}"
grep -E "(predictions_total|drift_detected_total)" /tmp/metrics.txt | grep -v "^#" || true

echo -e "\n${YELLOW}Для повного тесту дрейфу запустіть:${NC}"
echo -e "  python tests/test_drift.py"

echo -e "\n${YELLOW}Щоб переглянути логи:${NC}"
echo -e "  kubectl logs -n ml-service -l app.kubernetes.io/name=ml-inference-service -f"

echo -e "\nЗавершено: $(date)"
echo
read -p "Натисніть Enter, щоб закрити..."
