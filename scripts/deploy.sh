#!/bin/bash

# Скрипт для швидкого деплою ML Inference Service
# Використання: ./scripts/deploy.sh [rebuild|redeploy|clean]

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  ML Inference Service Deployment   ${NC}"
echo -e "${GREEN}=====================================${NC}\n"

# Функція для перевірки kubectl
check_kubectl() {
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}kubectl не знайдено. Встановіть kubectl.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ kubectl знайдено${NC}"
}

# Функція для перевірки Helm
check_helm() {
    if ! command -v helm &> /dev/null; then
        echo -e "${RED}helm не знайдено. Встановіть Helm.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ helm знайдено${NC}"
}

# Функція для перевірки Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}docker не знайдено. Встановіть Docker.${NC}"
        exit 1
    fi
    echo -e "${GREEN}✓ docker знайдено${NC}"
}

# Функція для тренування моделі
train_model() {
    echo -e "\n${YELLOW}Тренування моделі...${NC}"
    python model/train.py
    echo -e "${GREEN}✓ Модель натреновано${NC}"
}

# Функція для білду Docker образу
build_image() {
    echo -e "\n${YELLOW}Білд Docker образу...${NC}"

    mkdir -p app/models
    cp models/model.pkl app/models/
    
    cd app
    docker build -t ml-inference-service:latest .
    cd ..
    
    echo -e "${GREEN}✓ Docker образ зібрано${NC}"
}

# Функція для створення namespace
create_namespace() {
    if kubectl get namespace ml-service &> /dev/null; then
        echo -e "${GREEN}✓ Namespace ml-service вже існує${NC}"
    else
        echo -e "\n${YELLOW}Створення namespace ml-service...${NC}"
        kubectl create namespace ml-service
        echo -e "${GREEN}✓ Namespace створено${NC}"
    fi
}

# Функція для деплою через Helm
deploy_helm() {
    echo -e "\n${YELLOW}Деплой через Helm...${NC}"

    create_namespace

    echo -e "${YELLOW}Встановлення/оновлення Helm релізу...${NC}"
    helm upgrade --install ml-service ./helm --namespace ml-service --wait

    echo -e "${GREEN}✓ Helm деплой завершено${NC}"
}

# Функція для перевірки статусу
check_status() {
    echo -e "\n${YELLOW}Перевірка статусу...${NC}"
    
    kubectl get pods -n ml-service
    kubectl get svc -n ml-service
    
    echo -e "\n${GREEN}Чекаємо на готовність подів...${NC}"
    kubectl wait --for=condition=Ready pods -l app.kubernetes.io/name=ml-inference-service -n ml-service --timeout=120s
    
    echo -e "${GREEN}✓ Сервіс запущено${NC}"
}

# Функція для очищення
clean() {
    echo -e "\n${YELLOW}Очищення ресурсів...${NC}"
    
    helm uninstall ml-service -n ml-service 2>/dev/null || true
    kubectl delete namespace ml-service 2>/dev/null || true
    
    echo -e "${GREEN}✓ Очищення завершено${NC}"
}

# Функція для port-forward
port_forward() {
    echo -e "\n${YELLOW}Запуск port-forward...${NC}"
    echo -e "${GREEN}API буде доступний на: http://localhost:8000${NC}"
    echo -e "${YELLOW}Натисніть Ctrl+C для зупинки${NC}\n"
    
    kubectl port-forward -n ml-service svc/ml-service-ml-inference-service 8000:8000
}

# Головна логіка
main() {
    check_kubectl
    check_helm
    check_docker
    
    case "${1:-deploy}" in
        rebuild)
            train_model
            build_image
            deploy_helm
            check_status
            echo -e "\n${GREEN}✓ Rebuild завершено!${NC}"
            echo -e "${YELLOW}Запустіть: ./scripts/deploy.sh port-forward${NC}"
            ;;
        redeploy)
            deploy_helm
            check_status
            echo -e "\n${GREEN}✓ Redeploy завершено!${NC}"
            ;;
        clean)
            clean
            ;;
        port-forward|pf)
            port_forward
            ;;
        test)
            echo -e "\n${YELLOW}Запуск тестів...${NC}"
            python tests/test_drift.py
            ;;
        *)
            echo -e "${YELLOW}Використання:${NC}"
            echo -e "  ./scripts/deploy.sh ${GREEN}rebuild${NC}      - Повний rebuild (train + build + deploy)"
            echo -e "  ./scripts/deploy.sh ${GREEN}redeploy${NC}     - Redeploy без rebuild"
            echo -e "  ./scripts/deploy.sh ${GREEN}clean${NC}        - Видалити всі ресурси"
            echo -e "  ./scripts/deploy.sh ${GREEN}port-forward${NC} - Port-forward до API"
            echo -e "  ./scripts/deploy.sh ${GREEN}test${NC}         - Запустити тести"
            exit 1
            ;;
    esac
}

main "$@"