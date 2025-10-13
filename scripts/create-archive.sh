#!/bin/bash
set -euo pipefail

# Скрипт для створення архіву для здачі проєкту

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=====================================${NC}"
echo -e "${GREEN}  Creating Project Archive          ${NC}"
echo -e "${GREEN}=====================================${NC}\n"

read -p "Введіть ваше прізвище: " SURNAME
read -p "Введіть ваше ім'я: " NAME

ARCHIVE_NAME="FinalProject_${SURNAME}_${NAME}.zip"

if [ ! -f "README.md" ] || [ ! -d "app" ] || [ ! -d "helm" ]; then
    echo -e "${YELLOW}Помилка: запустіть скрипт з кореня проєкту${NC}"
    exit 1
fi

echo -e "${YELLOW}Очищення тимчасових файлів...${NC}"
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
find . -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.log" \) -delete 2>/dev/null || true

echo -e "${YELLOW}Створення архіву...${NC}"

if command -v powershell &> /dev/null; then
    powershell -Command "Compress-Archive -Path .\* -DestinationPath ..\${ARCHIVE_NAME} -Force"
    echo -e "${GREEN}✓ Архів створено: ../${ARCHIVE_NAME}${NC}"
elif command -v zip &> /dev/null; then
    cd ..
    zip -r "${ARCHIVE_NAME}" aiops-quality-project \
        -x "*/venv/*" "*/.git/*" "*/__pycache__/*" "*.pyc" "*.pyo" "*.log" "*.tmp" "*.swp"
    cd aiops-quality-project
    echo -e "${GREEN}✓ Архів створено: ../${ARCHIVE_NAME}${NC}"
else
    echo -e "${YELLOW}Помилка: zip не знайдено. Створіть архів вручну.${NC}"
    exit 1
fi

echo -e "\n${GREEN}=====================================${NC}"
echo -e "${GREEN}  Архів готовий до здачі!           ${NC}"
echo -e "${GREEN}=====================================${NC}\n"

echo -e "Файл: ${YELLOW}${ARCHIVE_NAME}${NC}"
echo -e "Розташування: ${YELLOW}$(cd .. && pwd)/${ARCHIVE_NAME}${NC}"

echo -e "\n${YELLOW}Наступні кроки:${NC}"
echo -e "1. Завантажте архів у LMS"
echo -e "2. Додайте посилання на GitHub репозиторій"
echo -e "3. Переконайтеся що гілка: ${GREEN}final-project${NC}"

echo -e "\n${YELLOW}Посилання на репозиторій:${NC}"
git remote get-url origin 2>/dev/null || echo "Git remote не налаштований"

echo -e "\n${GREEN}Готово! Успіхів! 🎉${NC}"
