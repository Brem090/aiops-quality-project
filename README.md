# AIOps Quality Project

Фінальний проєкт курсу з розробки MLOps-системи для інференсу моделі машинного навчання з автоматичним виявленням дрейфу даних.

## Про проєкт

Цей проєкт демонструє повний цикл розробки production-ready ML-сервісу: від тренування моделі до автоматизованого деплою через GitOps. Система автоматично виявляє відхилення у вхідних даних (drift detection) і може перезапускати тренування моделі через CI/CD-пайплайн.

## Зміст

- [Як влаштована система](#як-влаштована-система)
- [Що потрібно для запуску](#що-потрібно-для-запуску)
- [Встановлення](#встановлення)
- [Запуск](#запуск)
- [Як тестувати](#як-тестувати)
- [Моніторинг](#моніторинг)
- [CI/CD та Оновлення моделі](#cicd-та-оновлення-моделі)
- [Скріншоти](#скріншоти)
- [Troubleshooting](#troubleshooting)
- [Структура проєкту](#структура-проєкту)

## Як влаштована система

Проєкт складається з декількох компонентів, які працюють разом:

```
GitHub (вихідний код)
    ↓
GitHub Actions (автоматичне тестування, білд та ретрейн)
    ↓
Docker образ (контейнер з моделлю)
    ↓
ArgoCD (автоматичний деплой у Kubernetes)
    ↓
FastAPI сервіс (приймає запити та робить передбачення)
    ↓
Prometheus + Grafana (збирають метрики та візуалізують дашборди)
Loki + Promtail (Promtail збирає, Loki агрегує та зберігає логи)
```

### Основні компоненти

- **FastAPI сервіс** - приймає JSON-запити з 20 числовими ознаками, робить передбачення за допомогою натренованої моделі (наприклад, RandomForest) і повертає результат разом з імовірністю.

- **Drift Detector** - аналізує кожен запит за допомогою Z-score статистики. Якщо дані суттєво відрізняються від "нормальних" (більше ніж на поріг, який налаштовується через `ZSCORE_THRESHOLD`, за замовчуванням 3.0), система логує це як drift.

- **Prometheus** - збирає метрики: кількість запитів, швидкість відповіді API, кількість виявлених дрейфів.

- **Grafana** - візуалізує всі ці метрики у вигляді дашбордів у реальному часі.

- **Loki** - зберігає всі логи з сервісу, щоб можна було переглянути, що відбувалось у минулому.

- **ArgoCD** - слідкує за GitHub-репозиторієм і автоматично оновлює сервіс у Kubernetes після кожного коміту в цільову гілку.

## Що потрібно для запуску

### Програмне забезпечення

- Docker Desktop з увімкненим Kubernetes
- Python 3.11+
- Helm 3
- kubectl
- Git
- GitHub Account

### Перевірка встановлення

```bash
# Docker Desktop Kubernetes
kubectl version --short
# Очікуваний результат: Client/Server Version...

# Helm
helm version --short
# Очікуваний результат: v3.x.x

# Python
python --version
# Очікуваний результат: Python 3.11.x
```

Якщо всі команди працюють - можна починати.

## Встановлення

### Крок 1: Клонування репозиторію

```bash
# Клонуємо репозиторій (замініть YOUR_USERNAME)
git clone https://github.com/YOUR_USERNAME/aiops-quality-project.git
cd aiops-quality-project

# Переключаємось на гілку final-project
git checkout final-project
```

### Крок 2: Встановлення Python залежностей

```bash
# Створюємо віртуальне середовище
python -m venv venv

# Активуємо (Linux/Mac)
source venv/bin/activate

# Активуємо (Windows PowerShell)
# .\venv\Scripts\Activate.ps1

# Встановлюємо залежності
pip install -r app/requirements.txt
```

### Крок 3: Тренування початкової моделі

```bash
# Тренуємо модель
python model/train.py

# Перевіряємо, що модель створилась
ls models/
# Повинен бути: model.pkl, metadata.pkl
```

### Крок 4: Білд Docker образу (для локальних тестів)

```bash
# Переходимо в папку app
cd app

# Копіюємо модель у підпапку models всередині app
mkdir -p models
cp ../models/model.pkl models/model.pkl

# Будуємо Docker образ
# Для локального запуску можна залишити ml-inference-service
docker build -t ml-inference-service:latest .

# Перевіряємо
docker images | grep "ml-inference-service"

# Повертаємось у корінь проєкту
cd ..
```

### Крок 5: Встановлення ArgoCD

```bash
# Створюємо namespace
kubectl create namespace argocd

# Встановлюємо ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Чекаємо на готовність
kubectl wait --for=condition=Ready pods --all -n argocd --timeout=300s

# Отримуємо пароль admin (для Linux/macOS)
kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d; echo

# (Для Windows PowerShell)
[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String((kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}")))

# Port-forward для UI (в окремому терміналі)
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

**Відкрийте:** https://localhost:8080
- **Login:** admin
- **Password:** (з команди вище)

### Крок 6: Встановлення Prometheus та Grafana

```bash
# Додаємо Helm репозиторії
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Встановлюємо kube-prometheus-stack
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --wait \
  --set grafana.enabled=true \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false \
  --set nodeExporter.enabled=false \
  --set grafana.additionalDataSources[0].name=Loki \
  --set grafana.additionalDataSources[0].type=loki \
  --set grafana.additionalDataSources[0].access=proxy \
  --set grafana.additionalDataSources[0].url=http://loki.monitoring.svc.cluster.local:3100

# Чекаємо на готовність
kubectl wait --for=condition=Ready pods --all -n monitoring --timeout=300s
```

### Крок 7: Встановлення Loki Stack

Для уникнення проблем сумісності Grafana та Loki, одразу оновлюємо версії.

```bash
# Встановлюємо Loki Stack
helm upgrade --install loki grafana/loki-stack \
  --namespace monitoring \
  --wait \
  --set loki.enabled=true \
  --set promtail.enabled=true \
  --set fluent-bit.enabled=false \
  --set grafana.enabled=false \
  --set loki.persistence.enabled=true \
  --set loki.persistence.size=1Gi \
  --set loki.auth_enabled=false \
  --set loki.image.tag=2.9.4 \
  --set promtail.image.tag=2.9.4

# Перевіряємо чи усе добре
kubectl get pods -n monitoring
```

## Запуск

### Метод 1: Через Helm (для швидкого тесту)

```bash
# Створюємо namespace для сервісу
kubectl create namespace ml-service

# Встановлюємо через Helm (реліз називаємо "ml-service")
helm install ml-service ./helm --namespace ml-service

# Перевіряємо статус
kubectl get pods -n ml-service
kubectl get svc -n ml-service
```

### Метод 2: Через ArgoCD (GitOps - рекомендований)

```bash
# Оновіть argocd/application.yaml з вашим GitHub репозиторієм
# Замініть YOUR_USERNAME на ваш GitHub username у полі repoURL

# Перейдіть у кореневу директорію проєкту
# cd aiops-quality-project

# Застосовуємо Application
kubectl apply -f argocd/application.yaml -n argocd

# Перевіряємо статус
kubectl get application -n argocd ml-inference-service

# Перевіряємо статус у ArgoCD UI
# https://localhost:8080
```

ArgoCD автоматично:
1. Синхронізує Helm chart з Git.
2. Задеплоїть сервіс у namespace `ml-service`.
3. Автоматично оновлюватиме сервіс при змінах у репозиторії.

### Перевірка деплою

```bash
# Переглянути поди
kubectl get pods -n ml-service

# Очікуваний результат:
# NAME                     READY   STATUS    RESTARTS   AGE
# ml-inference-service-..  1/1     Running   0          2m

# Переглянути логи
kubectl logs -n ml-service -l app.kubernetes.io/name=ml-inference-service -f
```

## Як тестувати

### 1. Port-forward до сервісу (окреме вікно терміналу)

```bash
# Завдяки fullnameOverride, ім'я сервісу стабільне
kubectl port-forward -n ml-service svc/ml-inference-service 8000:8000
```

### 2. Перевірка health endpoint (у новому вікні терміналу)

```bash
curl http://localhost:8000/health
```

Очікувана відповідь:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "detector_status": "ready",
  "ref_samples": 50
}
```

### 3. Тестовий запит на передбачення

#### Через curl:

```bash
curl -X POST "http://localhost:8000/predict" \
-H "Content-Type: application/json" \
-d '{
  "features": [
    0.5, -0.3, 1.2, 0.8, -0.5, 0.2, 0.9, -0.1,
    0.4, 0.7, -0.6, 0.3, 0.1, -0.4, 0.6, 0.2,
    -0.8, 0.5, 0.9, -0.2
  ]
}'
```

Очікувана відповідь:

```json
{
  "prediction": 1,
  "probability": 0.6534023807361028,
  "drift_detected": false,
  "timestamp": "..."
}
```

#### Через Python скрипт (генерація дрейфу):

```bash
# Встановлюємо 'requests' для тестового скрипта
pip install requests

# Запускаємо тестовий скрипт
python tests/test_drift.py
```

Скрипт відправить:
- 30 нормальних запитів
- 20 аномальних запитів (для тригера drift)

### 4. Перевірка метрик

```bash
curl http://localhost:8000/metrics
```

Шукайте метрики:
- `predictions_total` - загальна кількість передбачень
- `drift_detected_total` - кількість виявлених дрейфів
- `prediction_latency_seconds` - час відповіді

## Моніторинг

### Grafana Dashboard

#### Доступ до Grafana

```bash
# Отримуємо пароль admin (для Linux/macOS)
kubectl get secret -n monitoring monitoring-grafana -o jsonpath="{.data.admin-password}" | base64 -d; echo

# (Для Windows PowerShell)
[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String((kubectl get secret -n monitoring monitoring-grafana -o jsonpath="{.data.admin-password}")))

# Port-forward (окреме вікно)
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80
```

**Відкрийте:** http://localhost:3000
- **Login:** admin
- **Password:** (з команди вище)

#### Імпорт Dashboard

1. У Grafana UI: Dashboards → Import
2. Upload `grafana/dashboard.json`
3. Натисніть Import

#### Панелі Dashboard:

- **Predictions Per Second** - кількість запитів/сек
- **Prediction Latency** - p50, p95, p99 латентність
- **Total Drift Detections** - загальна кількість drift подій
- **Drift Detection Rate** - частота виявлення drift
- **Service Status** - статус сервісу (Up/Down)
- **Total Predictions** - загальна кількість передбачень
- **Service Logs** - логи з Loki

### Prometheus Metrics

```bash
# Port-forward до Prometheus (окреме вікно)
kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090
```

**Відкрийте:** http://localhost:9090

Prometheus знаходить цей сервіс завдяки ресурсу ServiceMonitor, який створюється Helm-чартом (див. `helm/templates/servicemonitor.yaml`).

#### Корисні PromQL запити (Graph):

```promql
# Кількість передбачень за останню хвилину
rate(predictions_total{namespace="ml-service"}[1m])

# 95-й персентиль латентності
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket{namespace="ml-service"}[5m]))

# Частота drift detection
rate(drift_detected_total{namespace="ml-service"}[5m])
```

### Loki Logs

У Grafana:
1. Explore → Loki
2. Запит для логів сервісу: `{namespace="ml-service"}`
3. Фільтр для дрейфу: `{namespace="ml-service"} |= "Drift detected"`

## CI/CD та Оновлення моделі

Цей проєкт використовує GitOps-підхід. Ручне оновлення образів або деплойментів не рекомендується. Весь процес контролюється через Git.

### GitHub Actions Workflow

**Workflow файл:** `.github/workflows/ci-cd.yaml`

#### Автоматичні тригери:

- **Push до main / final-project:** Запускає тести, білд Docker-образу та пуш його у Registry. ArgoCD автоматично побачить оновлення тегу образу (якщо пайплайн оновлює values.yaml) і оновить деплоймент.

- **Коміт з `[retrain]`:** Якщо повідомлення коміту містить `[retrain]`, пайплайн додатково запустить крок перетренування моделі (`model/train.py`) перед білдом і деплоєм.

- **Manual trigger:** Можна запустити вручну через GitHub UI.

### Конфігурація Registry та CI/CD

Ваш пайплайн CI/CD повинен мати доступ до Docker Registry (Docker Hub або GHCR) для пушу образів.

**Налаштування values.yaml:** Вкажіть шлях до вашого образу в `helm/values.yaml`:

```yaml
# helm/values.yaml
image:
  repository: ghcr.io/YOUR_USERNAME/aiops-quality-project/ml-inference-service
  tag: "latest" # Цей тег може оновлюватись автоматично через CI/CD
  pullPolicy: IfNotPresent
```

**Доступи для GitHub Actions (якщо використовуєте GHCR):** Переконайтесь, що ваш `GITHUB_TOKEN` у workflow має права `packages: write` для публікації образів у GitHub Container Registry.

### Ручний запуск перетренування (Retrain)

Це рекомендований спосіб оновлення моделі.

#### Метод 1: Через GitHub UI

1. Перейдіть: Actions → CI/CD Pipeline
2. Натисніть **Run workflow**
3. Виберіть гілку: **final-project**
4. Встановіть **retrain: true**
5. Натисніть **Run workflow**

#### Метод 2: Через коміт

```bash
# Створіть порожній коміт з магічним повідомленням
git commit --allow-empty -m "[retrain] Trigger model retraining"
git push origin final-project
```

### Перевірка пайплайну

```bash
# Перегляньте Actions на GitHub
# https://github.com/YOUR_USERNAME/aiops-quality-project/actions
```

Пайплайн виконає:
1. Тести (test job)
2. (Якщо `[retrain]`) Перетренування моделі (retrain-model job)
3. Білд та Пуш Docker образу (build-and-push job)
4. (Опційно) Оновлення Helm values.yaml з новим тегом образу

ArgoCD автоматично підхопить зміни та задеплоює нову версію сервісу.

## Скріншоти

Тут можна розмістити візуальні підтвердження роботи системи.

- **Grafana Dashboard:** Загальний вигляд дашборду, що показує метрики (RPS, Latency) та панель логів.
- **ArgoCD UI:** Скріншот "дерева" додатку зі статусами Synced та Healthy.
- **Drift Detection:** Збільшений скріншот панелі "Total Drift Detections" у Grafana та відповідні логи "Drift detected" у Loki.
- **GitHub Actions:** Скріншот успішного виконання `[retrain]` пайплайну.

## Troubleshooting

### Pod не запускається

```bash
# Дивимось статус
kubectl describe pod -n ml-service -l app.kubernetes.io/name=ml-inference-service

# Перевіряємо логи
kubectl logs -n ml-service -l app.kubernetes.io/name=ml-inference-service

# Типові проблеми:
# - ImagePullBackOff: образ не знайдено (перевірте `image.repository` та `image.tag` у values.yaml)
# - CrashLoopBackOff: помилка в коді (дивіться логи)
```

**Рішення:**

```bash
# Примусовий рестарт деплойменту (після виправлення `values.yaml` або коду)
kubectl rollout restart deployment -n ml-service ml-inference-service
```

### ArgoCD не синхронізується

```bash
# Перевірити Application
kubectl get application -n argocd ml-inference-service -o yaml

# Примусова синхронізація через kubectl (альтернатива ArgoCD CLI)
kubectl patch application ml-inference-service -n argocd \
  --type merge -p '{"operation":{"sync":{}}}'
```

### Prometheus не збирає метрики (scrape)

```bash
# Перевірити ServiceMonitor
kubectl get servicemonitor -n ml-service ml-inference-service

# Перевірити, що labels співпадають
kubectl get svc -n ml-service ml-inference-service --show-labels
```

Переконайтесь, що ServiceMonitor (який створюється з Helm-чарту) існує і його labels (напр., `release: monitoring`) збігаються з `serviceMonitorSelector` вашого Prometheus (встановленого на Кроці 6).

### Grafana не показує логи

```bash
# Перевірити Loki
kubectl get pods -n monitoring | grep loki

# Перевірити Promtail
kubectl logs -n monitoring -l app.kubernetes.io/name=promtail -f

# Перевірити Data Source в Grafana
# URL: http://loki.monitoring.svc.cluster.local:3100
```

## Структура проєкту

```
aiops-quality-project/
├── app/
│   ├── main.py            # FastAPI додаток
│   ├── requirements.txt   # Python залежності
│   └── Dockerfile         # Docker образ
├── model/
│   └── train.py           # Скрипт тренування
├── models/
│   ├── model.pkl          # Збережена модель
│   └── metadata.pkl       # Метадані моделі (для Z-score)
├── helm/
│   ├── Chart.yaml         # Helm chart metadata
│   ├── values.yaml        # Конфігурація (див. приклад нижче)
│   └── templates/
│       ├── deployment.yaml  # K8s Deployment
│       ├── service.yaml     # K8s Service
│       ├── servicemonitor.yaml # Для Prometheus
│       └── _helpers.tpl     # Helm helpers
├── argocd/
│   └── application.yaml   # ArgoCD Application
├── grafana/
│   └── dashboard.json     # Grafana Dashboard
├── tests/
│   └── test_drift.py      # Тестовий скрипт для дрейфу
├── .github/
│   └── workflows/
│       └── ci-cd.yaml     # GitHub Actions
├── .gitignore
└── README.md
```