# AIOps Quality Project

Фінальний проєкт курсу з розробки MLOps системи для inference моделі машинного навчання з автоматичним виявленням дрейфу даних.

## Про проєкт

Цей проєкт демонструє повний цикл розробки production-ready ML сервісу: від тренування моделі до автоматизованого деплою через GitOps. Система автоматично виявляє відхилення у вхідних даних (drift detection) і може перезапускати тренування моделі через CI/CD pipeline.

## Зміст

- [Як влаштована система](#як-влаштована-система)
- [Що потрібно для запуску](#що-потрібно-для-запуску)
- [Встановлення](#встановлення)
- [Запуск](#запуск)
- [Як тестувати](#як-тестувати)
- [Моніторинг](#моніторинг)
- [Оновлення моделі](#оновлення-моделі)
- [Скріншоти](#скріншоти)
- [Проблеми та рішення](#проблеми-та-рішення)

---

## Як влаштована система

Проєкт складається з декількох компонентів які працюють разом:

```
GitHub (вихідний код)
    ↓
GitHub Actions (автоматичне тестування та білд)
    ↓
Docker образ (контейнер з моделлю)
    ↓
ArgoCD (автоматичний деплой у Kubernetes)
    ↓
FastAPI сервіс (приймає запити та робить передбачення)
    ↓
Prometheus + Grafana (збирають метрики та показують графіки)
Loki + Promtail (збирають та зберігають логи)
```

---

### Основні компоненти

**FastAPI сервіс** - приймає JSON запити з 20 числовими ознаками, робить передбачення за допомогою натренованої моделі (RandomForest, GradientBoosting, LogisticRegression або NeuralNetwork) і повертає результат разом з ймовірністю.

**Drift Detector** - аналізує кожен запит за допомогою Z-score статистики. Якщо дані сильно відрізняються від "нормальних" (більше ніж на 3 стандартні відхилення), система логує це як drift і може автоматично запустити перетренування моделі.

**Prometheus** - збирає метрики: скільки запитів прийшло, як швидко відповідає API, скільки разів виявлено drift.

**Grafana** - показує всі ці метрики у вигляді красивих графіків у реальному часі.

**Loki** - зберігає всі логи з сервісу, щоб можна було подивитись що відбувалось у минулому.

**ArgoCD** - слідкує за GitHub репозиторієм і автоматично оновлює сервіс у Kubernetes коли ви робите commit.

---

## Що потрібно для запуску

### Програмне забезпечення

- **Docker Desktop** з увімкненим Kubernetes
- **Python 3.11+**
- **Helm 3**
- **kubectl**
- **Git**
- **GitHub Account**

### Перевірка встановлення

```bash
# Docker Desktop Kubernetes
kubectl version --short
# Очікуваний результат: Client/Server Version

# Helm
helm version --short
# Очікуваний результат: v3.x.x

# Python
python --version
# Очікуваний результат: Python 3.11.x
```

Якщо всі команди працюють - можна починати.

---

## Встановлення

### Крок 1: Клонування репозиторію

```bash
# Клонуємо репозиторій
git clone https://github.com/YOUR_USERNAME/aiops-quality-project.git
cd aiops-quality-project

# Переключаємось на гілку final-project
git checkout final-project
```

### Крок 2: Встановлення Python залежностей

```bash
# Створюємо віртуальне середовище
python -m venv venv

# Активуємо (Windows)
venv\Scripts\activate

# Активуємо (Linux/Mac)
source venv/bin/activate

# Встановлюємо залежності
pip install -r app/requirements.txt
```

### Крок 3: Тренування початкової моделі

```bash
# Тренуємо модель
python model/train.py

# Перевіряємо що модель створилась
ls models/
# Повинен бути: model.pkl, metadata.pkl
```

### Крок 4: Білд Docker образу

```bash
# Переходимо в папку app
cd app

# Копіюємо модель у підпапку models всередині app
New-Item -ItemType Directory -Force -Path "models" | Out-Null

Copy-Item -Path "..\models\model.pkl" -Destination "models\model.pkl" -Force

# Будуємо Docker образ
docker build -t ml-inference-service:latest .

# Перевіряємо
docker images | Select-String "ml-inference-service"
```

### Крок 5: Встановлення ArgoCD

```bash
# Створюємо namespace
kubectl create namespace argocd

# Встановлюємо ArgoCD
kubectl apply -n argocd -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Чекаємо на готовність
kubectl wait --for=condition=Ready pods --all -n argocd --timeout=300s

# Отримуємо пароль admin
[System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String(
    (kubectl -n argocd get secret argocd-initial-admin-secret -o jsonpath="{.data.password}")
))

# Port-forward для UI (в окремому терміналі)
kubectl port-forward svc/argocd-server -n argocd 8080:443
```

**Відкрийте**: https://localhost:8080
- **Login**: admin
- **Password**: (з команди вище)

### Крок 6: Встановлення Prometheus та Grafana

```bash
# Додаємо Helm репозиторії
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Встановлюємо kube-prometheus-stack
helm upgrade --install monitoring prometheus-community/kube-prometheus-stack `
  --namespace monitoring `
  --create-namespace `
  --wait `
  --set grafana.enabled=true `
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false `
  --set nodeExporter.enabled=false `
  --set grafana.additionalDataSources[0].name=Loki `
  --set grafana.additionalDataSources[0].type=loki `
  --set grafana.additionalDataSources[0].access=proxy `
  --set grafana.additionalDataSources[0].url=http://loki.monitoring.svc.cluster.local:3100

# Чекаємо на готовність
kubectl wait --for=condition=Ready pods --all -n monitoring --timeout=300s
```

### Крок 7: Встановлення Loki Stack

```bash
# Встановлюємо Loki Stack
helm upgrade --install loki grafana/loki-stack `
  --namespace monitoring `
  --wait `
  --set loki.enabled=true `
  --set promtail.enabled=true `
  --set fluent-bit.enabled=false `
  --set grafana.enabled=false `
  --set loki.persistence.enabled=true `
  --set loki.persistence.size=1Gi `
  --set loki.auth_enabled=false

# Оновлення до новішої версії (у разі необхідності, коли версії Loki-контейнеру і Loki Helm-чарту не збігаються)

kubectl describe pod -n monitoring loki-0 | findstr "Image" # має бути більше 2.9, інакше може бути помилка під'єднання Loki в Grafana

# якщо версія стара, потрібно використати цю команду
helm upgrade loki grafana/loki-stack `
  --namespace monitoring `
  --reuse-values `
  --set loki.image.tag=2.9.4 `
  --set promtail.image.tag=2.9.4 `
  --wait

# Перевіряємо чи усе добре
kubectl get pods -n monitoring
```

---

## Запуск

### Метод 1: Через Helm

```bash

# Створюємо namespace для сервісу
kubectl create namespace ml-service

# Встановлюємо через Helm
helm install ml-service ./helm --namespace ml-service

# Перевіряємо статус
kubectl get pods -n ml-service
kubectl get svc -n ml-service
```

### Метод 2: Через ArgoCD (GitOps - рекомендований)

```bash
# Оновіть argocd/application.yaml з вашим GitHub репозиторієм
# Замініть YOUR_USERNAME на ваш GitHub username

# Перейдіть у кореневу директорію проєкту
cd aiops-quality-project

# Застосовуємо Application
kubectl apply -f argocd/application.yaml -n argocd

# Перевіряємо статус
kubectl get application -n argocd ml-inference-service

# Перевіряємо статус у ArgoCD UI
# https://localhost:8080
```

ArgoCD автоматично:
- Синхронізує Helm chart з Git
- Деплоїть сервіс у namespace `ml-service`
- Автоматично оновлює при змінах у репозиторії

### Перевірка деплою

```bash
# Переглянути поди
kubectl get pods -n ml-service

# Очікуваний результат:
# NAME                                    READY   STATUS    RESTARTS   AGE
# ml-inference-service-xxxx...     1/1     Running   0          2m

# Переглянути логи
kubectl logs -n ml-service -l app.kubernetes.io/name=ml-inference-service -f
```

---

## Як тестувати

### 1. Port-forward до сервісу (окреме вікно терміналу)

```bash
kubectl port-forward -n ml-service svc/ml-inference-service 8000:8000
```

### 2. Перевірка health endpoint (у новому вікні терміналу)

```bash
curl http://localhost:8000/health
```

**Очікувана відповідь**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### 3. Тестовий запит на передбачення

#### Через curl:

```bash
$body = @{
    features = @(0.5, -0.3, 1.2, 0.8, -0.5, 0.2, 0.9, -0.1,
                 0.4, 0.7, -0.6, 0.3, 0.1, -0.4, 0.6, 0.2,
                 -0.8, 0.5, 0.9, -0.2)
} | ConvertTo-Json

Invoke-WebRequest -Uri "http://localhost:8000/predict" `
  -Method Post `
  -ContentType "application/json" `
  -Body $body | Select-Object -ExpandProperty Content

```

**Очікувана відповідь**:
```json
{
  "prediction": 1,
  "probability": 0.6534023807361028,
  "drift_detected": false,
  "timestamp": "2025-10-14T10:20:53.864439"
}
```

#### Через Python скрипт:

```bash
# Запускаємо тестовий скрипт
python tests/test_drift.py
```

Скрипт відправить:
- 30 нормальних запитів
- 20 аномальних запитів (для тригера drift)

### 4. Перевірка метрик

```bash
curl http://localhost:8000/metrics -UseBasicParsing | Select-Object -ExpandProperty Content
```

Шукайте метрики:
- `predictions_total` - загальна кількість передбачень
- `drift_detected_total` - кількість виявлених дрейфів
- `prediction_latency_seconds` - час відповіді

---

## Моніторинг

### Grafana Dashboard

#### Доступ до Grafana

```bash
# Отримуємо пароль admin
[System.Text.Encoding]::UTF8.GetString(
    [System.Convert]::FromBase64String(
        (kubectl get secret -n monitoring monitoring-grafana -o jsonpath="{.data.admin-password}")
    )
)

# Port-forward (окреме вікно)
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80
```

**Відкрийте**: http://localhost:3000
- **Login**: admin
- **Password**: (з команди вище)

#### Імпорт Dashboard

1. У Grafana UI: **Dashboards → Import**
2. Upload `grafana/dashboard.json`
3. Натисніть **Import**

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

**Відкрийте**: http://localhost:9090

Корисні PromQL запити (Graph):
```promql
# Кількість передбачень за останню хвилину
rate(predictions_total[1m])

# 95-й персентиль латентності
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))

# Частота drift detection
rate(drift_detected_total[5m])
```

### Loki Logs

У Grafana:
1. **Explore → Loki**
2. Запит: `{namespace="monitoring}`
3. Фільтр для drift: `{namespace="monitoring"} |= "Drift detected"`

---

## CI/CD Pipeline

### GitHub Actions Workflow

Workflow файл: `.github/workflows/ci-cd.yaml`

#### Автоматичні тригери:

1. **Push до main/final-project** → білд та деплой
2. **Коміт з `[retrain]`** → retrain моделі
3. **Manual trigger** → через GitHub UI

### Ручний запуск retrain

#### Метод 1: Через GitHub UI

1. Перейдіть: **Actions → CI/CD Pipeline**
2. Натисніть **Run workflow**
3. Виберіть гілку: `final-project`
4. Встановіть `retrain: true`
5. Натисніть **Run workflow**

#### Метод 2: Через коміт

```bash
git commit --allow-empty -m "[retrain] Test retrain"
git push origin final-project
```

### Перевірка pipeline

```bash
# Перегляньте Actions на GitHub
# https://github.com/YOUR_USERNAME/aiops-quality-project/actions
```

Pipeline виконає:
1. Тести (test job)
2. Retrain моделі (retrain-model job)
3. Білд Docker образу (build-and-push job)
4. Оновлення Helm values.yaml
5. ArgoCD автоматично підхопить зміни та задеплоїть

---

## Перевірка функціональності

### Чеклист перевірки

#### 1. API

```bash
kubectl port-forward -n ml-service svc/ml-service-ml-inference-service 8000:8000
curl http://localhost:8000/health
```

#### 2. Передбачення

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9,2.0]}'
```

#### 3. Drift detection

```bash
# Запускаємо тест
python tests/test_drift.py

# Перевіряємо логи
kubectl logs -n ml-service -l app.kubernetes.io/name=ml-inference-service | grep "Drift detected"
```

#### 4. Логи

```bash
# В Grafana Explore
{namespace="monitoring"}
```

#### 5. Метрики

```bash
# Prometheus targets
# http://localhost:9090/targets
# Шукайте: ml-inference-service
```

#### 6. Grafana Dashboard

```bash
# Відкрийте dashboard "ML Inference Service Dashboard"
# Перевірте що панелі показують дані
```

#### 7. ArgoCD sync

```bash
# В ArgoCD UI
# https://localhost:8080
# Application: ml-inference-service
# Status: Synced, Healthy
```

#### 8. GitHub Actions pipeline 

```bash
# Тригеруємо workflow
git commit --allow-empty -m "[retrain] Test retrain"
git push

# Перевіряємо на GitHub Actions
```

---

## 🛠️ Troubleshooting

### Pod не запускається

```bash
# Дивимось статус
kubectl describe pod -n ml-service <pod-name>

# Перевіряємо логи
kubectl logs -n ml-service <pod-name>

# Типові проблеми:
# - ImagePullBackOff: образ не знайдено
# - CrashLoopBackOff: помилка в коді
```

**Рішення**:
```bash
# Перебілдити образ
cd app
docker build -t ml-inference-service:latest .

# Перезапустити pod
kubectl rollout restart deployment -n ml-service
```

### ArgoCD не синхронізується

```bash
# Перевірити Application
kubectl get application -n argocd ml-inference-service -o yaml

# Примусова синхронізація
kubectl patch application ml-inference-service -n argocd \
  --type merge -p '{"operation":{"initiatedBy":{"username":"admin"},"sync":{"revision":"HEAD"}}}'
```

### Prometheus не scrape метрики

```bash
# Перевірити ServiceMonitor
kubectl get servicemonitor -n ml-service

# Перевірити що labels співпадають
kubectl get svc -n ml-service --show-labels
```

### Grafana не показує логи

```bash
# Перевірити Loki
kubectl get pods -n monitoring | grep loki

# Перевірити Promtail
kubectl logs -n monitoring -l app=promtail

# Додати Loki data source в Grafana
# URL: http://loki.monitoring.svc.cluster.local:3100
```

### Drift не виявляється

```bash
# Перевірити що використовується правильний тест
python tests/test_drift.py

# Подивитись логи детально
kubectl logs -n ml-service -l app.kubernetes.io/name=ml-inference-service -f | grep -A 5 -B 5 "z-score"
```

---

## 📁 Структура проєкту

```
aiops-quality-project/
├── app/
│   ├── main.py              # FastAPI додаток
│   ├── requirements.txt     # Python залежності
│   └── Dockerfile           # Docker образ
├── model/
│   └── train.py             # Скрипт тренування
├── models/
│   ├── model.pkl            # Збережена модель
│   └── metadata.pkl         # Метадані моделі
├── helm/
│   ├── Chart.yaml           # Helm chart metadata
│   ├── values.yaml          # Конфігурація
│   └── templates/
│       ├── deployment.yaml  # K8s Deployment
│       ├── service.yaml     # K8s Service
│       └── _helpers.tpl     # Helm helpers
├── argocd/
│   └── application.yaml     # ArgoCD Application
├── scripts/
│   └── quick-test.sh        # Швидкий тест ML Inference Servicer
├── grafana/
│   └── dashboard.json       # Grafana Dashboard
├── tests/
│   └── test_drift.py        # Тестовий скрипт
├── .github/
│   └── workflows/
│       └── ci-cd.yaml       # GitHub Actions
├── .gitignore
└── README.md
```

---

## 🔄 Оновлення моделі

### Процес оновлення:

1. **Локальне тренування**:
```bash
python model/train.py
```

2. **Білд нового образу**:
```bash
cd app
cp ../models/model.pkl .
docker build -t ml-inference-service:v2 .
```

3. **Оновлення Helm values**:
```yaml
# helm/values.yaml
image:
  tag: "v2"
```

4. **Коміт та push**:
```bash
git add helm/values.yaml
git commit -m "Update model to v2"
git push origin final-project
```