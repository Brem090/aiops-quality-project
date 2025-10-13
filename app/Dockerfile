# app/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 1) Залежності (файл у app/)
COPY app/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 2) Код (файл у app/)
COPY app/main.py /app/main.py

# 3) Модель (файл у models/)
RUN mkdir -p /models
COPY models/model.pkl /models/model.pkl

# 4) (Опційно) Явно вкажемо шлях моделі для коду
ENV MODEL_PATH=/models/model.pkl

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
