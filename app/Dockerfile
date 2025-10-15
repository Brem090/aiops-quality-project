FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY main.py /app/main.py

# Модель має бути всередині контексту, тобто у ./models
RUN mkdir -p /app/models
COPY models/model.pkl /app/models/model.pkl

ENV MODEL_PATH=/app/models/model.pkl

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

