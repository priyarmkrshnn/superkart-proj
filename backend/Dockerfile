FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached between code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the API code and the serialized model
COPY app.py superkart_model.joblib ./

EXPOSE 7860

# 2 Gunicorn workers serving the Flask app object "superkart_api" defined in app.py
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:7860", "app:superkart_api"]
