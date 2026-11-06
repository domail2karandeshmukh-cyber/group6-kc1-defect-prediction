FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ARTIFACT_DIR=/app/artifacts
WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY api_app.py streamlit_app.py train.py ./
RUN mkdir -p /app/artifacts
EXPOSE 8000 8501 5000
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]
