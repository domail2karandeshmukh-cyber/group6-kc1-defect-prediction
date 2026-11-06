#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "[1/5] Checking Docker Desktop..."
if ! command -v docker >/dev/null 2>&1; then
  echo "Docker CLI is missing. Install/open Docker Desktop first." >&2
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker Desktop is installed but the engine is not running. Open Docker Desktop and try again." >&2
  exit 1
fi
docker compose version
echo "[2/5] Building Group 6 Docker image..."
docker compose build
echo "[3/5] Training/exporting KC1 models and MLflow experiments..."
docker compose --profile train run --rm trainer
echo "[4/5] Starting FastAPI, Streamlit and MLflow..."
docker compose up -d
echo "[5/5] Checking API health..."
ready=0
for attempt in $(seq 1 25); do
  if curl --fail --silent --max-time 2 http://localhost:8000/health >/dev/null; then
    ready=1
    break
  fi
  sleep 2
done
if [ "$ready" -eq 0 ]; then
  echo "API did not become healthy. Inspect: docker compose logs api" >&2
  exit 1
fi
echo "SUCCESS: Docker API is responding"
docker compose ps
printf '\nOpen these URLs on your Mac:\n'
printf 'Streamlit:  http://localhost:8501\n'
printf 'Swagger API: http://localhost:8000/docs\n'
printf 'MLflow:     http://localhost:5000\n'
