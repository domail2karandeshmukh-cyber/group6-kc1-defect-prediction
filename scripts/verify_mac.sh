#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Docker version"; docker --version
echo "Image"; docker image ls group6-kc1
echo "Containers"; docker compose ps
echo "API /health"; curl -fsS http://localhost:8000/health; echo
echo "API /docs status"; curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8000/docs
echo "Streamlit status"; curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8501
echo "MLflow status"; curl -s -o /dev/null -w '%{http_code}\n' http://localhost:5000
echo "API sample prediction"
if [ ! -f artifacts/streamlit_defaults.json ]; then
  echo "Missing artifacts/streamlit_defaults.json — run trainer first" >&2
  exit 1
fi
curl --fail --show-error --silent \
  -X POST -H 'Content-Type: application/json' \
  --data-binary @artifacts/streamlit_defaults.json \
  http://localhost:8000/predict
echo
printf '\nVerification complete (inspect any non-200 status above).\n'
