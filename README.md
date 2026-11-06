# Group 6 — KC1 Software Defect Prediction (Docker)

Repackages the existing NASA PROMISE KC1 model, MLflow experiment tracking,
FastAPI prediction endpoint, and Streamlit dashboard into Docker.

## Prerequisites

- Docker Desktop for Mac, with Docker Engine running
- Internet access **for the first KC1 dataset download** (or a local `artifacts/kc1.csv`)
- Disk space for Python dependencies, model and MLflow experiments

## Quick start (on your Mac)

```bash
bash scripts/start_mac.sh
```

Or use the individual commands below. For complete manual instructions, read `MANUAL_BUILD_GUIDE.md`.

## Commands (run from this folder in Terminal)

```bash
docker --version
docker compose version
docker compose build
docker compose --profile train run --rm trainer
docker compose up -d
# Verify:
docker compose ps
curl http://localhost:8000/health
```

The training command recreates the original four candidate supervised models,
logs threshold experiments and unsupervised GMM comparisons to MLflow,
and writes the selected sklearn pipeline to the mounted `artifacts/`
folder. **Train before starting the API.** The initial run needs KC1 data;
if downloading fails, put the dataset in `artifacts/kc1.csv`.

## URLs

- Streamlit dashboard: http://localhost:8501
- FastAPI Swagger: http://localhost:8000/docs
- API health check: http://localhost:8000/health
- MLflow experiment viewer: http://localhost:5000

The frontend calls the backend at `http://api:8000` **inside the Docker Compose
network**. The ports exposed to the Mac are bound to localhost for a classroom
or local demo, not to the public internet.

## Inspect the image/container

```bash
docker image ls group6-kc1
docker compose ps
docker compose logs -f api
```

To show the **Docker image** to your teacher, use `docker image ls`. To show
running **containers**, use `docker compose ps`. To demonstrate the complete
pipeline, open the Streamlit interface and click *Predict defect*.

## Export a Docker image for submission (only if requested)

```bash
docker save -o group6-kc1-image.tar group6-kc1:latest
```

The `.tar` export can be large. Do not submit it unless the teacher requests a
built image rather than the source ZIP.

## Stop

```bash
docker compose down
```

## Project layout

- `Dockerfile`: builds one reusable Python Docker image.
- `compose.yaml`: defines training, API, UI, and MLflow viewer services.
- `train.py`: trains/logs candidates, selects model, and saves artifacts.
- `api_app.py`: serves `/health` and `/predict` using the exported pipeline.
- `streamlit_app.py`: 21-input interactive prediction interface.
- `requirements.txt`: Python dependencies.
- `artifacts/`: persistent dataset, saved model, config, and MLflow runs.

## Implementation notes

- The Docker image is built from Python 3.11 with dependencies installed.
- MLflow tracking is persisted to SQLite and local run artifacts.
- API predictions load the same **selected fitted sklearn pipeline** exported
  by training. This avoids embedding Colab-specific paths in an image.
- Model choice is based on held-out Recall then Average Precision; threshold
  is based on F1 then Recall on candidate thresholds 0.30 / 0.50 / 0.70,
  following the notebook's **classroom experimental workflow**. In production,
  threshold selection should use a separate validation split, not the test set.
- If your instructor requires an already-trained model that exactly matches the
  original Colab run, export that run's artifacts and do not retrain, as different
  versions of the libraries can produce differences.

## Troubleshooting

- `docker: command not found`: install and launch Docker Desktop.
- `No such file ... kc1_defect_pipeline.joblib`: run the trainer service first.
- Dataset download error: place the KC1 CSV at `artifacts/kc1.csv`.
- UI says API unavailable: check `docker compose ps` and `docker compose logs api`.
- Port already used: stop the competing app or adjust the left port number in
  `compose.yaml`.
- Docker Desktop on Apple Silicon uses the appropriate arm64 image; no manual
  x86 emulation is required for these Python packages.

## Verified local demonstration (6 November 2026)

The Group 6 team ran the Docker workflow on a MacBook Air with Docker Desktop:

- `docker compose build` completed and created `group6-kc1:latest`.
- `docker compose --profile train run --rm trainer` completed training on 1,212 KC1 rows and 21 features.
- Balanced Logistic Regression was selected (test F1 = 0.500; threshold = 0.50); GMM selected 4 components by BIC.
- `docker compose up -d` launched the API, Streamlit and MLflow services.
- `docker compose ps` showed the API healthy and all three services running.
- The Streamlit sample predicted class 1 (defective), with 72.3% model-estimated probability.
- `docker save -o Group6_KC1_Docker_Image.tar group6-kc1:latest` exported a Docker image archive; its `.tar` contents were checked.

These results are from the team's local demonstration. Rebuilding on other systems may differ depending on data and dependency versions.

## GitHub and Docker image submission

This GitHub repository contains **source code and documentation**, not the exported Docker image.
The `artifacts/` folder is generated by the trainer and remains local. To reproduce:

```bash
docker compose build
docker compose --profile train run --rm trainer
docker compose up -d
```

For a teacher requesting the actual image, provide the locally exported `.tar` or publish to Docker Hub separately. The image alone does not contain the mounted model artifacts.

Do **not** upload the exported `.tar`, `.zip`, `.joblib`, dataset or local MLflow database directly to the Git repository.

