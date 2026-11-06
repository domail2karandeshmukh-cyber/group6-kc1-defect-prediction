# Group 6 KC1 — Manual Docker build + presentation guide

This is a reproducible Docker **source project**. It is not an already-built Docker image. A Docker image is created when `docker compose build` succeeds on your computer.

## 1. Verify Docker Desktop on your Mac

Open Docker Desktop and wait until the engine is running. Then open **Terminal**.

```bash
docker --version
docker compose version
docker info
```

If `docker info` fails, Docker Desktop is not ready. If Docker cannot be found, open/reinstall Docker Desktop.

## 2. Extract the provided project

Double-click the downloaded ZIP in Finder. Open Terminal and drag the extracted `Group6_KC1_Docker_Project_FINAL` folder onto a command after `cd `, then press Enter.

Example if it is in Downloads:

```bash
cd ~/Downloads/Group6_KC1_Docker_Project_FINAL
ls
```

You should see `Dockerfile`, `compose.yaml`, `requirements.txt`, `train.py`, `api_app.py`, `streamlit_app.py`.

## 3. Two ways to build

**Method A — One command (fast demo):**

```bash
bash scripts/start_mac.sh
```

The script runs exactly the Method B commands and checks the API.

**Method B — Manual steps (best for learning and demonstrating):**

```bash
docker compose build
```

This builds image `group6-kc1:latest` using `Dockerfile`: Python 3.11, required packages, and application files.

```bash
docker image ls group6-kc1
```

This lists the **built image**; it does not yet mean the application is running.

```bash
docker compose --profile train run --rm trainer
```

This downloads the KC1 dataset if it is not already present, deduplicates it, fits the four classifiers, logs runs to MLflow, evaluates thresholds and GMM models, then exports the selected model to `artifacts/`.

```bash
docker compose up -d
```

This starts the API, UI, and MLflow containers. The `-d` flag runs them in the background.

```bash
docker compose ps
```

This lists running containers and their status.

```bash
bash scripts/verify_mac.sh
```

This checks the image, containers, endpoint responses and a sample prediction.

## 4. Open the three interfaces

| Service | Local URL | Purpose |
|---|---|---|
| Streamlit | http://localhost:8501 | Enter the 21 code metrics and click Predict defect |
| FastAPI / Swagger | http://localhost:8000/docs | Demonstrate GET /health and POST /predict |
| MLflow | http://localhost:5000 | Show tracked classifier, threshold and GMM runs |

## 5. What to show the teacher

1. `Dockerfile` — explain base Python image, installing packages, copying source, default command.
2. `compose.yaml` — explain four services: trainer (on demand), API, Streamlit UI, MLflow viewer.
3. `docker image ls group6-kc1` — proves image exists **after it has been built**.
4. `docker compose ps` — proves services are running.
5. `http://localhost:8000/health` and `http://localhost:8000/docs` — API reachable.
6. `http://localhost:8501` — select *Load KC1 sample* and *Predict defect*; read the result.
7. `http://localhost:5000` — show experiments and model metrics.

Take screenshots of **your actual run**, not simulated screenshots.

## 6. Troubleshooting

- **`docker: command not found`** → Ensure Docker Desktop installed and CLI accessible.
- **Docker daemon unavailable** → Start Docker Desktop and wait for engine startup.
- **KC1 dataset fetch fails** → Download `kc1.csv` using your browser from the source URL in `train.py`, put it at `artifacts/kc1.csv`, rerun trainer.
- **Trainer prints unknown labels/feature names** → Check dataset is truly NASA/PROMISE KC1 CSV, not another project.
- **Port 8000/8501/5000 is busy** → Shut down other locally running services, or change the **left** port in compose.yaml. Keep container port unchanged.
- **API cannot load model** → Run the trainer successfully first; confirm `artifacts/kc1_defect_pipeline.joblib` exists.
- **UI cannot reach API** → `docker compose logs api` and `docker compose logs ui`.
- **Rerun training** → `docker compose --profile train run --rm trainer` then `docker compose restart api ui`.
- **Stop containers** → `docker compose down`.

## 7. Image export / Docker Hub

If the teacher explicitly asks for a transferable image rather than files to build it:

```bash
docker save -o group6-kc1-image.tar group6-kc1:latest
```

This produces a large image archive on **your Mac** after the build. The source ZIP here is much smaller and is *not* the same thing as an exported image. A `docker push` to Docker Hub is a separate step requiring an account and explicit publishing instructions.

## 8. Recreate the Docker packaging manually

If you want to reproduce rather than simply run the supplied files:

1. Start from your KC1 project Python sources and trained artifacts.
2. Create `requirements.txt` listing the imports needed at runtime.
3. Create `Dockerfile`: `FROM python:3.11-slim`; set `WORKDIR /app`; `COPY` requirements; `RUN pip install`; `COPY` app files; `CMD` launch Uvicorn.
4. Create `compose.yaml` with one image reused for `trainer`, `api`, `ui`, `mlflow`; mount a persistent `./artifacts` folder; expose ports 8000, 8501 and 5000 to localhost.
5. Build with `docker compose build`, train, start services, and test.
6. Save screenshots and summarize the architecture in your own words.

**Why the trainer runs separately:** You generally want to train once, then reuse the saved model for as many API predictions as you want; the API should not retrain every time it starts.

**Important experimental limitation:** Model and threshold selection in this classroom project use the held-out test set, which can bias reported performance. Production workflows should choose models and thresholds with validation data and reserve the test set for final evaluation.

**AI use:** It is fine to study this code and use assistance where allowed. Follow your teacher's rules on attribution. Do not claim to have manually built and verified a Docker image until you actually run the build and tests.
