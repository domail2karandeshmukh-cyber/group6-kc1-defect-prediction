# Docker Demonstration: Quick Questions and Answers

**What is a Docker image?** A read-only package containing our application code,
Python runtime and installed dependencies.

**What is a Docker container?** A running instance of that image.

**What does the Dockerfile do?** Specifies the base Python environment,
installs dependencies and copies the application's code.

**Why Docker Compose?** It starts multiple cooperating services: FastAPI,
Streamlit and MLflow, with a separate trainer command.

**What is port 8000?** Our FastAPI server. `/health` verifies that the model
loaded, and `/predict` accepts 21 numerical KC1 metrics.

**What is port 8501?** The Streamlit web interface.

**What is port 5000?** The MLflow experiment viewer.

**How does the UI reach the API?** Inside Compose it uses `http://api:8000`;
`api` is the service name, resolved by Docker networking.

**How does the model get inside the container?** Training saves a fitted model
and config under `artifacts/`, mounted into the API and UI containers.

**What is an image vs a Dockerfile?** A Dockerfile is the recipe; the image is
the built package; the container is the running process.

**What is an MLflow run?** A recorded training experiment with parameters,
metrics and model artifacts.

**Why use `127.0.0.1` in ports?** To keep the demo available on the local Mac
rather than exposing it to every device on the network.

**How do I prove it works?** Show `docker image ls`, `docker compose ps`,
`curl http://localhost:8000/health`, the Swagger docs, and a Streamlit prediction.

**How do I share an image?** `docker save` exports it to a `.tar` archive.
Docker Hub is another option if publishing is required and permitted.

**What is in the GitHub repository?** Source code, Docker files, scripts,
tests and documentation. The trained model, MLflow database, dataset copy and
Docker image are generated locally and excluded by `.gitignore`.

**Why is the Docker image not on GitHub?** It is a 309 MB binary archive, and
Git is meant for source code. The image can be rebuilt with
`docker compose build`, loaded from the exported `.tar` with `docker load`, or
pushed to a registry such as Docker Hub.

**Why Balanced Logistic Regression?** It had the highest recall on the test
set (it found 36 of the 63 defective modules) with F1 = 0.500 at threshold
0.50. Class balancing gives the minority defective class more weight.

**What does the GMM add?** An unsupervised view: it groups modules by their
metrics without using the labels. BIC chose 4 groups, and their defect rates
differ (about 10 % to 44 %).

**Important boundary:** The team built and ran this setup with Docker Desktop
on macOS on 6 November 2026 (see *Results* in `README.md`). Present only what
you have actually run; on another computer, repeat the build, training and
health checks before calling it a verified running Docker image.
