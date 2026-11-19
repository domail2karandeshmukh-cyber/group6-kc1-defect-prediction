
from pathlib import Path
import json
import math
import os
import joblib

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ConfigDict, field_validator

DEPLOY_DIR = Path(os.environ.get("ARTIFACT_DIR", "/app/artifacts"))
CONFIG_PATH = DEPLOY_DIR / "deployment_config.json"

with open(CONFIG_PATH, "r") as f:
    CONFIG = json.load(f)

# The model exported by the training service is saved with the same sklearn pipeline.
# Loading the local artifact avoids Colab-only paths and run-ID URI issues.
MODEL = joblib.load(DEPLOY_DIR / "kc1_defect_pipeline.joblib")
FEATURE_NAMES = CONFIG["feature_names"]
THRESHOLD = float(CONFIG["deployment_threshold"])

app = FastAPI(
    title="Group 6 KC1 Software Defect Prediction API",
    version="1.0.1",
    description="REST API serving the MLflow-tracked Group 6 KC1 defect prediction model."
)

def json_safe(value):
    """Infinity and NaN can't be written in JSON, so an error message shows them as text."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def refuse_invalid_body(request: Request, exc: RequestValidationError):
    # A number such as 1e999 is read as infinity. It is refused like any other invalid value, with a 422:
    # without this handler, the error message itself fails because it can't hold infinity, and the API answers 500.
    return JSONResponse(status_code=422, content={"detail": json_safe(jsonable_encoder(exc.errors()))})


class KC1Input(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    loc: float = Field(allow_inf_nan=False)
    v_g: float = Field(allow_inf_nan=False, alias="v(g)")
    ev_g: float = Field(allow_inf_nan=False, alias="ev(g)")
    iv_g: float = Field(allow_inf_nan=False, alias="iv(g)")
    n: float
    v: float
    l: float
    d: float
    i: float
    e: float
    b: float
    t: float
    lOCode: float
    lOComment: float
    lOBlank: float
    locCodeAndComment: float
    uniq_Op: float
    uniq_Opnd: float
    total_Op: float
    total_Opnd: float
    branchCount: float

    @field_validator("*", mode="after")
    @classmethod
    def require_finite(cls, value):
        import math
        if not math.isfinite(value):
            raise ValueError("Every KC1 metric must be finite")
        return value

@app.get("/")
def root():
    return {
        "message": "Group 6 KC1 Software Defect Prediction API",
        "model": CONFIG["selected_model_name"]
    }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "model": CONFIG["selected_model_name"],
        "threshold": THRESHOLD,
        "feature_count": len(FEATURE_NAMES)
    }

@app.post("/predict")
def predict(payload: KC1Input):
    # Use aliases so DataFrame columns exactly match the KC1 training columns.
    values = payload.model_dump(by_alias=True)
    row = pd.DataFrame([[values[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)

    if hasattr(MODEL, "predict_proba"):
        defect_probability = float(MODEL.predict_proba(row)[0, 1])
        prediction = int(defect_probability >= THRESHOLD)
    else:
        prediction = int(MODEL.predict(row)[0])
        defect_probability = None

    return {
        "prediction": prediction,
        "label": "Defective" if prediction == 1 else "Non-defective",
        "defect_probability": defect_probability,
        "threshold": THRESHOLD,
        "model": CONFIG["selected_model_name"]
    }
