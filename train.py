"""Retrain and log the Group 6 KC1 models, then export Docker-ready artifacts.

Uses the same deduplication, split, four classifiers and model-selection rule
as the original Colab experiment. Run it once before starting the API.
"""
import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 606
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", "/app/artifacts")).resolve()
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
FEATURE_NAMES = ['loc', 'v(g)', 'ev(g)', 'iv(g)', 'n', 'v', 'l', 'd', 'i', 'e', 'b', 't', 'lOCode', 'lOComment', 'lOBlank', 'locCodeAndComment', 'uniq_Op', 'uniq_Opnd', 'total_Op', 'total_Opnd', 'branchCount']
EXPERIMENT_NAME = "Group6_KC1_Defect_Prediction"


def load_data():
    provided_path = os.environ.get("KC1_DATA_PATH")
    local_path = Path(provided_path) if provided_path else ARTIFACT_DIR / "kc1.csv"
    url = "https://raw.githubusercontent.com/ApoorvaKrisna/NASA-promise-dataset-repository/main/kc1.csv"
    if local_path.is_file():
        df = pd.read_csv(local_path)
        print(f"Loading local dataset: {local_path}")
    else:
        try:
            df = pd.read_csv(url)
            df.to_csv(ARTIFACT_DIR / "kc1.csv", index=False)
            print("Downloaded KC1 and saved a local copy")
        except Exception as exc:
            print(f"GitHub dataset unavailable ({exc}); trying OpenML dataset 1067")
            try:
                from sklearn.datasets import fetch_openml
                df = fetch_openml(data_id=1067, as_frame=True, parser="auto").frame.copy()
                df.to_csv(ARTIFACT_DIR / "kc1.csv", index=False)
                print("Downloaded KC1 from OpenML fallback")
            except Exception as fallback_exc:
                raise RuntimeError(
                    "KC1 could not be downloaded from GitHub or OpenML. "
                    "Save kc1.csv into artifacts/ and retry."
                ) from fallback_exc

    if "defects" not in df.columns:
        df = df.rename(columns={df.columns[-1]: "defects"})
    label_map = {"true": 1, "false": 0, "yes": 1, "no": 0, "1": 1, "0": 0,
                 "def": 1, "nodef": 0, "positive": 1, "negative": 0}
    labels = df["defects"].astype(str).str.strip().str.lower().map(label_map)
    if labels.isna().any():
        raise ValueError("Dataset has an unrecognized defect label")
    df["defects"] = labels.astype(int)
    df = df.drop_duplicates().reset_index(drop=True)
    if sorted(df.columns.drop("defects")) != sorted(FEATURE_NAMES):
        raise ValueError("KC1 dataset columns do not match the 21 expected predictors")
    x = df.loc[:, FEATURE_NAMES].apply(pd.to_numeric, errors="raise")
    if x.isna().any().any():
        raise ValueError("KC1 predictors contain missing values")
    return x, df["defects"]


def metric_row(name, truth, probability, threshold=0.5):
    pred = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
    return {
        "Model": name,
        "Threshold": threshold,
        "Precision": precision_score(truth, pred, zero_division=0),
        "Recall": recall_score(truth, pred, zero_division=0),
        "F1": f1_score(truth, pred, zero_division=0),
        "AP": average_precision_score(truth, probability),
        "FP": int(fp), "FN": int(fn), "TP": int(tp), "TN": int(tn),
    }


def main():
    x, y = load_data()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    print(f"KC1 analysis data: {len(x)} rows, {len(FEATURE_NAMES)} features")
    print(f"Train / test: {len(x_train)} / {len(x_test)}")
    models = {
        "Logistic default": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE)),
        ]),
        "Logistic balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "Random Forest default": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Random Forest balanced": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    db = ARTIFACT_DIR / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db}")
    exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        exp_id = mlflow.create_experiment(
            EXPERIMENT_NAME,
            artifact_location=(ARTIFACT_DIR / "mlruns").as_uri(),
        )
    else:
        exp_id = exp.experiment_id
    mlflow.set_experiment(EXPERIMENT_NAME)

    fitted, probabilities, run_ids, rows = {}, {}, {}, []
    for name, template in models.items():
        clf = clone(template)
        with mlflow.start_run(run_name=name) as run:
            clf.fit(x_train, y_train)
            prob = clf.predict_proba(x_test)[:, 1]
            metrics = metric_row(name, y_test, prob)
            algo = "LogisticRegression" if isinstance(clf, Pipeline) else "RandomForestClassifier"
            mlflow.log_params({
                "model_name": name, "algorithm": algo,
                "random_state": RANDOM_STATE, "threshold": 0.5,
            })
            mlflow.log_metrics({
                "precision": metrics["Precision"], "recall": metrics["Recall"],
                "f1": metrics["F1"], "average_precision": metrics["AP"],
                "false_positives": metrics["FP"], "false_negatives": metrics["FN"],
            })
            mlflow.sklearn.log_model(
                sk_model=clf,
                name="model",
                signature=infer_signature(x_train, clf.predict(x_train)),
                input_example=x_train.head(3),
                serialization_format="cloudpickle",
            )
            fitted[name] = clf
            probabilities[name] = prob
            run_ids[name] = run.info.run_id
            rows.append(metrics)

    comparison = pd.DataFrame(rows).sort_values(
        by=["Recall", "AP"], ascending=[False, False]
    ).reset_index(drop=True)
    # Experiment 2: log candidate thresholds for balanced Logistic Regression.
    # These are exploratory classroom comparisons, NOT independent validation.
    threshold_rows = []
    for candidate_threshold in (0.30, 0.50, 0.70):
        row = metric_row(
            "Logistic balanced", y_test,
            probabilities["Logistic balanced"], candidate_threshold
        )
        threshold_rows.append(row)
        with mlflow.start_run(run_name=f"Threshold {candidate_threshold:.2f} (balanced Logistic)"):
            mlflow.log_params({"experiment": "threshold", "model": "Logistic balanced", "threshold": candidate_threshold})
            mlflow.log_metrics({
                "precision": row["Precision"], "recall": row["Recall"],
                "f1": row["F1"], "false_positives": row["FP"],
                "false_negatives": row["FN"]
            })
    pd.DataFrame(threshold_rows).to_csv(ARTIFACT_DIR / "balanced_logistic_thresholds.csv", index=False)

    # Experiment 3: unsupervised GMM; no defect labels used to fit the model.
    gmm_scaler = StandardScaler()
    train_scaled = gmm_scaler.fit_transform(x_train)
    test_scaled = gmm_scaler.transform(x_test)
    bic_rows = []
    gmm_models = {}
    for k in (2, 3, 4):
        gmm = GaussianMixture(
            n_components=k, covariance_type="diag", n_init=10,
            random_state=RANDOM_STATE
        )
        gmm.fit(train_scaled)
        bic = float(gmm.bic(train_scaled))
        gmm_models[k] = gmm
        bic_rows.append({"components": k, "bic": bic})
        with mlflow.start_run(run_name=f"GMM {k} components"):
            mlflow.log_params({"experiment": "unsupervised_gmm", "components": k, "covariance_type": "diag"})
            mlflow.log_metric("bic", bic)
    bic_table = pd.DataFrame(bic_rows)
    bic_table.to_csv(ARTIFACT_DIR / "gmm_bic.csv", index=False)
    best_k = int(bic_table.loc[bic_table["bic"].idxmin(), "components"])
    assigned = gmm_models[best_k].predict(test_scaled)
    posthoc = pd.DataFrame({"Component": assigned, "Defect": y_test.to_numpy()})
    component_summary = (
        posthoc.groupby("Component")
        .agg(Modules=("Defect", "size"), Defects=("Defect", "sum"),
             Defect_Prevalence=("Defect", "mean"))
        .reset_index()
    )
    component_summary.to_csv(ARTIFACT_DIR / "gmm_component_summary.csv", index=False)
    joblib.dump({"scaler": gmm_scaler, "model": gmm_models[best_k], "components": best_k},
                ARTIFACT_DIR / "gmm_analysis.joblib")

    selected = str(comparison.loc[0, "Model"])
    # Same decision-threshold choice as the original Colab notebook.
    threshold_grid = pd.DataFrame([
        metric_row(selected, y_test, probabilities[selected], t)
        for t in (0.30, 0.50, 0.70)
    ]).sort_values(by=["F1", "Recall"], ascending=[False, False])
    threshold = float(threshold_grid.iloc[0]["Threshold"])
    config = {
        "experiment_name": EXPERIMENT_NAME,
        "tracking_uri": f"sqlite:///{db}",
        "selected_model_name": selected,
        "selected_run_id": run_ids[selected],
        "selected_model_uri": f"runs:/{run_ids[selected]}/model",
        "deployment_threshold": threshold,
        "feature_names": FEATURE_NAMES,
        "target": "defects",
        "class_0": "Non-defective",
        "class_1": "Defective",
    }
    joblib.dump(fitted[selected], ARTIFACT_DIR / "kc1_defect_pipeline.joblib")
    (ARTIFACT_DIR / "deployment_config.json").write_text(json.dumps(config, indent=2))
    sample = {k: float(v) for k, v in x_test.iloc[0].items()}
    (ARTIFACT_DIR / "streamlit_defaults.json").write_text(json.dumps(sample, indent=2))
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    threshold_grid.to_csv(ARTIFACT_DIR / "threshold_comparison.csv", index=False)
    print(comparison[["Model", "Precision", "Recall", "F1", "AP"]].round(3).to_string(index=False))
    print(f"Selected: {selected}; threshold: {threshold:.2f}")
    print(f"Unsupervised GMM: {best_k} components selected by minimum BIC")
    print("Created deployment_config.json, kc1_defect_pipeline.joblib, and streamlit_defaults.json")


if __name__ == "__main__":
    main()
