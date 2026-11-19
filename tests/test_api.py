"""API smoke tests: require a trained artifact directory supplied by KC1_TEST_ARTIFACT_DIR.
These are integration tests intended to run after `train.py`.
"""
import json
import os
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client():
    if not os.path.isfile(os.path.join(os.environ.get("ARTIFACT_DIR", "artifacts"), "deployment_config.json")):
        pytest.skip("Training artifacts are not present; run trainer first")
    from api_app import app
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["feature_count"] == 21


def test_invalid_payload(client):
    r = client.post("/predict", json={"loc": 123})
    assert r.status_code == 422


def test_non_finite_values_are_refused(client):
    """1e999 is read as infinity, and Python's JSON reader accepts NaN: both are refused with a 422, not a 500."""
    body = {name: 1.0 for name in client.get("/openapi.json").json()["components"]["schemas"]["KC1Input"]["properties"]}
    for value in ("1e999", "-1e999", "NaN"):
        raw = json.dumps(body).replace('"loc": 1.0', f'"loc": {value}', 1)
        r = client.post("/predict", content=raw, headers={"content-type": "application/json"})
        assert r.status_code == 422, (value, r.status_code)
        assert r.json()["detail"][0]["loc"] == ["body", "loc"]
