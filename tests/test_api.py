"""API smoke tests: require a trained artifact directory supplied by KC1_TEST_ARTIFACT_DIR.
These are integration tests intended to run after `train.py`.
"""
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
