import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.api.middleware import rate_limit


@pytest.fixture
def client():
    rate_limit.rate_limiter.max_requests = 1000
    rate_limit.rate_limiter.window_seconds = 60
    rate_limit.rate_limiter._buckets = {}
    return TestClient(app)


def test_health_has_request_id(client: TestClient):
    res = client.get("/api/v1/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}
    assert "x-request-id" in res.headers


def test_rate_limit_returns_structured_error(client: TestClient):
    rate_limit.rate_limiter.max_requests = 1
    rate_limit.rate_limiter._buckets = {}

    first = client.get("/api/v1/health")
    assert first.status_code == 200

    second = client.get("/api/v1/health")
    assert second.status_code == 429
    payload = second.json()
    assert payload["error"]["code"] == "rate_limited"
    assert "request_id" in payload["error"]
    assert "x-request-id" in second.headers


def test_retrieve_invalid_request(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    async def _raise_value_error(_request):
        raise ValueError("bad request")

    monkeypatch.setattr(
        "app.api.v1.retrieve.public_svc.retrieve_documents",
        _raise_value_error,
    )

    res = client.post(
        "/api/v1/retrieve",
        json={"queries": ["test"], "collection_name": "docs"},
        headers={"X-API-Key": "dev-secret-key"},
    )
    assert res.status_code == 400
    payload = res.json()
    assert payload["error"]["code"] == "invalid_request"
    assert payload["error"]["message"] == "bad request"
