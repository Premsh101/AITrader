"""Tests for API-key protection and the LIVE-mode confirmation guard."""

from app.security import API_KEY_ENV


def test_health_is_open(client):
    res = client.get("/health")
    assert res.status_code == 200


def test_mutating_endpoint_503_when_key_unconfigured(client, monkeypatch):
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    res = client.post("/toggle-mode", json={"mode": "PAPER"})
    assert res.status_code == 503


def test_mutating_endpoint_401_without_key(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    res = client.post("/toggle-mode", json={"mode": "PAPER"})
    assert res.status_code == 401


def test_mutating_endpoint_401_with_wrong_key(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    res = client.post(
        "/toggle-mode",
        json={"mode": "PAPER"},
        headers={"X-API-Key": "wrong"},
    )
    assert res.status_code == 401


def test_toggle_paper_with_key_succeeds(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    res = client.post(
        "/toggle-mode",
        json={"mode": "PAPER"},
        headers={"X-API-Key": "secret-key"},
    )
    assert res.status_code == 200
    assert res.json() == {"mode": "PAPER", "is_live_mode": False}


def test_toggle_live_without_confirm_rejected(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    res = client.post(
        "/toggle-mode",
        json={"mode": "LIVE"},
        headers={"X-API-Key": "secret-key"},
    )
    assert res.status_code == 400
    assert "confirm" in res.json()["detail"].lower()


def test_toggle_live_with_confirm_succeeds(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    res = client.post(
        "/toggle-mode",
        json={"mode": "LIVE", "confirm": True},
        headers={"X-API-Key": "secret-key"},
    )
    assert res.status_code == 200
    assert res.json()["is_live_mode"] is True


def test_patch_config_requires_key(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    res = client.patch("/config", json={"is_live_mode": False})
    assert res.status_code == 401


def test_order_requires_key(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    res = client.post(
        "/order", json={"symbol": "TCS", "quantity": 1, "side": "BUY"}
    )
    assert res.status_code == 401


def test_paper_order_end_to_end_with_reference_price(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    res = client.post(
        "/order",
        json={
            "symbol": "TCS.NS",
            "quantity": 2,
            "side": "BUY",
            "reference_price": 4100.25,
        },
        headers={"X-API-Key": "secret-key"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["symbol"] == "TCS"
    assert float(body["buy_price"]) == 4100.25


def test_paper_order_without_price_rejected(client, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "secret-key")
    res = client.post(
        "/order",
        json={"symbol": "NOTCACHED", "quantity": 1, "side": "BUY"},
        headers={"X-API-Key": "secret-key"},
    )
    assert res.status_code == 400
