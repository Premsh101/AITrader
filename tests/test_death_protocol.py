"""Integration tests: halt flag blocks config paths and PATCH /config re-arms."""

from decimal import Decimal

from app.models import SystemConfig, Trade, TradeMode, TradeStatus
from app.security import API_KEY_ENV


def test_config_reports_halted(client, db_session):
    db_session.add(SystemConfig(is_live_mode=False, is_halted=True))
    db_session.commit()
    res = client.get("/config")
    assert res.status_code == 200
    assert res.json()["is_halted"] is True


def test_patch_clears_halt_and_resets_hwm(client, db_session, monkeypatch):
    monkeypatch.setenv(API_KEY_ENV, "k")
    db_session.add(
        SystemConfig(is_live_mode=False, is_halted=True, peak_equity=Decimal("60000"))
    )
    db_session.commit()

    res = client.patch(
        "/config", json={"is_halted": False}, headers={"X-API-Key": "k"}
    )
    assert res.status_code == 200
    assert res.json()["is_halted"] is False

    config = db_session.query(SystemConfig).first()
    db_session.refresh(config)
    assert config.is_halted is False
    assert config.peak_equity is None  # HWM re-armed
    assert config.is_live_mode is False  # untouched


def test_close_trade_records_exit_reason(db_session):
    from app.main import _close_trade

    trade = Trade(
        symbol="TCS",
        buy_price=Decimal("100"),
        quantity=5,
        status=TradeStatus.OPEN,
        mode=TradeMode.PAPER,
    )
    db_session.add(trade)
    db_session.commit()

    _close_trade(db_session, trade, exit_price=94.0, reason="stop-loss")
    assert trade.status == TradeStatus.CLOSED
    assert trade.exit_reason == "stop-loss"
    # gross −30; minus 25 bps charges on (500 + 470) turnover = 2.425
    assert float(trade.pnl) == (94.0 - 100.0) * 5 - 2.425


def test_close_trade_deducts_charges(db_session):
    from app.main import _close_trade

    trade = Trade(
        symbol="INFY", buy_price=Decimal("1000"), quantity=10,
        status=TradeStatus.OPEN, mode=TradeMode.PAPER,
    )
    db_session.add(trade)
    db_session.commit()

    _close_trade(db_session, trade, exit_price=1100.0, reason="guardian")
    # gross = 100 x 10 = 1000; charges = 25bps of (10000+11000) = 52.5
    assert float(trade.charges) == 52.5
    assert float(trade.pnl) == 1000.0 - 52.5


def test_ghost_recording_and_endpoint(client, db_session):
    import pandas as pd
    from app.main import _record_ghosts

    md = {"TCS": pd.DataFrame({"close": [4000.0, 4100.0]})}
    _record_ghosts(db_session, ["TCS"], "executive-skip", md)

    res = client.get("/ghost-trades")
    assert res.status_code == 200
    body = res.json()
    assert body[0]["symbol"] == "TCS"
    assert body[0]["reason"] == "executive-skip"
    assert body[0]["evaluated"] is False
