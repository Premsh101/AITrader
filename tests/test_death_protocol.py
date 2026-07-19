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
    assert float(trade.pnl) == (94.0 - 100.0) * 5
