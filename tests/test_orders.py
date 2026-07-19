"""Tests for ShoonyaEngine paper-order safety and position sizing."""

from unittest.mock import MagicMock

import pytest

from app.models import Trade, TradeMode, TradeStatus
from app.shoonya_service import ShoonyaEngine, position_size


@pytest.fixture()
def engine() -> ShoonyaEngine:
    eng = ShoonyaEngine()
    # Any broker interaction in paper mode is a bug: make it loud.
    eng._api = MagicMock(
        side_effect=AssertionError("broker API must not be touched in paper mode")
    )
    eng.login = MagicMock(
        side_effect=AssertionError("login must not be attempted in paper mode")
    )
    return eng


def test_paper_order_requires_reference_price(engine, db_session):
    with pytest.raises(ValueError, match="reference_price"):
        engine.place_logic_order(
            symbol_base="RELIANCE",
            quantity=1,
            side="BUY",
            db=db_session,
            reference_price=None,
        )


def test_paper_order_rejects_zero_price(engine, db_session):
    with pytest.raises(ValueError):
        engine.place_logic_order(
            symbol_base="RELIANCE",
            quantity=1,
            side="BUY",
            db=db_session,
            reference_price=0.0,
        )


def test_paper_order_records_reference_price_and_never_calls_broker(engine, db_session):
    trade = engine.place_logic_order(
        symbol_base="RELIANCE",
        quantity=3,
        side="BUY",
        db=db_session,
        reference_price=2500.5,
    )
    assert trade.mode == TradeMode.PAPER
    assert trade.status == TradeStatus.OPEN
    assert float(trade.buy_price) == 2500.5
    assert trade.quantity == 3
    assert trade.symbol == "RELIANCE"  # base symbol stored, no suffixes
    # The MagicMock side effects above would have raised on any broker call;
    # additionally assert nothing was even attempted.
    assert not engine._api.mock_calls

    stored = db_session.query(Trade).one()
    assert float(stored.buy_price) == 2500.5


def test_paper_order_normalises_yahoo_symbol(engine, db_session):
    trade = engine.place_logic_order(
        symbol_base="TCS.NS",
        quantity=1,
        side="BUY",
        db=db_session,
        reference_price=100.0,
    )
    assert trade.symbol == "TCS"


def test_invalid_side_rejected(engine, db_session):
    with pytest.raises(ValueError, match="side"):
        engine.place_logic_order(
            symbol_base="TCS", quantity=1, side="HODL", db=db_session,
            reference_price=100.0,
        )


def test_invalid_quantity_rejected(engine, db_session):
    with pytest.raises(ValueError, match="quantity"):
        engine.place_logic_order(
            symbol_base="TCS", quantity=0, side="BUY", db=db_session,
            reference_price=100.0,
        )


def test_position_size():
    # 10000 default capital per slot
    assert position_size(2500.0) == 4
    assert position_size(9999.0) == 1
    assert position_size(50_000.0) == 1  # never zero
    with pytest.raises(ValueError):
        position_size(0.0)
