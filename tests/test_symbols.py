"""Tests for the app.symbols mapping module."""

from app import symbols


def test_to_yahoo_basic():
    assert symbols.to_yahoo("RELIANCE") == "RELIANCE.NS"


def test_to_yahoo_idempotent():
    assert symbols.to_yahoo("RELIANCE.NS") == "RELIANCE.NS"
    assert symbols.to_yahoo(symbols.to_yahoo("TCS")) == "TCS.NS"


def test_to_base():
    assert symbols.to_base("RELIANCE.NS") == "RELIANCE"
    assert symbols.to_base("RELIANCE") == "RELIANCE"


def test_index_ticker_passthrough():
    assert symbols.to_base("^NSEI") == "^NSEI"


def test_to_shoonya():
    assert symbols.to_shoonya("RELIANCE") == ("NSE", "RELIANCE-EQ")
    assert symbols.to_shoonya("RELIANCE.NS") == ("NSE", "RELIANCE-EQ")


def test_round_trip():
    for base in ["RELIANCE", "M&M", "BAJAJ-AUTO", "NAM-INDIA"]:
        assert symbols.to_base(symbols.to_yahoo(base)) == base
        exchange, trading = symbols.to_shoonya(base)
        assert exchange == "NSE"
        assert trading == f"{base}-EQ"


def test_hyphenated_symbols_not_mangled():
    # BAJAJ-AUTO ends in "-AUTO", not "-EQ"; the EQ suffix must still be added.
    assert symbols.to_shoonya("BAJAJ-AUTO") == ("NSE", "BAJAJ-AUTO-EQ")
