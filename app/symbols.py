"""
symbols.py – Symbol-format conversion helpers for AITrader.

Canonical convention: the database and all internal logic store the *base*
NSE symbol (e.g. ``"RELIANCE"``).  Conversion to provider-specific formats
happens only at the edges:

  • Yahoo Finance (yfinance) wants ``"RELIANCE.NS"``   → :func:`to_yahoo`
  • Shoonya (Finvasia) wants ``("NSE", "RELIANCE-EQ")`` → :func:`to_shoonya`
  • Anything coming back from Yahoo is normalised with  :func:`to_base`

All three functions are pure and idempotent where it makes sense, so calling
``to_yahoo`` on an already-suffixed symbol is safe.
"""

from __future__ import annotations

YAHOO_SUFFIX = ".NS"
SHOONYA_EXCHANGE = "NSE"
SHOONYA_EQ_SUFFIX = "-EQ"


def to_yahoo(base: str) -> str:
    """Convert a base NSE symbol to Yahoo Finance format.

    Idempotent: ``to_yahoo("RELIANCE") == to_yahoo("RELIANCE.NS") == "RELIANCE.NS"``.

    Args:
        base: Base symbol, e.g. ``"RELIANCE"`` (a ``".NS"`` suffix is tolerated).

    Returns:
        Yahoo Finance ticker, e.g. ``"RELIANCE.NS"``.
    """
    base = base.strip().upper()
    if base.endswith(YAHOO_SUFFIX):
        return base
    return f"{base}{YAHOO_SUFFIX}"


def to_base(yahoo: str) -> str:
    """Strip the Yahoo Finance suffix, returning the base NSE symbol.

    Idempotent: passing an already-bare symbol returns it unchanged.

    Args:
        yahoo: Yahoo Finance ticker, e.g. ``"RELIANCE.NS"``.

    Returns:
        Base symbol, e.g. ``"RELIANCE"``.
    """
    yahoo = yahoo.strip().upper()
    if yahoo.endswith(YAHOO_SUFFIX):
        return yahoo[: -len(YAHOO_SUFFIX)]
    return yahoo


def to_shoonya(base: str) -> tuple[str, str]:
    """Convert a base NSE symbol to the (exchange, tradingsymbol) pair Shoonya expects.

    Args:
        base: Base symbol, e.g. ``"RELIANCE"`` (Yahoo-suffixed input is tolerated).

    Returns:
        Tuple of exchange and trading symbol, e.g. ``("NSE", "RELIANCE-EQ")``.
    """
    base = to_base(base)
    if base.endswith(SHOONYA_EQ_SUFFIX):
        return (SHOONYA_EXCHANGE, base)
    return (SHOONYA_EXCHANGE, f"{base}{SHOONYA_EQ_SUFFIX}")
