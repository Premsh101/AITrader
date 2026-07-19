"""
shoonya_service.py – Shoonya (Finvasia) broker integration with Paper/Live toggle.

The ShoonyaEngine class:
  - Authenticates via TOTP using the NorenRestApi SDK.
  - Reads the is_live_mode flag from the database before every order.
  - In Paper Mode: simulates the trade WITHOUT any broker call; the fill price
    is the caller-supplied ``reference_price`` (a 0.0-price trade is impossible).
  - In Live Mode: places a real order, storing the broker order id.
  - get_market_data(): fetches OHLCV data via yfinance.

Symbol convention: all public methods take the BASE NSE symbol (e.g.
``"RELIANCE"``).  Conversion to Yahoo/Shoonya formats happens internally via
:mod:`app.symbols`.
"""

import logging
import math
import os
import time
from typing import Optional

import pandas as pd
import requests
from sqlalchemy.orm import Session

from app import symbols
from app.models import SystemConfig, Trade, TradeMode, TradeStatus

logger = logging.getLogger(__name__)

# Rupees of capital allocated per portfolio slot; sizes orders as
# floor(capital / price) with a minimum of one share.
TRADE_CAPITAL_PER_SLOT = float(os.environ.get("TRADE_CAPITAL_PER_SLOT", "10000"))


def position_size(reference_price: float) -> int:
    """Number of shares to buy for one slot at *reference_price*."""
    if reference_price <= 0:
        raise ValueError(f"reference_price must be positive, got {reference_price}")
    return max(1, math.floor(TRADE_CAPITAL_PER_SLOT / reference_price))


class ShoonyaEngine:
    """Handles all broker interactions and enforces the Paper/Live toggle."""

    def __init__(self) -> None:
        self._api = None  # Lazily initialised on first login

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def login(self) -> bool:
        """Authenticate with the Shoonya API using TOTP.

        Credentials are read from environment variables so that secrets are
        never hard-coded in source code.

        Required environment variables:
            SHOONYA_USER      – Shoonya user ID
            SHOONYA_PASSWORD  – Shoonya password
            SHOONYA_TOTP_KEY  – Base-32 TOTP secret from the Shoonya app
            SHOONYA_VENDOR_CODE – Vendor / API key code
            SHOONYA_API_SECRET  – API secret
            SHOONYA_IMEI        – Device IMEI registered with the account
        """
        try:
            # Import here to keep the module loadable even if the SDK is absent
            # (e.g. during unit tests with mocks).
            import pyotp
            from NorenRestApiPy.NorenApi import NorenApi  # type: ignore[import]

            user = os.environ["SHOONYA_USER"]
            password = os.environ["SHOONYA_PASSWORD"]
            totp_key = os.environ["SHOONYA_TOTP_KEY"]
            vendor_code = os.environ["SHOONYA_VENDOR_CODE"]
            api_secret = os.environ["SHOONYA_API_SECRET"]
            imei = os.environ["SHOONYA_IMEI"]

            totp = pyotp.TOTP(totp_key).now()

            api = NorenApi(
                host="https://api.shoonya.com/NorenWClientTP/",
                websocket="wss://api.shoonya.com/NorenWSTP/",
            )
            response = api.login(
                userid=user,
                password=password,
                twoFA=totp,
                vendor_code=vendor_code,
                api_secret=api_secret,
                imei=imei,
            )

            if response and response.get("stat") == "Ok":
                self._api = api
                logger.info("Shoonya login successful for user %s", user)
                return True

            logger.error("Shoonya login failed: %s", response)
            return False

        except KeyError as exc:
            logger.error("Missing environment variable for Shoonya login: %s", exc)
            return False
        except Exception as exc:  # noqa: BLE001
            logger.exception("Unexpected error during Shoonya login: %s", exc)
            return False

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    # User-Agent that mimics a standard Chrome browser on Windows so that
    # Yahoo Finance does not identify the request as a server-side scraper
    # and block it (the most common cause of JSONDecodeError on a VPS).
    _YF_USER_AGENT = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
    _YF_MAX_RETRIES = 3
    _YF_RETRY_BACKOFF = 5  # seconds between retries

    def _yf_session(self) -> requests.Session:
        """Return a requests Session with a browser-like User-Agent header."""
        session = requests.Session()
        session.headers.update({"User-Agent": self._YF_USER_AGENT})
        return session

    def get_market_data(
        self,
        yahoo_symbols: list[str],
        period: str = "3mo",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV history for a list of Yahoo Finance tickers.

        Data is fetched from Yahoo Finance via *yfinance*; features are daily
        indicators, so daily bars suffice for both Paper and Live modes.

        A custom requests Session with a browser User-Agent is injected so that
        Yahoo Finance does not block VPS/server traffic.  Failed batches are
        retried up to ``_YF_MAX_RETRIES`` times with ``_YF_RETRY_BACKOFF``-second
        delays before being skipped.

        Args:
            yahoo_symbols: Tickers in Yahoo Finance format (``"RELIANCE.NS"``,
                           index tickers like ``"^NSEI"`` pass through as-is).
            period:        yfinance period string (default ``"3mo"``).

        Returns:
            Dict mapping the *input* Yahoo ticker → OHLCV DataFrame (columns
            lower-cased).  Symbols that fail to download are omitted.  Callers
            re-key to base symbols at the edge via :func:`app.symbols.to_base`.
        """
        import yfinance as yf  # type: ignore[import]

        result: dict[str, pd.DataFrame] = {}
        batch_size = 100  # yfinance handles ~100 tickers efficiently per call
        session = self._yf_session()

        for i in range(0, len(yahoo_symbols), batch_size):
            batch = yahoo_symbols[i : i + batch_size]
            raw = None

            for attempt in range(1, self._YF_MAX_RETRIES + 1):
                try:
                    raw = yf.download(
                        tickers=batch,
                        period=period,
                        interval="1d",
                        group_by="ticker",
                        auto_adjust=True,
                        progress=False,
                        threads=True,
                        session=session,
                    )
                    if raw is not None and not raw.empty:
                        break  # successful download
                    logger.warning(
                        "Empty data for batch at index %d (attempt %d/%d) – retrying in %ds",
                        i, attempt, self._YF_MAX_RETRIES, self._YF_RETRY_BACKOFF,
                    )
                except Exception:
                    logger.exception(
                        "Market data fetch failed for batch at index %d (attempt %d/%d)",
                        i, attempt, self._YF_MAX_RETRIES,
                    )
                if attempt < self._YF_MAX_RETRIES:
                    time.sleep(self._YF_RETRY_BACKOFF)

            if raw is None or raw.empty:
                logger.warning(
                    "[System] Market data fetch failed - check connectivity or IP block."
                )
                continue

            for sym in batch:
                try:
                    if len(batch) == 1:
                        df = raw.copy()
                    else:
                        df = raw[sym].copy()
                    df = df.dropna(how="all")
                    if df.empty or len(df) < 30:
                        continue
                    df.columns = [c.lower() for c in df.columns]
                    result[sym] = df
                except (KeyError, TypeError):
                    pass

        logger.info(
            "Fetched market data for %d / %d symbols", len(result), len(yahoo_symbols)
        )
        return result

    # ------------------------------------------------------------------
    # Price feed
    # ------------------------------------------------------------------

    def get_live_price(self, symbol_base: str) -> float | None:
        """Fetch the Last Traded Price (LTP) for a base symbol from Shoonya.

        The Shoonya-format symbol is built internally, so the caller never
        deals with broker formats.  Returns the LTP as a float, or ``None``
        if the price cannot be obtained (including when login fails).

        Args:
            symbol_base: Base NSE symbol, e.g. ``"RELIANCE"``.
        """
        if self._api is None:
            logger.warning("get_live_price called before login; attempting login.")
            if not self.login():
                return None

        exchange, trading_symbol = symbols.to_shoonya(symbol_base)

        try:
            response = self._api.get_quotes(exchange=exchange, token=trading_symbol)
            if response and response.get("stat") == "Ok":
                ltp = float(response["lp"])
                logger.debug("LTP for %s: %s", symbol_base, ltp)
                return ltp
            logger.warning("Could not fetch LTP for %s: %s", symbol_base, response)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error fetching LTP for %s: %s", symbol_base, exc)
            return None

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_logic_order(
        self,
        symbol_base: str,
        quantity: int,
        side: str,
        db: Session,
        reference_price: Optional[float] = None,
    ) -> Trade:
        """Place a buy or sell order, respecting the Paper/Live toggle.

        The method reads ``SystemConfig.is_live_mode`` from the database at
        call time so that the toggle takes effect without restarting the
        service.

        Args:
            symbol_base:     Base NSE symbol, e.g. ``"RELIANCE"``.
            quantity:        Number of shares to trade.
            side:            ``"BUY"`` or ``"SELL"`` (case-insensitive).
            db:              Active SQLAlchemy session.
            reference_price: Last known price (e.g. last close from market
                             data).  REQUIRED in paper mode; used as the
                             fallback fill price in live mode.

        Returns:
            The persisted :class:`Trade` ORM object.
        """
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side '{side}'. Must be 'BUY' or 'SELL'.")
        if quantity <= 0:
            raise ValueError(f"Invalid quantity {quantity}. Must be positive.")

        symbol_base = symbols.to_base(symbol_base)

        # Read live-mode flag; default to Paper if no config row exists yet
        config: SystemConfig | None = db.query(SystemConfig).first()
        is_live = config.is_live_mode if config else False

        if is_live:
            return self._place_live_order(
                symbol_base, quantity, side, reference_price, db
            )
        return self._place_paper_order(
            symbol_base, quantity, side, reference_price, db
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_paper_order(
        self,
        symbol_base: str,
        quantity: int,
        side: str,
        reference_price: Optional[float],
        db: Session,
    ) -> Trade:
        """Simulate a trade – never touches the broker API or live price feed.

        Raises:
            ValueError: if *reference_price* is missing or non-positive; a
                        paper trade recorded at price 0.0 must be impossible.
        """
        if reference_price is None or reference_price <= 0:
            raise ValueError(
                f"Paper order for {symbol_base} requires a positive "
                f"reference_price (got {reference_price})."
            )

        logger.info(
            "[PAPER] Simulated %s %d x %s @ %.4f",
            side, quantity, symbol_base, reference_price,
        )

        trade = Trade(
            symbol=symbol_base,
            buy_price=reference_price if side == "BUY" else None,
            sell_price=reference_price if side == "SELL" else None,
            quantity=quantity,
            status=TradeStatus.OPEN,
            mode=TradeMode.PAPER,
        )
        db.add(trade)
        db.commit()
        db.refresh(trade)
        return trade

    def _place_live_order(
        self,
        symbol_base: str,
        quantity: int,
        side: str,
        reference_price: Optional[float],
        db: Session,
    ) -> Trade:
        """Execute a real order via the Shoonya API."""
        if self._api is None:
            logger.warning("Live order requested but not logged in; attempting login.")
            if not self.login():
                raise RuntimeError("Could not authenticate with Shoonya API.")

        try:
            exchange, trading_symbol = symbols.to_shoonya(symbol_base)
            transaction_type = "B" if side == "BUY" else "S"

            response = self._api.place_order(
                buy_or_sell=transaction_type,
                product_type="I",      # Intraday – broker squares off ~15:20 IST
                exchange=exchange,
                tradingsymbol=trading_symbol,
                quantity=quantity,
                discloseqty=0,
                price_type="MKT",      # Market order
                price=0,
                trigger_price=None,
                retention="DAY",
                remarks="AITrader",
            )

            if not (response and response.get("stat") == "Ok"):
                raise RuntimeError(f"Shoonya order placement failed: {response}")

            broker_order_id = response.get("norenordno")

            logger.info(
                "[LIVE] Order placed – %s %d x %s (order id %s). Response: %s",
                side, quantity, symbol_base, broker_order_id, response,
            )

            # Best-effort fill price: LTP immediately after placement, falling
            # back to the caller's reference price.  The true fill price would
            # require polling the order-book/trade-book API for this order id;
            # this is a known approximation.
            fill_price = self.get_live_price(symbol_base) or reference_price or 0.0

            trade = Trade(
                symbol=symbol_base,
                buy_price=fill_price if side == "BUY" else None,
                sell_price=fill_price if side == "SELL" else None,
                quantity=quantity,
                status=TradeStatus.OPEN,
                mode=TradeMode.LIVE,
                broker_order_id=broker_order_id,
            )
            db.add(trade)
            db.commit()
            db.refresh(trade)
            return trade

        except Exception as exc:
            db.rollback()
            logger.exception("Error placing live order for %s: %s", symbol_base, exc)
            raise
