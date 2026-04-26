"""
shoonya_service.py – Shoonya (Finvasia) broker integration with Paper/Live toggle.

The ShoonyaEngine class:
  - Authenticates via TOTP using the NorenRestApi SDK.
  - Reads the is_live_mode flag from the database before every order.
  - In Paper Mode: simulates the trade and records it without calling the broker API.
  - In Live Mode:  calls api.place_order and records the real order.
  - get_market_data(): fetches OHLCV data via yfinance (Paper) or Shoonya quotes (Live).
"""

import logging
import os
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy.orm import Session

from app.models import SystemConfig, Trade, TradeMode, TradeStatus

logger = logging.getLogger(__name__)


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

    def get_market_data(
        self,
        symbols: list[str],
        period: str = "3mo",
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV history for a list of symbols.

        In Paper mode (or when the Shoonya API is unavailable) data is fetched
        from Yahoo Finance via *yfinance*.  Symbols must be in Yahoo Finance
        format (e.g. ``"RELIANCE.NS"``).

        Args:
            symbols: List of ticker symbols (Yahoo Finance format).
            period:  yfinance period string (default ``"3mo"``).

        Returns:
            Dict mapping symbol → OHLCV DataFrame (columns lower-cased).
            Symbols that fail to download are omitted.
        """
        import yfinance as yf  # type: ignore[import]

        result: dict[str, pd.DataFrame] = {}
        batch_size = 100  # yfinance handles ~100 tickers efficiently per call

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i : i + batch_size]
            try:
                raw = yf.download(
                    tickers=batch,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if raw.empty:
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
            except Exception:
                logger.exception("Market data fetch failed for batch starting at index %d", i)

        logger.info("Fetched market data for %d / %d symbols", len(result), len(symbols))
        return result

    # ------------------------------------------------------------------
    # Price feed
    # ------------------------------------------------------------------

    def get_live_price(self, symbol: str) -> float | None:
        """Fetch the Last Traded Price (LTP) for *symbol* from Shoonya.

        Returns the LTP as a float, or None if the price cannot be obtained.
        The API must be logged in before calling this method.

        Args:
            symbol: NSE trading symbol, e.g. ``"NSE:RELIANCE-EQ"``.
        """
        if self._api is None:
            logger.warning("get_live_price called before login; attempting login.")
            if not self.login():
                return None

        if ":" not in symbol:
            raise ValueError(
                f"Invalid symbol format '{symbol}'. Expected 'EXCHANGE:SYMBOL', "
                "e.g. 'NSE:RELIANCE-EQ'."
            )

        try:
            exchange, trading_symbol = symbol.split(":", 1)
            response = self._api.get_quotes(exchange=exchange, token=trading_symbol)
            if response and response.get("stat") == "Ok":
                ltp = float(response["lp"])
                logger.debug("LTP for %s: %s", symbol, ltp)
                return ltp
            logger.warning("Could not fetch LTP for %s: %s", symbol, response)
            return None
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error fetching LTP for %s: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def place_logic_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        db: Session,
    ) -> Trade:
        """Place a buy or sell order, respecting the Paper/Live toggle.

        The method reads ``SystemConfig.is_live_mode`` from the database at
        call time so that the toggle takes effect without restarting the service.

        Args:
            symbol:   NSE trading symbol, e.g. ``"NSE:RELIANCE-EQ"``.
            quantity: Number of shares/lots to trade.
            side:     ``"BUY"`` or ``"SELL"`` (case-insensitive).
            db:       Active SQLAlchemy session (injected via FastAPI dependency).

        Returns:
            The persisted :class:`Trade` ORM object.
        """
        side = side.upper()
        if side not in ("BUY", "SELL"):
            raise ValueError(f"Invalid side '{side}'. Must be 'BUY' or 'SELL'.")

        # Read live-mode flag; default to Paper if no config row exists yet
        config: SystemConfig | None = db.query(SystemConfig).first()
        is_live = config.is_live_mode if config else False

        ltp = self.get_live_price(symbol)

        if is_live:
            trade = self._place_live_order(symbol, quantity, side, ltp, db)
        else:
            trade = self._place_paper_order(symbol, quantity, side, ltp, db)

        return trade

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _place_paper_order(
        self,
        symbol: str,
        quantity: int,
        side: str,
        ltp: float | None,
        db: Session,
    ) -> Trade:
        """Simulate a trade without calling the broker API."""
        price = ltp or 0.0
        logger.info(
            "[PAPER] Simulated %s %d x %s @ %.4f", side, quantity, symbol, price
        )

        trade = Trade(
            symbol=symbol,
            buy_price=price if side == "BUY" else None,
            sell_price=price if side == "SELL" else None,
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
        symbol: str,
        quantity: int,
        side: str,
        ltp: float | None,
        db: Session,
    ) -> Trade:
        """Execute a real order via the Shoonya API."""
        if self._api is None:
            logger.warning("Live order requested but not logged in; attempting login.")
            if not self.login():
                raise RuntimeError("Could not authenticate with Shoonya API.")

        try:
            if ":" not in symbol:
                raise ValueError(
                    f"Invalid symbol format '{symbol}'. Expected 'EXCHANGE:SYMBOL', "
                    "e.g. 'NSE:RELIANCE-EQ'."
                )

            exchange, trading_symbol = symbol.split(":", 1)
            transaction_type = "B" if side == "BUY" else "S"

            response = self._api.place_order(
                buy_or_sell=transaction_type,
                product_type="I",      # Intraday
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

            logger.info(
                "[LIVE] Order placed – %s %d x %s. Response: %s",
                side, quantity, symbol, response,
            )

            price = ltp or 0.0
            trade = Trade(
                symbol=symbol,
                buy_price=price if side == "BUY" else None,
                sell_price=price if side == "SELL" else None,
                quantity=quantity,
                status=TradeStatus.OPEN,
                mode=TradeMode.LIVE,
            )
            db.add(trade)
            db.commit()
            db.refresh(trade)
            return trade

        except Exception as exc:
            db.rollback()
            logger.exception("Error placing live order for %s: %s", symbol, exc)
            raise
