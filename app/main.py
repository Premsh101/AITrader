"""
main.py – FastAPI application entry point for AITrader.

Startup (lifespan):
  • Loads the three AI brains from /app/models.
  • Launches a background asyncio loop that fires every 60 seconds.

Trading loop (every 60 s):
  1. Fetch OHLCV data for the 1 500-stock universe via ShoonyaEngine.
  2. Generate RSI/MACD/etc. features with pandas_ta.
  3. Hunter finds BUY signals.
  4. Executive selects up to 5 slots to fill.
  5. Guardian checks every OPEN trade and closes those it recommends.

Endpoints:
  GET  /health        – liveness probe
  GET  /stats         – total P&L, win rate, daily P&L, active positions
  GET  /trades        – paginated trade history
  GET  /activity      – last N activity-log entries
  POST /toggle-mode   – switch between PAPER and LIVE in SystemConfig
  GET  /config        – current system config
  PATCH /config       – update config fields
  POST /order         – manual order placement
"""

from __future__ import annotations

import asyncio
import logging
from collections import deque
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.ai_brains import BrainManager
from app.database import get_db, SessionLocal
from app.feature_engine import generate_features
from app.models import SystemConfig, Trade, TradeMode, TradeStatus
from app.shoonya_service import ShoonyaEngine
from app.stock_list import NSE_SYMBOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons
# ---------------------------------------------------------------------------

shoonya = ShoonyaEngine()
brains = BrainManager()

# In-memory activity log (thread-safe deque)
activity_log: deque[dict[str, Any]] = deque(maxlen=200)

TRADING_LOOP_INTERVAL = 60  # seconds


def _log_activity(agent: str, message: str) -> None:
    activity_log.appendleft(
        {
            "agent": agent,
            "message": message,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    logger.info("[%s] %s", agent, message)


# ---------------------------------------------------------------------------
# Background trading loop
# ---------------------------------------------------------------------------


async def _trading_loop() -> None:
    """Runs every TRADING_LOOP_INTERVAL seconds."""
    while True:
        await asyncio.sleep(TRADING_LOOP_INTERVAL)
        try:
            await asyncio.to_thread(_run_trading_cycle)
        except Exception:
            logger.exception("Unhandled error in trading loop")


def _run_trading_cycle() -> None:
    """Synchronous trading cycle executed inside a thread pool."""
    db: Session = SessionLocal()
    try:
        _log_activity("System", "Trading cycle started")

        # ── 1. Fetch market data ──────────────────────────────────────────
        _log_activity("Hunter", f"Fetching market data for {len(NSE_SYMBOLS)} symbols…")
        market_data = shoonya.get_market_data(NSE_SYMBOLS)
        if not market_data:
            _log_activity("System", "No market data returned – skipping cycle")
            return

        # ── 2. Generate features ──────────────────────────────────────────
        symbol_features: dict[str, Any] = {}
        for sym, df in market_data.items():
            feat = generate_features(df)
            if feat is not None:
                symbol_features[sym] = feat

        _log_activity("Hunter", f"Features computed for {len(symbol_features)} symbols")

        # ── 3. Hunter finds buy signals ───────────────────────────────────
        buy_signals = brains.hunter.find_signals(symbol_features)
        _log_activity(
            "Hunter",
            f"Found {len(buy_signals)} buy signal(s): {', '.join(buy_signals[:10])}{'…' if len(buy_signals) > 10 else ''}",
        )

        # ── 4. Executive selects slots ────────────────────────────────────
        open_count = db.query(Trade).filter(Trade.status == TradeStatus.OPEN).count()
        available_slots = max(0, 5 - open_count)
        selected = brains.executive.select_slots(buy_signals, symbol_features, available_slots)

        config: SystemConfig | None = db.query(SystemConfig).first()
        is_live = config.is_live_mode if config else False
        mode = TradeMode.LIVE if is_live else TradeMode.PAPER

        for sym in selected:
            _log_activity("Executive", f"Placing {mode.value} BUY order → {sym}")
            try:
                shoonya.place_logic_order(
                    symbol=sym,
                    quantity=1,
                    side="BUY",
                    db=db,
                )
            except Exception:
                logger.exception("Order placement failed for %s", sym)

        # ── 5. Guardian checks open trades ────────────────────────────────
        open_trades = db.query(Trade).filter(Trade.status == TradeStatus.OPEN).all()
        for trade in open_trades:
            feat = symbol_features.get(trade.symbol)
            if feat is None:
                continue
            if brains.guardian.should_close(feat):
                _log_activity("Guardian", f"Closing position → {trade.symbol}")
                try:
                    # Fetch current price to calculate P&L
                    ltp = shoonya.get_live_price(trade.symbol) if is_live else None
                    if ltp is None:
                        # Use last close from market data as proxy
                        df = market_data.get(trade.symbol)
                        ltp = float(df["close"].iloc[-1]) if df is not None else None

                    if ltp is None or ltp == 0.0:
                        logger.warning(
                            "Cannot close trade id=%s – price unavailable, skipping", trade.id
                        )
                        continue

                    trade.sell_price = Decimal(str(ltp))
                    if trade.buy_price is not None:
                        trade.pnl = Decimal(str(ltp)) - trade.buy_price
                    trade.status = TradeStatus.CLOSED
                    trade.updated_at = datetime.now(timezone.utc)
                    db.commit()
                except Exception:
                    db.rollback()
                    logger.exception("Guardian failed to close trade id=%s", trade.id)
            else:
                _log_activity("Guardian", f"Holding position → {trade.symbol}")

        # ── Update last_sync_time ─────────────────────────────────────────
        if config:
            config.last_sync_time = datetime.now(timezone.utc)
            db.commit()

        _log_activity("System", "Trading cycle complete")

    finally:
        db.close()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(application: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────
    logger.info("Loading AI brains…")
    brains.load_all()

    task = asyncio.create_task(_trading_loop())
    logger.info("Background trading loop started (interval=%ds)", TRADING_LOOP_INTERVAL)

    yield

    # ── Shutdown ──────────────────────────────────────────────────────────
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    logger.info("Background trading loop stopped")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AITrader",
    description="Production-grade AI trading bot with Paper/Live toggle.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class OrderRequest(BaseModel):
    symbol: str
    quantity: int
    side: str  # "BUY" or "SELL"


class LiveModeUpdate(BaseModel):
    is_live_mode: bool


class ToggleModeRequest(BaseModel):
    mode: str  # "PAPER" or "LIVE"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.get("/stats", response_model=dict)
def get_stats(db: Session = Depends(get_db)):
    """Return aggregate P&L statistics and active position count."""
    # Use IST (Asia/Kolkata) for NSE market day boundaries
    try:
        from zoneinfo import ZoneInfo
        ist = ZoneInfo("Asia/Kolkata")
    except Exception:
        from datetime import timezone as _tz
        ist = _tz.utc  # fallback if zoneinfo unavailable

    now_ist = datetime.now(ist)
    today_start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_utc = today_start_ist.astimezone(timezone.utc)

    closed_trades = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED).all()

    total_pnl = sum(float(t.pnl) for t in closed_trades if t.pnl is not None)
    winning_trades = [t for t in closed_trades if t.pnl is not None and float(t.pnl) > 0]
    win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0.0

    daily_closed = [
        t for t in closed_trades
        if t.updated_at and t.updated_at >= today_start_utc
    ]
    daily_pnl = sum(float(t.pnl) for t in daily_closed if t.pnl is not None)

    active_positions = db.query(Trade).filter(Trade.status == TradeStatus.OPEN).count()

    return {
        "total_pnl": round(total_pnl, 2),
        "daily_pnl": round(daily_pnl, 2),
        "win_rate": round(win_rate, 2),
        "active_positions": active_positions,
        "total_trades": len(closed_trades),
        "winning_trades": len(winning_trades),
    }


@app.get("/trades", response_model=list)
def list_trades(
    limit: int = 50,
    offset: int = 0,
    db: Session = Depends(get_db),
):
    """Return recent trade records."""
    trades = db.query(Trade).order_by(Trade.id.desc()).offset(offset).limit(limit).all()
    return [
        {
            "id": t.id,
            "symbol": t.symbol,
            "quantity": t.quantity,
            "mode": t.mode,
            "status": t.status,
            "buy_price": str(t.buy_price) if t.buy_price is not None else None,
            "sell_price": str(t.sell_price) if t.sell_price is not None else None,
            "pnl": str(t.pnl) if t.pnl is not None else None,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        }
        for t in trades
    ]


@app.get("/activity", response_model=list)
def get_activity(limit: int = 50):
    """Return the most recent activity-log entries."""
    return list(activity_log)[:limit]


@app.post("/toggle-mode", response_model=dict)
def toggle_mode(payload: ToggleModeRequest, db: Session = Depends(get_db)):
    """Switch the system between PAPER and LIVE mode."""
    mode_upper = payload.mode.upper()
    if mode_upper not in ("PAPER", "LIVE"):
        raise HTTPException(status_code=400, detail="mode must be 'PAPER' or 'LIVE'")

    is_live = mode_upper == "LIVE"
    config = db.query(SystemConfig).first()
    if not config:
        config = SystemConfig(is_live_mode=is_live)
        db.add(config)
    else:
        config.is_live_mode = is_live
    db.commit()
    db.refresh(config)

    label = "LIVE" if is_live else "PAPER"
    _log_activity("System", f"Mode switched to {label}")
    return {"mode": label, "is_live_mode": config.is_live_mode}


@app.get("/config", response_model=dict)
def get_config(db: Session = Depends(get_db)):
    """Return the current system configuration."""
    config = db.query(SystemConfig).first()
    if not config:
        return {"is_live_mode": False, "last_sync_time": None}
    return {
        "is_live_mode": config.is_live_mode,
        "last_sync_time": config.last_sync_time,
    }


@app.patch("/config", response_model=dict)
def update_config(payload: LiveModeUpdate, db: Session = Depends(get_db)):
    """Toggle Paper / Live mode at runtime."""
    config = db.query(SystemConfig).first()
    if not config:
        config = SystemConfig(is_live_mode=payload.is_live_mode)
        db.add(config)
    else:
        config.is_live_mode = payload.is_live_mode
    db.commit()
    db.refresh(config)
    return {"is_live_mode": config.is_live_mode}


@app.post("/order", response_model=dict)
def place_order(payload: OrderRequest, db: Session = Depends(get_db)):
    """Place a paper or live order depending on the current mode."""
    try:
        trade = shoonya.place_logic_order(
            symbol=payload.symbol,
            quantity=payload.quantity,
            side=payload.side,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    return {
        "id": trade.id,
        "symbol": trade.symbol,
        "quantity": trade.quantity,
        "mode": trade.mode,
        "status": trade.status,
        "buy_price": str(trade.buy_price) if trade.buy_price is not None else None,
        "sell_price": str(trade.sell_price) if trade.sell_price is not None else None,
    }
