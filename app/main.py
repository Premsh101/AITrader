"""
main.py – FastAPI application entry point for AITrader.

Startup (lifespan):
  • Loads the three AI brains from /app/models and validates their
    observation dimensions.
  • Launches a background asyncio loop.

Trading loop (every TRADING_LOOP_INTERVAL seconds, default 300):
  0. Gate: NSE market hours (Mon–Fri 09:15–15:30 IST) and brain readiness.
     If the brains are not ready the cycle is SKIPPED with an ERROR — the
     system never trades on anything but the validated models.
  1. Fetch daily OHLCV for the universe + ^NSEI (TTL-cached; daily bars do
     not change intraday, so Yahoo is only hit every FETCH_TTL_SECONDS).
  2. Generate 15-dim market features per symbol.
  3. Hunter finds BUY signals; symbols already held are dropped.
  4. Executive ranks candidates by approve probability and fills open slots
     (no new entries after 15:15 IST – intraday product is squared off by
     the broker around 15:20).
  5. Guardian evaluates every OPEN trade with a 17-dim observation
     (market features + unrealized P&L + bars in trade) and closes those
     it recommends.

Endpoints:
  GET  /health        – liveness probe
  GET  /stats         – total P&L, win rate, daily P&L, active positions
  GET  /trades        – paginated trade history
  GET  /activity      – last N activity-log entries
  POST /toggle-mode   – switch PAPER/LIVE (API key; LIVE needs confirm=true)
  GET  /config        – current system config
  PATCH /config       – update config fields (API key)
  POST /order         – manual order placement (API key)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from app import risk, symbols
from app.ai_brains import MAX_SLOTS, BrainManager
from app.database import get_db, SessionLocal
from app.feature_engine import (
    build_executive_obs,
    build_guardian_obs,
    generate_features,
)
from app.models import GhostTrade, SystemConfig, Trade, TradeMode, TradeStatus
from app.security import allowed_origins, require_api_key
from app.shoonya_service import TRADE_CAPITAL_PER_SLOT, ShoonyaEngine
from app.stock_list import NSE_SYMBOLS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global singletons & configuration
# ---------------------------------------------------------------------------

shoonya = ShoonyaEngine()
brains = BrainManager()

# In-memory activity log (thread-safe deque)
activity_log: deque[dict[str, Any]] = deque(maxlen=200)

IST = ZoneInfo("Asia/Kolkata")

TRADING_LOOP_INTERVAL = int(os.environ.get("TRADING_LOOP_INTERVAL", "300"))
# Daily-bar features don't change intraday; cache Yahoo fetches this long.
FETCH_TTL_SECONDS = int(os.environ.get("FETCH_TTL_SECONDS", "900"))
# Set TRADING_ALWAYS_ON=1 to bypass the market-hours gate (testing only).
TRADING_ALWAYS_ON = os.environ.get("TRADING_ALWAYS_ON", "0") == "1"

NIFTY_YAHOO = "^NSEI"

# TTL cache for market data: {"ts": monotonic timestamp, "data": {base: df}}
_market_cache: dict[str, Any] = {"ts": 0.0, "data": {}}


def _log_activity(agent: str, message: str, level: int = logging.INFO) -> None:
    activity_log.appendleft(
        {
            "agent": agent,
            "message": message,
            "level": logging.getLevelName(level),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    logger.log(level, "[%s] %s", agent, message)


# ---------------------------------------------------------------------------
# Market-hours helpers
# ---------------------------------------------------------------------------


def _is_market_open(now_ist: datetime) -> bool:
    """True Mon–Fri between 09:15 and 15:30 IST (exchange holidays not modelled)."""
    if now_ist.weekday() >= 5:  # Saturday / Sunday
        return False
    minutes = now_ist.hour * 60 + now_ist.minute
    return (9 * 60 + 15) <= minutes <= (15 * 60 + 30)


def _entries_allowed(now_ist: datetime) -> bool:
    """No new entries after 15:15 IST – the intraday product ("I") is
    force-squared-off by the broker around 15:20; the Guardian keeps running
    until close."""
    minutes = now_ist.hour * 60 + now_ist.minute
    return minutes < (15 * 60 + 15)


# ---------------------------------------------------------------------------
# Market data (TTL-cached, keyed by BASE symbol)
# ---------------------------------------------------------------------------


def _get_market_data_cached() -> dict[str, Any]:
    """Fetch daily OHLCV for the universe + ^NSEI, re-keyed to base symbols.

    Results are cached for FETCH_TTL_SECONDS so the trading loop can run more
    often than Yahoo is polled.  The NIFTY index frame is stored under the
    ``"^NSEI"`` key.
    """
    now = time.monotonic()
    if _market_cache["data"] and (now - _market_cache["ts"]) < FETCH_TTL_SECONDS:
        return _market_cache["data"]

    yahoo_tickers = [symbols.to_yahoo(s) for s in NSE_SYMBOLS] + [
        NIFTY_YAHOO,
        risk.VIX_YAHOO,
    ]
    # 1y of history: features need ~60 bars, the regime filter needs 200.
    raw = shoonya.get_market_data(yahoo_tickers, period="1y")
    data = {symbols.to_base(yahoo_sym): df for yahoo_sym, df in raw.items()}

    if data:
        _market_cache["data"] = data
        _market_cache["ts"] = now
    return data


def _close_trade(db: Session, trade: Trade, exit_price: float, reason: str) -> None:
    """Mark an open trade CLOSED at *exit_price*, recording why.

    ``pnl`` is stored NET of estimated round-trip charges (``charges``
    column shows the deduction) so dashboard totals reflect keepable money.
    """
    trade.sell_price = Decimal(str(exit_price))
    if trade.buy_price is not None:
        gross = (Decimal(str(exit_price)) - trade.buy_price) * trade.quantity
        charges = risk.round_trip_charges(
            float(trade.buy_price) * trade.quantity, exit_price * trade.quantity
        )
        trade.charges = Decimal(str(round(charges, 4)))
        trade.pnl = gross - trade.charges
    trade.status = TradeStatus.CLOSED
    trade.exit_reason = reason
    trade.updated_at = datetime.now(timezone.utc)
    db.commit()


def _record_ghosts(db: Session, syms: list[str], reason: str, market_data) -> None:
    """Log rejected Hunter signals to the ghost ledger (dual-ledger design)."""
    for sym in syms:
        price = _last_close(market_data.get(sym))
        if price is not None:
            db.add(GhostTrade(symbol=sym, reference_price=Decimal(str(price)), reason=reason))
    if syms:
        db.commit()


def _evaluate_ghosts(db: Session, market_data) -> None:
    """Backfill 5-bar outcomes for ghost trades older than ~a trading week."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=7)
    pending = (
        db.query(GhostTrade)
        .filter(GhostTrade.evaluated.is_(False), GhostTrade.created_at < cutoff)
        .limit(200)
        .all()
    )
    for ghost in pending:
        df = market_data.get(ghost.symbol)
        created = ghost.created_at
        if created is not None and created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if df is None or created is None:
            continue
        try:
            idx = df.index.tz_localize("UTC") if df.index.tz is None else df.index
            after = df.loc[idx > created, "close"].head(5)
            if len(after) < 5:
                continue
            ref = float(ghost.reference_price)
            ghost.max_gain_pct = Decimal(str(round(float(after.max()) / ref - 1.0, 4)))
            ghost.evaluated = True
        except Exception:
            logger.debug("Ghost evaluation failed for %s", ghost.symbol, exc_info=True)
    db.commit()


def _last_close(df) -> float | None:
    try:
        value = float(df["close"].iloc[-1])
        return value if value > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Background trading loop
# ---------------------------------------------------------------------------


async def _trading_loop() -> None:
    """Runs every TRADING_LOOP_INTERVAL seconds during market hours."""
    while True:
        await asyncio.sleep(TRADING_LOOP_INTERVAL)

        now_ist = datetime.now(IST)
        if not TRADING_ALWAYS_ON and not _is_market_open(now_ist):
            logger.debug("Market closed (%s IST) – sleeping", now_ist.strftime("%H:%M"))
            continue

        try:
            await asyncio.to_thread(_run_trading_cycle)
        except Exception:
            logger.exception("Unhandled error in trading loop")


def _run_trading_cycle() -> None:
    """Synchronous trading cycle executed inside a thread pool."""
    # Hard gate: never trade unless every brain has a validated model.
    if not brains.all_ready:
        _log_activity(
            "System",
            "AI brains not ready (missing/invalid model files) – skipping "
            "trading cycle. No heuristic fallback exists by design.",
            level=logging.ERROR,
        )
        return

    db: Session = SessionLocal()
    try:
        config: SystemConfig | None = db.query(SystemConfig).first()
        if not config:
            config = SystemConfig(is_live_mode=False)
            db.add(config)
            db.commit()
            db.refresh(config)

        # Death-protocol halt: nothing runs until the operator re-enables.
        if config.is_halted:
            _log_activity(
                "Risk",
                "System HALTED by death protocol – no trading until is_halted "
                "is cleared via PATCH /config.",
                level=logging.ERROR,
            )
            return

        _log_activity("System", "Trading cycle started")
        now_ist = datetime.now(IST)

        # ── 1. Fetch market data (TTL-cached) ─────────────────────────────
        market_data = _get_market_data_cached()
        if not market_data:
            _log_activity(
                "System",
                "No market data returned – skipping cycle",
                level=logging.ERROR,
            )
            return

        nifty_df = market_data.get(symbols.to_base(NIFTY_YAHOO))
        nifty_ret_5d = 0.0
        if nifty_df is not None and len(nifty_df) > 5:
            try:
                nifty_ret_5d = float(nifty_df["close"].pct_change(5).iloc[-1])
            except Exception:
                logger.warning("Could not compute ^NSEI 5-day return", exc_info=True)

        # ── 2. Generate features (base-symbol keyed) ──────────────────────
        symbol_features: dict[str, np.ndarray] = {}
        index_keys = {symbols.to_base(NIFTY_YAHOO), symbols.to_base(risk.VIX_YAHOO)}
        for sym, df in market_data.items():
            if sym in index_keys:
                continue
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

        # ── 3b. Equity tracking & death protocol ──────────────────────────
        open_trades = db.query(Trade).filter(Trade.status == TradeStatus.OPEN).all()
        realized = float(
            db.query(func.coalesce(func.sum(Trade.pnl), 0))
            .filter(Trade.status == TradeStatus.CLOSED)
            .scalar()
        )
        unrealized = 0.0
        for t in open_trades:
            last = _last_close(market_data.get(t.symbol))
            if last is not None and t.buy_price is not None:
                unrealized += (last - float(t.buy_price)) * t.quantity
        equity = risk.BASE_CAPITAL + realized + unrealized

        peak_equity = max(float(config.peak_equity or equity), equity)
        config.peak_equity = Decimal(str(round(peak_equity, 2)))
        db.commit()

        if equity < risk.death_threshold(peak_equity):
            _log_activity(
                "Risk",
                f"DEATH PROTOCOL: equity ₹{equity:,.0f} fell below "
                f"{100 * (1 - risk.DEATH_DRAWDOWN):.0f}% of peak "
                f"₹{peak_equity:,.0f}. Flattening all positions and halting.",
                level=logging.ERROR,
            )
            for t in open_trades:
                exit_price = _last_close(market_data.get(t.symbol))
                if exit_price is not None:
                    _close_trade(db, t, exit_price, "death-protocol")
            config.is_halted = True
            db.commit()
            return

        # ── 4. Executive selects slots (excluding held symbols) ───────────
        held = {t.symbol for t in open_trades}
        buy_signals = [s for s in buy_signals if s not in held]

        # Liquidity floor: the 25 bps cost model is only realistic in names
        # that actually trade; drop thin ones before the Executive sees them.
        illiquid = [s for s in buy_signals if not risk.is_liquid(market_data.get(s))]
        if illiquid:
            _log_activity(
                "Risk",
                f"Dropped {len(illiquid)} illiquid signal(s): {', '.join(illiquid[:5])}",
            )
            buy_signals = [s for s in buy_signals if s not in illiquid]
            _record_ghosts(db, illiquid, "illiquid", market_data)

        open_count = len(open_trades)
        available_slots = max(0, MAX_SLOTS - open_count)
        open_positions_frac = open_count / MAX_SLOTS

        executive_obs = {
            sym: build_executive_obs(
                symbol_features[sym], open_positions_frac, nifty_ret_5d
            )
            for sym in buy_signals
            if sym in symbol_features
        }
        selected = brains.executive.select_slots(
            buy_signals, executive_obs, available_slots
        )
        rejected = [s for s in buy_signals if s not in selected]
        _record_ghosts(
            db, rejected,
            "executive-skip" if available_slots > 0 else "no-slots",
            market_data,
        )

        is_live = config.is_live_mode
        mode = TradeMode.LIVE if is_live else TradeMode.PAPER

        if selected and not TRADING_ALWAYS_ON and not _entries_allowed(now_ist):
            _log_activity(
                "Executive",
                f"Skipping {len(selected)} entry(ies) – past 15:15 IST square-off cutoff",
            )
            _record_ghosts(db, selected, "cutoff", market_data)
            selected = []

        # Regime filter: no new entries while NIFTY is below its 200-day
        # average; open positions keep being managed.
        if selected and not risk.regime_allows_entries(nifty_df):
            _log_activity(
                "Risk",
                f"NIFTY below its {risk.REGIME_SMA_BARS}-day average – "
                f"blocking {len(selected)} new entry(ies)",
            )
            _record_ghosts(db, selected, "regime-block", market_data)
            selected = []

        # VIX filter: a fear spike blocks new entries; exits keep running.
        if selected and risk.vix_entries_blocked(
            market_data.get(symbols.to_base(risk.VIX_YAHOO))
        ):
            _log_activity(
                "Risk",
                f"India VIX spiked more than {risk.VIX_SPIKE_PCT:.0%} today – "
                f"blocking {len(selected)} new entry(ies)",
            )
            _record_ghosts(db, selected, "vix-block", market_data)
            selected = []

        for sym in selected:
            reference_price = _last_close(market_data.get(sym))
            if reference_price is None:
                _log_activity(
                    "Executive",
                    f"No reference price for {sym} – order skipped",
                    level=logging.ERROR,
                )
                continue
            # Volatility-scaled sizing: feature 9 is ATR/close for the symbol.
            atr_pct = float(symbol_features[sym][9]) if sym in symbol_features else 0.0
            quantity = risk.vol_scaled_quantity(
                reference_price, atr_pct, TRADE_CAPITAL_PER_SLOT
            )
            _log_activity(
                "Executive",
                f"Placing {mode.value} BUY order → {sym} x{quantity} @ ~{reference_price:.2f}",
            )
            try:
                shoonya.place_logic_order(
                    symbol_base=sym,
                    quantity=quantity,
                    side="BUY",
                    db=db,
                    reference_price=reference_price,
                )
            except Exception:
                _log_activity(
                    "Executive",
                    f"Order placement failed for {sym}",
                    level=logging.ERROR,
                )
                logger.exception("Order placement failed for %s", sym)

        # ── 5. Risk overlays + Guardian check every open trade ───────────
        for trade in open_trades:
            if trade.status != TradeStatus.OPEN:
                continue  # closed above by the death protocol
            feat = symbol_features.get(trade.symbol)
            last_close = _last_close(market_data.get(trade.symbol))
            if feat is None or last_close is None or trade.buy_price is None:
                continue

            buy_price = float(trade.buy_price)
            unrealized_pnl_pct = (last_close - buy_price) / buy_price

            # Track the highest close since entry for the profit ladder.
            peak_price = max(float(trade.peak_price or buy_price), last_close)
            trade.peak_price = Decimal(str(peak_price))
            peak_pnl_pct = (peak_price - buy_price) / buy_price

            # Calendar days ≈ daily bars; close enough for the 0–20 bar
            # feature and avoids needing an exchange calendar.
            created = trade.created_at
            if created is not None and created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            bars_in_trade = (
                (datetime.now(timezone.utc) - created).days if created else 0
            )

            # Deterministic overlays first – the Guardian cannot veto them.
            reason = risk.overlay_exit_reason(
                unrealized_pnl_pct, peak_pnl_pct, bars_in_trade
            )
            if reason is None:
                obs = build_guardian_obs(feat, unrealized_pnl_pct, bars_in_trade)
                if brains.guardian.should_close(obs, symbol=trade.symbol):
                    reason = "guardian"

            if reason is None:
                db.commit()  # persist peak_price update
                _log_activity("Guardian", f"Holding position → {trade.symbol}")
                continue

            agent = "Guardian" if reason == "guardian" else "Risk"
            _log_activity(agent, f"Closing position → {trade.symbol} ({reason})")
            try:
                ltp = shoonya.get_live_price(trade.symbol) if is_live else None
                _close_trade(db, trade, ltp if ltp else last_close, reason)
            except Exception:
                db.rollback()
                _log_activity(
                    agent,
                    f"Failed to close trade id={trade.id} ({trade.symbol})",
                    level=logging.ERROR,
                )
                logger.exception("Failed to close trade id=%s", trade.id)

        # ── Ghost-ledger outcomes ─────────────────────────────────────────
        _evaluate_ghosts(db, market_data)

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
    logger.info(
        "Background trading loop started (interval=%ds, fetch TTL=%ds)",
        TRADING_LOOP_INTERVAL,
        FETCH_TTL_SECONDS,
    )

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class OrderRequest(BaseModel):
    symbol: str
    quantity: int
    side: str  # "BUY" or "SELL"
    reference_price: float | None = None


class ConfigUpdate(BaseModel):
    is_live_mode: bool | None = None
    # Clearing this re-arms trading after the death protocol fired.
    is_halted: bool | None = None


class ToggleModeRequest(BaseModel):
    mode: str  # "PAPER" or "LIVE"
    confirm: bool = False  # must be True when switching to LIVE


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    return {"status": "ok", "brains_ready": brains.all_ready}


@app.get("/stats", response_model=dict)
def get_stats(db: Session = Depends(get_db)):
    """Return aggregate P&L statistics and active position count."""
    now_ist = datetime.now(IST)
    today_start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    today_start_utc = today_start_ist.astimezone(timezone.utc)

    closed_trades = db.query(Trade).filter(Trade.status == TradeStatus.CLOSED).all()

    total_pnl = sum(float(t.pnl) for t in closed_trades if t.pnl is not None)
    winning_trades = [t for t in closed_trades if t.pnl is not None and float(t.pnl) > 0]
    win_rate = (len(winning_trades) / len(closed_trades) * 100) if closed_trades else 0.0

    def _as_utc(dt: datetime | None) -> datetime | None:
        # SQLite returns naive datetimes even for DateTime(timezone=True).
        if dt is not None and dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt

    daily_closed = [
        t for t in closed_trades
        if t.updated_at and _as_utc(t.updated_at) >= today_start_utc
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
            "broker_order_id": t.broker_order_id,
            "exit_reason": t.exit_reason,
            "created_at": t.created_at.isoformat() if t.created_at else None,
            "updated_at": t.updated_at.isoformat() if t.updated_at else None,
        }
        for t in trades
    ]


@app.get("/ghost-trades", response_model=list)
def list_ghost_trades(limit: int = 50, db: Session = Depends(get_db)):
    """Rejected signals and how they worked out (the dual-ledger view)."""
    ghosts = (
        db.query(GhostTrade).order_by(GhostTrade.id.desc()).limit(limit).all()
    )
    return [
        {
            "id": g.id,
            "symbol": g.symbol,
            "reason": g.reason,
            "reference_price": str(g.reference_price),
            "max_gain_pct": str(g.max_gain_pct) if g.max_gain_pct is not None else None,
            "evaluated": g.evaluated,
            "created_at": g.created_at.isoformat() if g.created_at else None,
        }
        for g in ghosts
    ]


@app.get("/activity", response_model=list)
def get_activity(limit: int = 50):
    """Return the most recent activity-log entries."""
    return list(activity_log)[:limit]


@app.post("/toggle-mode", response_model=dict, dependencies=[Depends(require_api_key)])
def toggle_mode(payload: ToggleModeRequest, db: Session = Depends(get_db)):
    """Switch the system between PAPER and LIVE mode.

    Switching to LIVE trades real money and therefore additionally requires
    ``confirm: true`` in the request body.
    """
    mode_upper = payload.mode.upper()
    if mode_upper not in ("PAPER", "LIVE"):
        raise HTTPException(status_code=400, detail="mode must be 'PAPER' or 'LIVE'")

    is_live = mode_upper == "LIVE"
    if is_live and not payload.confirm:
        raise HTTPException(
            status_code=400,
            detail="Switching to LIVE places real orders with real money. "
            "Repeat the request with confirm=true to proceed.",
        )

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
        return {"is_live_mode": False, "is_halted": False, "last_sync_time": None}
    return {
        "is_live_mode": config.is_live_mode,
        "is_halted": config.is_halted,
        "peak_equity": str(config.peak_equity) if config.peak_equity is not None else None,
        "last_sync_time": config.last_sync_time,
    }


@app.patch("/config", response_model=dict, dependencies=[Depends(require_api_key)])
def update_config(payload: ConfigUpdate, db: Session = Depends(get_db)):
    """Update runtime config: Paper/Live mode and the death-protocol halt."""
    config = db.query(SystemConfig).first()
    if not config:
        config = SystemConfig(is_live_mode=False)
        db.add(config)
    if payload.is_live_mode is not None:
        config.is_live_mode = payload.is_live_mode
    if payload.is_halted is not None:
        config.is_halted = payload.is_halted
        if not payload.is_halted:
            # Re-arm with a fresh high-water mark so the protocol doesn't
            # immediately re-fire on the drawn-down equity.
            config.peak_equity = None
            _log_activity("Risk", "Death-protocol halt cleared by operator")
    db.commit()
    db.refresh(config)
    return {"is_live_mode": config.is_live_mode, "is_halted": config.is_halted}


@app.post("/order", response_model=dict, dependencies=[Depends(require_api_key)])
def place_order(payload: OrderRequest, db: Session = Depends(get_db)):
    """Place a paper or live order depending on the current mode.

    ``reference_price`` is required in paper mode unless the symbol is in the
    cached market data, in which case the last close is used.
    """
    symbol_base = symbols.to_base(payload.symbol)

    reference_price = payload.reference_price
    if reference_price is None:
        cached_df = _market_cache["data"].get(symbol_base)
        if cached_df is not None:
            reference_price = _last_close(cached_df)

    try:
        trade = shoonya.place_logic_order(
            symbol_base=symbol_base,
            quantity=payload.quantity,
            side=payload.side,
            db=db,
            reference_price=reference_price,
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
        "broker_order_id": trade.broker_order_id,
    }
