"""
main.py – FastAPI application entry point for AITrader.
"""

import logging

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import SystemConfig, Trade, TradeMode, TradeStatus
from app.shoonya_service import ShoonyaEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

app = FastAPI(
    title="AITrader",
    description="Production-grade AI trading bot with Paper/Live toggle.",
    version="1.0.0",
)

shoonya = ShoonyaEngine()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class OrderRequest(BaseModel):
    symbol: str
    quantity: int
    side: str  # "BUY" or "SELL"


class LiveModeUpdate(BaseModel):
    is_live_mode: bool


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health")
def health_check():
    return {"status": "ok"}


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
        }
        for t in trades
    ]
