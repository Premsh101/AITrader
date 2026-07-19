"""
models.py – SQLAlchemy 2.0 ORM models for AITrader.

Tables:
  - trades        : Individual trade records (paper and live).
  - system_config : Global runtime configuration (e.g. live/paper toggle).
"""

import enum
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Enum, Float, Integer, Numeric, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class TradeStatus(str, enum.Enum):
    OPEN = "Open"
    CLOSED = "Closed"


class TradeMode(str, enum.Enum):
    PAPER = "Paper"
    LIVE = "Live"


class Trade(Base):
    """Represents a single trade entry in the system."""

    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    buy_price: Mapped[float | None] = mapped_column(Numeric(18, 4), nullable=True)
    sell_price: Mapped[float | None] = mapped_column(Numeric(18, 4), nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[TradeStatus] = mapped_column(
        Enum(TradeStatus), nullable=False, default=TradeStatus.OPEN
    )
    mode: Mapped[TradeMode] = mapped_column(
        Enum(TradeMode), nullable=False, default=TradeMode.PAPER
    )
    pnl: Mapped[float | None] = mapped_column(Numeric(18, 4), nullable=True)
    broker_order_id: Mapped[str | None] = mapped_column(String(50), nullable=True)
    # Highest close observed since entry; drives the profit-ladder overlay.
    peak_price: Mapped[float | None] = mapped_column(Numeric(18, 4), nullable=True)
    # Estimated round-trip charges (₹) deducted from pnl at close.
    charges: Mapped[float | None] = mapped_column(Numeric(18, 4), nullable=True)
    # Why the position was closed: "guardian", "stop-loss", "time-exit",
    # "profit-trail", "breakeven-stop", "death-protocol", "manual".
    exit_reason: Mapped[str | None] = mapped_column(String(30), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return (
            f"<Trade id={self.id} symbol={self.symbol!r} "
            f"status={self.status} mode={self.mode}>"
        )


class GhostTrade(Base):
    """A Hunter signal that was NOT taken, tracked as a missed opportunity.

    The original design's dual-ledger: comparing ghost outcomes with real
    trades shows whether the rejection gates are too strict and gives the
    Executive counterfactual training data.
    """

    __tablename__ = "ghost_trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    symbol: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    reference_price: Mapped[float] = mapped_column(Numeric(18, 4), nullable=False)
    # Why it was rejected: "executive-skip", "illiquid", "regime-block",
    # "vix-block", "cutoff", "no-slots".
    reason: Mapped[str] = mapped_column(String(30), nullable=False)
    # Filled in ~5 bars later: best close since the signal, as a fraction.
    max_gain_pct: Mapped[float | None] = mapped_column(Numeric(9, 4), nullable=True)
    evaluated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<GhostTrade id={self.id} symbol={self.symbol!r} reason={self.reason}>"


class SystemConfig(Base):
    """Global runtime configuration stored in the database."""

    __tablename__ = "system_config"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    is_live_mode: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )
    last_sync_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    # High-water mark of tracked equity; feeds the 25%-drawdown death rule.
    peak_equity: Mapped[float | None] = mapped_column(Numeric(18, 2), nullable=True)
    # Set by the death protocol; while True the trading loop refuses to run.
    is_halted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    def __repr__(self) -> str:
        return (
            f"<SystemConfig id={self.id} is_live_mode={self.is_live_mode} "
            f"last_sync_time={self.last_sync_time}>"
        )
