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

    def __repr__(self) -> str:
        return (
            f"<SystemConfig id={self.id} is_live_mode={self.is_live_mode} "
            f"last_sync_time={self.last_sync_time}>"
        )
