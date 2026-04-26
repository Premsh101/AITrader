"""
database.py – SQLAlchemy 2.0 engine and session setup.

The database connection string is pulled from the DATABASE_URL environment
variable so that it can be injected at runtime (e.g. via Docker Compose or a
Coolify secret).
"""

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

DATABASE_URL: str = os.environ.get(
    "DATABASE_URL",
    "postgresql://trader:trader_pass@db:5432/aitrader",
)

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,      # Automatically reconnect after database restarts
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    class_=Session,
)


class Base(DeclarativeBase):
    """Shared declarative base for all SQLAlchemy models."""


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a database session and ensures cleanup."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
