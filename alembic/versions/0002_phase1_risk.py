"""Phase 1 risk columns: trade peak/exit tracking, equity HWM, halt flag.

Idempotent like 0001: each column is added only if missing.

Revision ID: 0002_phase1_risk
Revises: 0001_initial
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0002_phase1_risk"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def _columns(table: str) -> set[str]:
    return {c["name"] for c in sa.inspect(op.get_bind()).get_columns(table)}


def upgrade() -> None:
    trades = _columns("trades")
    if "peak_price" not in trades:
        op.add_column("trades", sa.Column("peak_price", sa.Numeric(18, 4), nullable=True))
    if "exit_reason" not in trades:
        op.add_column("trades", sa.Column("exit_reason", sa.String(30), nullable=True))

    config = _columns("system_config")
    if "peak_equity" not in config:
        op.add_column("system_config", sa.Column("peak_equity", sa.Numeric(18, 2), nullable=True))
    if "is_halted" not in config:
        op.add_column(
            "system_config",
            sa.Column("is_halted", sa.Boolean(), nullable=False, server_default=sa.false()),
        )


def downgrade() -> None:
    op.drop_column("trades", "peak_price")
    op.drop_column("trades", "exit_reason")
    op.drop_column("system_config", "peak_equity")
    op.drop_column("system_config", "is_halted")
