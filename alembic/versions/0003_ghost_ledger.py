"""Ghost ledger table + trade charges column (idempotent).

Revision ID: 0003_ghost_ledger
Revises: 0002_phase1_risk
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_ghost_ledger"
down_revision = "0002_phase1_risk"
branch_labels = None
depends_on = None


def upgrade() -> None:
    inspector = sa.inspect(op.get_bind())

    if not inspector.has_table("ghost_trades"):
        op.create_table(
            "ghost_trades",
            sa.Column("id", sa.Integer(), primary_key=True, index=True),
            sa.Column("symbol", sa.String(50), nullable=False, index=True),
            sa.Column("reference_price", sa.Numeric(18, 4), nullable=False),
            sa.Column("reason", sa.String(30), nullable=False),
            sa.Column("max_gain_pct", sa.Numeric(9, 4), nullable=True),
            sa.Column("evaluated", sa.Boolean(), nullable=False, server_default=sa.false()),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )

    trades = {c["name"] for c in inspector.get_columns("trades")}
    if "charges" not in trades:
        op.add_column("trades", sa.Column("charges", sa.Numeric(18, 4), nullable=True))


def downgrade() -> None:
    op.drop_table("ghost_trades")
    op.drop_column("trades", "charges")
