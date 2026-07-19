"""Initial schema: trades (incl. broker_order_id) and system_config.

This migration is written idempotently because existing deployments created
the tables outside Alembic (there was previously no versions/ directory):

  • If a table does not exist, it is created in full.
  • If ``trades`` exists but predates ``broker_order_id``, the column is added.

Fresh deployments and existing databases therefore both converge on the same
schema with a plain ``alembic upgrade head`` – no manual ``alembic stamp``
required.

Revision ID: 0001_initial
Revises:
Create Date: 2026-07-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def _inspector() -> sa.Inspector:
    return sa.inspect(op.get_bind())


def upgrade() -> None:
    inspector = _inspector()

    trade_status = sa.Enum("OPEN", "CLOSED", name="tradestatus")
    trade_mode = sa.Enum("PAPER", "LIVE", name="trademode")

    if not inspector.has_table("trades"):
        op.create_table(
            "trades",
            sa.Column("id", sa.Integer(), primary_key=True, index=True),
            sa.Column("symbol", sa.String(50), nullable=False, index=True),
            sa.Column("buy_price", sa.Numeric(18, 4), nullable=True),
            sa.Column("sell_price", sa.Numeric(18, 4), nullable=True),
            sa.Column("quantity", sa.Integer(), nullable=False),
            sa.Column("status", trade_status, nullable=False),
            sa.Column("mode", trade_mode, nullable=False),
            sa.Column("pnl", sa.Numeric(18, 4), nullable=True),
            sa.Column("broker_order_id", sa.String(50), nullable=True),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
            sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        )
    else:
        columns = {col["name"] for col in inspector.get_columns("trades")}
        if "broker_order_id" not in columns:
            op.add_column(
                "trades",
                sa.Column("broker_order_id", sa.String(50), nullable=True),
            )

    if not inspector.has_table("system_config"):
        op.create_table(
            "system_config",
            sa.Column("id", sa.Integer(), primary_key=True, index=True),
            sa.Column(
                "is_live_mode",
                sa.Boolean(),
                nullable=False,
                server_default=sa.false(),
            ),
            sa.Column("last_sync_time", sa.DateTime(timezone=True), nullable=True),
        )


def downgrade() -> None:
    op.drop_table("trades")
    op.drop_table("system_config")
    sa.Enum(name="tradestatus").drop(op.get_bind(), checkfirst=True)
    sa.Enum(name="trademode").drop(op.get_bind(), checkfirst=True)
