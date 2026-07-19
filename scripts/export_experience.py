"""Export the system's decision history (real trades + ghost trades) as a
CSV of (symbol, date) experience points for training/train_triad.py
--experience. Run wherever DATABASE_URL points at the live DB:

    python scripts/export_experience.py > experience.csv
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database import SessionLocal  # noqa: E402
from app.models import GhostTrade, Trade  # noqa: E402

db = SessionLocal()
print("symbol,date,kind")
for t in db.query(Trade).all():
    if t.created_at:
        print(f"{t.symbol},{t.created_at.date()},trade")
for g in db.query(GhostTrade).all():
    if g.created_at:
        print(f"{g.symbol},{g.created_at.date()},ghost")
db.close()
