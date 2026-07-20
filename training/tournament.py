"""
tournament.py – Validate MULTIPLE trained strategy variants and keep the best.

Pairs with ``train_triad.py --seeds N``: each seed is an independently
trained triad (different random initialisation → different learned
strategy).  Every variant is backtested on the SAME held-out window with
real costs; the winner's zips are promoted to the canonical filenames —
and it must still beat buy-and-hold ^NSEI to be deployable.

Each variant is additionally swept across Executive approve thresholds
(``--thresholds``): the SAME zips are re-graded with a stricter entry bar,
which costs minutes instead of a retraining run.  The winner is the best
(variant, threshold) pair; deploy that threshold via the
``EXECUTIVE_APPROVE_THRESHOLD`` environment variable.

Usage:
  python training/train_triad.py --seeds 3 --timesteps 1000000
  python training/tournament.py --seeds 3
  python training/tournament.py --seeds 3 --thresholds 0.5,0.65,0.75
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.common import EXECUTIVE_FILE, GUARDIAN_FILE, HUNTER_FILE, build_dataset, download_universe  # noqa: E402
from training.evaluate_triad import run_backtest  # noqa: E402
from app.stock_list import NSE_SYMBOLS  # noqa: E402

FILES = [HUNTER_FILE, GUARDIAN_FILE, EXECUTIVE_FILE]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--models-dir", default=os.path.join(REPO_ROOT, "models"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--thresholds",
        default="0.50,0.60,0.65,0.70",
        help="Comma-separated Executive approve thresholds to sweep per variant "
             "(re-grades the same zips with stricter entry bars; no retraining).",
    )
    args = parser.parse_args()

    thresholds = sorted({float(t) for t in args.thresholds.split(",") if t.strip()})

    import app.ai_brains as ai_brains

    ohlcv = download_universe(symbols=NSE_SYMBOLS[:5] if args.smoke else None)
    ds = build_dataset(ohlcv)

    results = []
    for variant in range(args.seeds):
        staging = tempfile.mkdtemp(prefix=f"triad_s{variant}_")
        try:
            for f in FILES:
                shutil.copy2(
                    os.path.join(args.models_dir, f.replace(".zip", f"_s{variant}.zip")),
                    os.path.join(staging, f),
                )
        except FileNotFoundError as exc:
            print(f"variant s{variant}: missing file ({exc}) – skipped")
            continue
        ai_brains.MODELS_DIR = staging
        brains = ai_brains.BrainManager()
        brains.load_all()
        if not brains.all_ready:
            print(f"variant s{variant}: failed load assertions – skipped")
            continue
        for thr in thresholds:
            # select_slots reads the module-level constant on every call, so
            # patching it re-grades the same loaded models at a stricter bar.
            ai_brains.EXECUTIVE_APPROVE_THRESHOLD = thr
            m = run_backtest(ds, brains)
            results.append((m["total_return"], variant, thr, m))
            print(
                f"variant s{variant} @ thr {thr:.2f}: return {m['total_return']:+.2%}, "
                f"{m['trade_count']} trades, win {m['win_rate']:.0%}, "
                f"maxDD {m['max_drawdown']:.1%}"
            )

    if not results:
        raise SystemExit("No variant produced a valid backtest.")

    results.sort(key=lambda r: r[0], reverse=True)
    best_return, best, best_thr, metrics = results[0]
    print(f"\nWINNER: variant s{best} @ threshold {best_thr:.2f} "
          f"({best_return:+.2%} vs baseline {metrics['baseline_nifty']:+.2%})")

    for f in FILES:
        shutil.copy2(
            os.path.join(args.models_dir, f.replace(".zip", f"_s{best}.zip")),
            os.path.join(args.models_dir, f),
        )
    if best_return > metrics["baseline_nifty"]:
        print("VERDICT: PASS – winner beats buy-and-hold after costs; canonical "
              "zips updated, OK to commit models/.")
        print(f"Deploy with EXECUTIVE_APPROVE_THRESHOLD={best_thr:.2f} in .env "
              "so serving uses the same entry bar the winner was graded at.")
    else:
        print("VERDICT: FAIL – even the best variant does not beat buy-and-hold. "
              "Do not deploy; iterate on rewards.")


if __name__ == "__main__":
    main()
