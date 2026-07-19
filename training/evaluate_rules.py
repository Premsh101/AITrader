"""
evaluate_rules.py – Grade the CLASSIC rule-based strategy on the identical
walk-forward window, costs, overlays and regime filter as the RL triad.

Usage:  python training/evaluate_rules.py [--smoke]
"""

from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training.common import build_dataset, download_universe  # noqa: E402
from training.evaluate_triad import run_backtest  # noqa: E402
from training.rule_strategy import RuleBrains  # noqa: E402
from app.stock_list import NSE_SYMBOLS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    ohlcv = download_universe(symbols=NSE_SYMBOLS[:5] if args.smoke else None)
    ds = build_dataset(ohlcv)
    m = run_backtest(ds, RuleBrains())

    print("\n──────── CLASSIC rule strategy · out-of-sample ────────")
    print(f"Window            : {m['window'][0]} → {m['window'][1]}")
    print(f"Total return      : {m['total_return']:+.2%}")
    print(f"Trades            : {m['trade_count']}")
    print(f"Mean trade return : {m['mean_trade_return']:+.3%}")
    print(f"Win rate          : {m['win_rate']:.1%}")
    print(f"Max drawdown      : {m['max_drawdown']:.2%}")
    print(f"^NSEI buy & hold  : {m['baseline_nifty']:+.2%}  (baseline)")
    if m["total_return"] > m["baseline_nifty"]:
        print("\nVERDICT: PASS – the rule strategy beats buy-and-hold after costs.")
    else:
        print("\nVERDICT: FAIL – does not beat buy-and-hold after costs.")


if __name__ == "__main__":
    main()
