# AITrader — Vision vs. Reality, and the Road Forward

Source: the original Gemini design conversation ("AI Survival Trading With
5000 INR", 154 pages, `AI Survival Trading With 5000 INR.pdf`), read in full
and reconciled against the current codebase (post repair PR #13).

## 1. The original vision (as agreed in that chat)

- An AI that "earns to survive": ₹5,000 starting capital, a **25% drawdown
  from peak equity = death** (flatten everything, halt, send a farewell
  report).
- Goal escalated from 10%/month to **10%/week** (portfolio-level acceptable).
  Gemini's own "Truth Audit" rated the 10%-weekly outcome at **15%
  confidence** and "beats a savings account / FD" at 70% — expectations
  should anchor to the latter.
- **The Triad** (user-defined): Hunter finds entries, Guardian manages
  exits, Executive allocates a **5-slot (20% each) "bullet" portfolio**.
- **Dual ledger**: every Executive-rejected signal recorded as a **ghost
  trade** (missed opportunity) so the Executive can learn counterfactually.
- Risk overlays: hard stop-loss −5%, 20-day max hold, profit-protection
  ladder (+5% → break-even stop; +10% → trail at +8%), India-VIX spike
  filter (>15%/day → no new entries), news kill-switch (deferred).
- Data ambitions: 1,500-stock universe, delivery %, option chain (PCR),
  bulk/block deals, India VIX; charges simulated so "10%" means net.
- Ops: paper trade ~1 month with real data before live; automated weekend
  retraining; merge-to-main auto-deploys (Coolify); Shoonya TOTP login.

## 2. The models that were actually shipped (recipes now RECOVERED)

The chat contains the exact observation definitions of the deployed
`apex_1500` models — previously believed lost:

| Brain | Obs | Features (exact order) | Actions | Steps |
|---|---|---|---|---|
| Hunter | (5,) | RSI/100 (NaN→0.5), MACD(12,26,9) raw, VWAP_Dist=(C−VWMA20)/VWMA20, ATR(14)/C, OBV.pct_change() | Discrete(2) buy/no | ~5M |
| Guardian | (5,) | profit_pct, days_held/20, RSI/100, MACD_slope, ATR(14)/C | Discrete(2) hold/sell | ~5M |
| Executive | (3,) | free_slots/5, avg 14d volatility, portfolio profit% | Discrete(3) accept 0/1/≤3 | ~1M |

Caveats visible in the transcript (why revival is doubtful even with the
recipes):

1. **No trading costs anywhere in the rewards** (10% target is gross).
2. **Guardian degenerate policy**: final ep_len_mean 19.99 of a 20-bar cap —
   it learned to almost never sell before the forced exit.
3. **Executive semantic mismatch**: it was trained to answer "how many of
   today's signals to accept" (0/1/3) — the serving app used its action as a
   per-symbol priority score, which was never its meaning. Its training also
   never queried the real Hunter/Guardian (random signals, hard-coded exits)
   and its profit feature used the wrong symbol's price.
4. **Never validated out-of-sample** — the backtest was proposed in the chat
   but never executed.

**Decision approach: settle it with data, not argument.** Phase 0 includes a
legacy adapter that computes the 5-dim/3-dim observations and runs the OLD
zips through the same walk-forward, cost-aware evaluator as the new models.
If they beat buy-and-hold ^NSEI after costs, they earn their place; if not,
the retrained triad replaces them with evidence in hand.

## 3. Where we are today (post repair PR)

✅ Built and working: Triad serving architecture; 5-slot cap; Paper/Live
toggle with confirmation; dashboard (stats, activity, trades); Postgres +
idempotent migrations; Coolify-compatible deploy; market-hours loop with
fetch caching; dedup of held symbols; API-key security + CORS + loopback
ports; obs-dim load assertions (mismatch can never be silent again);
cost-aware retraining pipeline with walk-forward evaluation gate; tests.

🟡 Partial vs vision: universe 145 vs 1,500; fixed ₹10k/slot vs 20%-of-equity
sizing; Executive obs is market-features+context, not the designed
(hunter-confidence, guardian-risk, drawdown, sector-win-rate) state; app P&L
is gross of charges; retraining is manual on Kaggle.

❌ Missing vs vision: death protocol (HWM −25% kill switch); ghost ledger;
hard stop-loss / profit ladder / time-exit overlays (exits rely purely on
the Guardian model); VIX filter; delivery %/option-chain/bulk-deal data;
weekly-goal tracking ("Wednesday rule"); alerts (Telegram/email); legacy
backtest.

## 4. The plan

### Phase 0 — Models (unblocks everything; already in motion)
1. Train the new triad on Kaggle (smoke → full 5M/5M/1M).
2. **Legacy adapter + backtest**: compute the recovered 5-dim/3-dim obs in
   the evaluator and score the old `apex_1500` zips on the same held-out
   window. Deploy whichever generation passes the gate; document the loser.

### Phase 1 — Deterministic safety net (small, high value; no ML)
3. Hard risk overlays in the trading loop, independent of the Guardian:
   stop-loss −5%, max hold 20 bars, profit ladder (+5% → breakeven,
   +10% → trail +8%).
4. **Death protocol**: track peak equity (HWM); if equity < 75% of peak →
   close all positions, set system to halted, ERROR activity + notification.
5. VIX filter: ^INDIAVIX daily spike >15% → no new entries that day.

### Phase 2 — Fidelity to the original design
6. Ghost ledger: table + recording of Hunter signals the Executive rejects,
   with 5-day forward outcome backfill; dashboard "missed opportunities"
   panel; feeds future Executive retraining.
7. Net-of-charges P&L: store charges per trade (STT/exchange/SEBI/stamp/GST
   approximation ≈ the training COST_BPS) so dashboard P&L is net.
8. Dynamic sizing: slot capital = equity/5 (configurable), replacing the
   fixed per-slot rupee amount.
9. Universe expansion toward the EQ-series master list (batched, cached
   fetching; start 300–500, then scale as fetch reliability allows).

### Phase 3 — Executive as designed
10. Retrain the Executive on the intended state: Hunter approve-probability,
    Guardian risk score, portfolio drawdown, slot occupancy, market context —
    trained against the real Hunter/Guardian (not random signals).
    Configurable confidence threshold (default 0.5, original vision 0.85).

### Phase 4 — Data enrichment & ops
11. Delivery % via nselib from the VPS (NSE blocks Kaggle IPs; an Indian VPS
    may pass) and option-chain/PCR via Shoonya once logged in — as Hunter
    features in the NEXT training generation.
12. Automated weekend retraining job + model hot-reload endpoint.
13. Alerts (Telegram or email) for entries, exits, death protocol, errors.

### Go-live rule (unchanged from the vision)
Paper trade ≥ 1 month; deploy real money only if the evaluator PASSes and
paper results stay healthy; start with capital you can afford to lose
entirely ("tuition fees", per the original chat's own truth audit).
