"""Daily signal log: what would the strategist trade TODAY, at what modeled
price, against what real quotes? (The paper-fill gate's raw material.)

Runs OFFLINE on the day's recorded chains (scripts/record_chains.py) plus
cached bars -- no broker calls, no quota. For every structure the strategist
offers, logs one row per leg with the real bid/ask/mid alongside a
theoretical BSM price at the leg's own IV, plus POP/EV and the structure's
economics. Accumulates data_store/signals/<date>.csv; over months this is
the modeled-vs-quoted comparison the synthetic backtest cannot provide.

Usage: python scripts/log_signals.py [YYYY-MM-DD]   (default: latest chain dir)
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

CHAINS_DIR = BACKEND / "data_store" / "chains"
SIGNALS_DIR = BACKEND / "data_store" / "signals"

# --- tradeability pre-filter -------------------------------------------------
# Day-1 live finding (2026-07-13): the strategist picked strikes by delta from
# the FULL chain, then 53/87 structures failed the execution gates (leg spread
# <= 10% of mid, OI >= 50). Filtering the chain BEFORE strike selection makes
# every proposed structure executable by construction -- the strategist lands
# on the nearest tradeable strike instead of a fantasy one. Thresholds match
# scripts/paper_trade.py exactly; keep them in lockstep.
MAX_LEG_SPREAD_PCT = 10.0
MIN_OI = 25            # was 50; relaxed as a measured experiment (spread gate unchanged)
MIN_TRADEABLE_PER_SIDE = 3   # need a few strikes per right or skip the symbol


def tradeable_chain(ch):
    """Subset of the recorded chain that passes the execution gates."""
    import pandas as pd
    df = ch.dropna(subset=["bid", "ask"]).copy()
    df = df[(df["bid"] > 0) & (df["ask"] >= df["bid"])]
    if df.empty:
        return df
    mid = (df["bid"] + df["ask"]) / 2.0
    df = df[(df["ask"] - df["bid"]) / mid * 100.0 <= MAX_LEG_SPREAD_PCT]
    df = df[pd.to_numeric(df["oi"], errors="coerce").fillna(0) >= MIN_OI]
    return df


def main() -> None:
    import numpy as np
    import pandas as pd
    from app.analytics import options as opt
    from app.analytics import options_math, technical
    from app.ml.data_store import BarStore

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    day = args[0] if args else max(p.name for p in CHAINS_DIR.iterdir() if p.is_dir())
    out_override = None
    if "--out" in sys.argv:
        out_override = Path(sys.argv[sys.argv.index("--out") + 1])
    chain_dir = CHAINS_DIR / day
    if not chain_dir.is_dir():
        print(f"no chain dir for {day}"); return
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_override or (SIGNALS_DIR / f"{day}.csv")
    if out_path.exists():
        print(f"{out_path.name} already exists -- idempotent skip"); return

    store = BarStore()
    bench = store.load("US.SPY")
    market_regime = opt.benchmark_regime(bench["close"]) if not bench.empty else None

    rows = []
    files = sorted(chain_dir.glob("*.parquet"))
    print(f"{day}: {len(files)} chains, regime={market_regime}", flush=True)
    for f in files:
        try:
            ch = pd.read_parquet(f)
            code = str(ch["underlying"].iloc[0])
            spot = float(ch["spot"].iloc[0])
            expiry = str(ch["expiry"].iloc[0])
            dte = int(ch["dte"].iloc[0])
            trade_ch = tradeable_chain(ch)
            per_side = trade_ch["right"].str.upper().value_counts() if not trade_ch.empty else {}
            if trade_ch.empty or min(per_side.get("CALL", 0), per_side.get("PUT", 0)) < MIN_TRADEABLE_PER_SIDE:
                raise ValueError(f"only {len(trade_ch)} tradeable strikes (need >={MIN_TRADEABLE_PER_SIDE}/side)")
            bars = store.load(code)
            if bars.empty or len(bars) < 260:
                raise ValueError("insufficient bars")
            ta = technical.analyze(code, code, bars)
            if ta.error or ta.decision is None:
                raise ValueError(f"technical: {ta.error or 'no decision'}")
            res = opt.build_analysis(
                code=code, name=code, as_of=day, spot=spot,
                decision=ta.decision, score=ta.score, bars=bars,
                contracts=trade_ch, expiry=expiry, dte=dte, holds=False, shares=0.0,
                confidence=ta.confidence, market_regime=market_regime)
            snap = ch.set_index("code")
            for s in res.strategies:
                for leg in s.legs:
                    q = snap.loc[leg.code] if leg.code in snap.index else None
                    theo = options_math.bsm_price(spot, leg.strike, leg.iv_pct, dte, leg.right)
                    rows.append({
                        "date": day, "code": code, "spot": spot, "expiry": expiry, "dte": dte,
                        "decision": ta.decision.value, "confidence": ta.confidence,
                        "strategy": s.name, "direction": s.direction,
                        "pop_pct": s.pop_pct, "ev_per_share": s.ev_per_share,
                        "net_debit_credit": s.net_debit_credit, "max_loss": s.max_loss,
                        "leg_action": leg.action, "leg_right": leg.right, "leg_strike": leg.strike,
                        "leg_code": leg.code, "leg_iv": leg.iv_pct, "leg_delta": leg.delta,
                        "bid": (float(q["bid"]) if q is not None and pd.notna(q["bid"]) else None),
                        "ask": (float(q["ask"]) if q is not None and pd.notna(q["ask"]) else None),
                        "mid_used": leg.price, "theo_bsm": theo,
                        "oi": (float(q["oi"]) if q is not None and pd.notna(q["oi"]) else None),
                        "n_tradeable": len(trade_ch), "n_chain": len(ch),
                    })
            print(f"  {code}: {len(res.strategies)} structure(s)", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad symbol never kills the log
            print(f"  {f.stem}: SKIP ({e})", flush=True)

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"logged {len(rows)} leg rows -> {out_path}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
