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
# on the nearest tradeable strike instead of a fantasy one.
#
# 2026-07-17: a pure %-of-mid gate measures PREMIUM, not liquidity. Over 4,538
# two-sided legs it passed 76% of ATM legs but only 27% of 0.15-delta wings --
# median wing spread $0.41 on a $2.02 mid (20%) vs ATM $0.67 on a $12.65 mid
# (5%). The absolute cost of crossing is comparable (~$20-35/contract); the
# ratio just flatters expensive legs. Credit spreads need a cheap wing, so they
# were being starved: 1 Bull Put Spread survived out of 67 structures. A leg is
# therefore ALSO tradeable when its absolute spread is tight -- still capped by
# WIDE_SPREAD_PCT_CAP so far-OTM junk (median 60% spread) stays out.
# Measured effect: wing pass 26.8% -> 49.8%, far-OTM 10% -> 27%, overall 58% -> 66%.
# ponytail: per-leg gate. The economically exact test is structure-level slippage
# vs the credit earned; revisit if legs-per-structure grows past 2.
MAX_LEG_SPREAD_PCT = 10.0    # normal gate: spread as % of mid
WIDE_ABS_SPREAD = 0.50       # ...or a tight ABSOLUTE spread (~$25/contract to cross)
WIDE_SPREAD_PCT_CAP = 30.0   # ...but never a leg where crossing eats a third of it
MIN_OI = 25            # was 50; relaxed as a measured experiment
MIN_TRADEABLE_PER_SIDE = 3   # need a few strikes per right or skip the symbol


def leg_tradeable(bid, ask, oi) -> bool:
    """One leg's execution gate. Single source of truth: the signal prefilter
    and paper_trade's entry check both call this, so they cannot drift."""
    try:
        bid = float(bid); ask = float(ask); oi = float(oi)
    except (TypeError, ValueError):
        return False
    if bid != bid or ask != ask or oi != oi:      # NaN-safe
        return False
    if bid <= 0 or ask < bid or oi < MIN_OI:
        return False
    mid = (bid + ask) / 2.0
    if mid <= 0:
        return False
    spr = ask - bid
    pct = spr / mid * 100.0
    return pct <= MAX_LEG_SPREAD_PCT or (spr <= WIDE_ABS_SPREAD and pct <= WIDE_SPREAD_PCT_CAP)


def tradeable_chain(ch):
    """Subset of the recorded chain that passes the execution gates."""
    import pandas as pd
    df = ch.dropna(subset=["bid", "ask"]).copy()
    if df.empty:
        return df
    oi = pd.to_numeric(df["oi"], errors="coerce").fillna(0)
    keep = [leg_tradeable(b, a, o) for b, a, o in zip(df["bid"], df["ask"], oi)]
    return df[pd.Series(keep, index=df.index)]


def main() -> None:
    import numpy as np
    import pandas as pd
    from app.analytics import options as opt
    from app.analytics import options_math, technical
    from app.ml.data_store import BarStore

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    # session date comes from the ET clock, NOT max() over chain dirs: the max()
    # fallback silently processed YESTERDAY's chains when tonight's recording
    # failed -- three green stages, zero real output. An explicit arg still wins.
    from scripts._session import session_date
    day = args[0] if args else session_date()
    out_override = None
    if "--out" in sys.argv:
        out_override = Path(sys.argv[sys.argv.index("--out") + 1])
    chain_dir = CHAINS_DIR / day
    if not chain_dir.is_dir():
        print(f"no chain dir for {day} -- refusing to fall back to an older night")
        return 1
    SIGNALS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_override or (SIGNALS_DIR / f"{day}.csv")
    if out_path.exists():
        print(f"{out_path.name} already exists -- idempotent skip"); return 0

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
            # GARCH steps in TRADING days but dte is CALENDAR: precompute the
            # forecast at the trading-day equivalent so the IV regime read
            # matches the validated tenor (49 cal ~ 34 trading)
            fv = options_math.forecast_vol_garch(bars, max(1, int(round(dte * 252 / 365))))
            res = opt.build_analysis(
                code=code, name=code, as_of=day, spot=spot,
                decision=ta.decision, score=ta.score, bars=bars,
                contracts=trade_ch, expiry=expiry, dte=dte, holds=False, shares=0.0,
                confidence=ta.confidence, market_regime=market_regime,
                forecast_vol_pct=fv if fv is not None else -1.0)
            snap = ch.set_index("code")
            sym_rows = []   # buffer per symbol: partial structures must never reach the CSV
            for s in res.strategies:
                for leg in s.legs:
                    q = snap.loc[leg.code] if leg.code in snap.index else None
                    # rate matters: at r=0 the logged theo manufactured a fake
                    # -4.9% call / +3.6% put "mispricing" that was entirely the
                    # omitted risk-free rate (diagnostic column only; EV/POP
                    # never consume theo_bsm)
                    theo = options_math.bsm_price(
                        spot, leg.strike, leg.iv_pct, dte, leg.right,
                        rate=options_math.RISK_FREE_RATE_PCT / 100.0)
                    sym_rows.append({
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
            rows.extend(sym_rows)   # only complete symbols reach the CSV
            print(f"  {code}: {len(res.strategies)} structure(s)", flush=True)
        except Exception as e:  # noqa: BLE001 - one bad symbol never kills the log
            print(f"  {f.stem}: SKIP ({e})", flush=True)

    # atomic write: a kill mid-write must not leave a partial CSV that the
    # exists-skip above would then lock in forever
    df = pd.DataFrame(rows)
    tmp = out_path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_path)
    # per-run archive so a later, smaller re-scan can never erase the file the
    # entry engine actually traded from (28% of live entries were unjoinable)
    arch = SIGNALS_DIR / "archive"
    arch.mkdir(exist_ok=True)
    df.to_csv(arch / f"{day}-{dt.datetime.now():%H%M%S}.csv", index=False)
    print(f"logged {len(rows)} leg rows -> {out_path}", flush=True)
    # honest exit: zero rows from a full chain dir is a broken night, not a green one
    return 1 if (len(rows) == 0 and len(files) > 0) else 0


if __name__ == "__main__":
    from scripts._lock import single_instance
    rc = 0
    with single_instance("log_signals") as got:
        if got:
            rc = main() or 0
        else:
            print("another log_signals instance holds the lock -- exiting")
    sys.stdout.flush()
    os._exit(rc)
