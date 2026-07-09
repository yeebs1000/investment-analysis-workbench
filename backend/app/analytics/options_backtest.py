"""Synthetic backtest for the options strategist.

No historical option chains exist for this account (verified live: Moomoo
serves real prices only for currently-listed contracts back to their listing
date; IBKR has no historical-options data permission at all). So this replays
the strategist over REAL historical EQUITY bars but with MODELED option
premiums: at each entry date it runs the same technical read and the same
`options.build_analysis` strategist the live app uses -- fed a SYNTHETIC chain
priced off Black-Scholes at an implied vol assumption -- then marks every
strategy it produced to the ACTUAL realized underlying price at expiry.

What this can and cannot show
-----------------------------
CAN: whether the strategist's *structure and strike selection* and its
predicted probability-of-profit are calibrated -- "when it says 65% POP, did
~65% actually win?" -- because entries, strikes, and the mark-to-expiry all run
through the same real logic on real price paths.

CANNOT: the IV-timing edge (sell-when-rich / buy-when-cheap). Entry IV is
modeled as trailing realized vol x (1 + vol-risk-premium), not a real quote, so
any edge that lives in *real IV vs realized* is assumed away. The IV *regime*
the strategist keys off still varies for real, though: build_analysis compares
this modeled IV against its own GARCH forward forecast, so realized-vol
momentum vs mean-reversion drives Elevated/Cheap/Normal from real data.

Marks are hold-to-expiry (European style); the strategist's 50%-profit / stop
management rules are NOT simulated -- that makes credit structures look worse
and debit structures look better than actively managed, a known conservative
bias on the premium-selling side. Tenor is measured in TRADING days for both
the BSM inputs and the exit bar, so it's internally consistent (a ~5% calendar
compression vs a real 35-DTE contract, second-order next to the modeled-IV
caveat above).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.analytics import options as options_engine, options_math, technical

# Synthetic-chain shape. Wide enough to contain the 15-delta wings the
# strategist reaches for even in a low-vol regime; fine enough that its
# select-by-delta lands close to the target.
STRIKE_SPAN = 0.45          # strikes span spot x [1-span, 1+span]
STRIKE_STEP = 0.02          # 2% spacing
SYNTH_OI = 5_000            # dummy open interest -> no spurious illiquidity veto
SYNTH_SPREAD_FRAC = 0.02    # modeled bid/ask = price x (1 +/- this)
DEFAULT_VRP = 0.05          # implied trades ~5% over realized on average (vol risk premium)
DEFAULT_HORIZON = 35        # tenor in trading days
DEFAULT_STEP = 10           # enter a new trade every N trading days (roll cadence)
MIN_TRAIL_BARS = 260        # need >= this trailing history (GARCH wants ~250)


def synth_chain(spot: float, iv_pct: float, dte: int) -> pd.DataFrame:
    """A Black-Scholes-priced option chain in the exact schema
    options.build_analysis consumes (right/strike/delta/iv/price/bid/ask/oi/
    code). One flat IV across strikes -- no synthetic skew, which is honest:
    we have no basis to invent a smile."""
    rate = options_math.RISK_FREE_RATE_PCT / 100.0
    lo, hi = spot * (1 - STRIKE_SPAN), spot * (1 + STRIKE_SPAN)
    strikes = np.round(np.arange(lo, hi, spot * STRIKE_STEP), 2)
    rows = []
    for right in ("CALL", "PUT"):
        for k in strikes:
            price = options_math.bsm_price(spot, k, iv_pct, dte, right, rate=rate)
            greeks = options_math.bsm_greeks(spot, k, iv_pct, dte, right, rate=rate)
            if price is None or greeks["delta"] is None:
                continue
            rows.append({
                "right": right.capitalize(),
                "strike": float(k),
                "delta": greeks["delta"],
                "iv": float(iv_pct),
                "price": round(price, 2),
                "bid": round(price * (1 - SYNTH_SPREAD_FRAC), 2),
                "ask": round(price * (1 + SYNTH_SPREAD_FRAC), 2),
                "oi": SYNTH_OI,
                "code": f"SYNTH.{right[0]}{k:.0f}",
            })
    return pd.DataFrame(rows)


@dataclass
class Trade:
    date: str
    symbol: str
    strategy: str
    direction: str
    decision: str
    pop_pct: float | None
    ev_per_share: float | None
    entry_spot: float
    exit_spot: float
    pnl_per_share: float
    win: bool


@dataclass
class StratStats:
    n: int = 0
    wins: int = 0
    pnl_sum: float = 0.0
    pop_sum: float = 0.0
    pop_n: int = 0

    def add(self, t: Trade) -> None:
        self.n += 1
        self.wins += 1 if t.win else 0
        self.pnl_sum += t.pnl_per_share
        if t.pop_pct is not None:
            self.pop_sum += t.pop_pct
            self.pop_n += 1

    @property
    def win_rate(self) -> float | None:
        return round(100.0 * self.wins / self.n, 1) if self.n else None

    @property
    def avg_pnl(self) -> float | None:
        return round(self.pnl_sum / self.n, 3) if self.n else None

    @property
    def predicted_pop(self) -> float | None:
        return round(self.pop_sum / self.pop_n, 1) if self.pop_n else None


def backtest_symbol(
    code: str, name: str, bars: pd.DataFrame, *,
    horizon: int = DEFAULT_HORIZON, step: int = DEFAULT_STEP, vrp: float = DEFAULT_VRP,
    ppy: float = 252.0,
) -> list[Trade]:
    """Walk `bars` forward, entering strategies every `step` trading days and
    marking them to the realized close `horizon` bars later. `bars` is an
    ascending OHLCV frame (normalize.bars_from_kline output)."""
    trades: list[Trade] = []
    n = len(bars)
    closes = bars["close"].to_numpy()
    idx = bars.index
    i = MIN_TRAIL_BARS
    while i + horizon < n:
        trail = bars.iloc[: i + 1]
        spot = float(closes[i])
        if not np.isfinite(spot) or spot <= 0:
            i += step
            continue
        ta = technical.analyze(code, name, trail, ppy=ppy)
        if ta.error or ta.decision is None:
            i += step
            continue
        rv = options_math.realized_vol_yang_zhang(trail)
        if not rv or rv <= 0:
            i += step
            continue
        iv = rv * (1.0 + vrp)
        chain = synth_chain(spot, iv, horizon)
        result = options_engine.build_analysis(
            code=code, name=name, as_of=str(idx[i].date()), spot=spot,
            decision=ta.decision, score=ta.score, bars=trail,
            contracts=chain, expiry=str(idx[i + horizon].date()), dte=horizon,
            holds=False, shares=0.0, confidence=ta.confidence,
        )
        exit_spot = float(closes[i + horizon])
        for s in result.strategies:
            if any(leg.price is None for leg in s.legs):
                continue
            pnl = float(options_math.payoff_at_expiry(s.legs, np.array([exit_spot]))[0])
            trades.append(Trade(
                date=str(idx[i].date()), symbol=code, strategy=s.name, direction=s.direction,
                decision=ta.decision.value, pop_pct=s.pop_pct, ev_per_share=s.ev_per_share,
                entry_spot=round(spot, 2), exit_spot=round(exit_spot, 2),
                pnl_per_share=round(pnl, 3), win=pnl > 0,
            ))
        i += step
    return trades


def aggregate(trades: list[Trade]) -> dict[str, StratStats]:
    stats: dict[str, StratStats] = {}
    for t in trades:
        stats.setdefault(t.strategy, StratStats()).add(t)
    stats.setdefault("__ALL__", StratStats())
    for t in trades:
        stats["__ALL__"].add(t)
    return stats


def format_report(stats: dict[str, StratStats], meta: dict) -> str:
    """Plain-text calibration report. The headline column is predicted-POP vs
    realized win-rate: if they track, the strategist's probability model is
    calibrated; a large persistent gap means it isn't."""
    lines = [
        "=" * 78,
        "OPTIONS STRATEGIST -- SYNTHETIC BACKTEST",
        "=" * 78,
        f"Symbols: {meta['n_symbols']}   Entries: every {meta['step']} trading days   "
        f"Tenor: {meta['horizon']} trading days   VRP: {meta['vrp']:.0%}",
        f"Modeled IV = trailing Yang-Zhang realized vol x (1+VRP); marks hold-to-expiry.",
        "-" * 78,
        f"{'Strategy':<26}{'n':>5}{'win%':>8}{'pred POP':>10}{'avg P&L/sh':>12}",
        "-" * 78,
    ]
    order = sorted((k for k in stats if k != "__ALL__"),
                   key=lambda k: stats[k].n, reverse=True)
    for k in order + ["__ALL__"]:
        s = stats[k]
        if not s.n:
            continue
        label = "ALL" if k == "__ALL__" else k
        pop = f"{s.predicted_pop:.1f}%" if s.predicted_pop is not None else "  -"
        lines.append(
            f"{label[:26]:<26}{s.n:>5}{s.win_rate:>7.1f}%{pop:>10}{s.avg_pnl:>12.3f}"
        )
    lines += [
        "-" * 78,
        "win% = fraction of entries with P&L > 0 at expiry on the REAL price path.",
        "pred POP = strategist's mean predicted probability of profit at entry.",
        "Caveat: entry premiums are MODELED (no historical chains exist), so this",
        "validates structure/strike/POP calibration, NOT the real IV-timing edge.",
        "=" * 78,
    ]
    return "\n".join(lines)


def run(symbols: list[str], *, horizon: int, step: int, vrp: float, lookback_days: int) -> str:
    """Fetch real Moomoo equity history for each symbol and backtest. Reuses the
    ML BarStore fetch path (same Moomoo throttle/cache as train.py)."""
    import time
    from app.ml import data_store
    from app.services.analysis_service import service

    KLINE_MIN_INTERVAL_S = 0.55  # Moomoo 60-calls/30s cap, same as train.py
    store = data_store.BarStore()
    all_trades: list[Trade] = []
    for i, code in enumerate(symbols, 1):
        if i > 1:
            time.sleep(KLINE_MIN_INTERVAL_S)
        try:
            bars = store.update(code, service._client, service._lock, lookback_days=lookback_days)
        except Exception as e:  # noqa: BLE001 - no quote permission etc.
            print(f"  [{i}/{len(symbols)}] {code}: SKIPPED -- {e}")
            continue
        if bars is None or len(bars) < MIN_TRAIL_BARS + horizon:
            print(f"  [{i}/{len(symbols)}] {code}: SKIPPED -- only {0 if bars is None else len(bars)} bars")
            continue
        t = backtest_symbol(code, code, bars, horizon=horizon, step=step, vrp=vrp)
        all_trades.extend(t)
        print(f"  [{i}/{len(symbols)}] {code}: {len(bars)} bars -> {len(t)} trades", flush=True)

    stats = aggregate(all_trades)
    return format_report(stats, {
        "n_symbols": len(symbols), "horizon": horizon, "step": step, "vrp": vrp,
    })


def main() -> None:
    import argparse
    from app.ml import universe

    ap = argparse.ArgumentParser(description="Synthetic backtest of the options strategist.")
    ap.add_argument("--symbols", default=None, help="comma-separated codes (e.g. US.AAPL,US.MSFT)")
    ap.add_argument("--universe", default=None,
                    help="holdings | sp500 | smallcap | <file> (ignored if --symbols given)")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="tenor in trading days")
    ap.add_argument("--step", type=int, default=DEFAULT_STEP, help="roll cadence in trading days")
    ap.add_argument("--vrp", type=float, default=DEFAULT_VRP, help="vol risk premium (IV = realized x (1+vrp))")
    ap.add_argument("--lookback-days", type=int, default=1095)
    ap.add_argument("--max-symbols", type=int, default=20)
    args = ap.parse_args()

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.universe:
        symbols = [c for c in universe.resolve_universe(source=args.universe)][: args.max_symbols]
    else:
        ap.error("pass --symbols or --universe")
    print(f"Backtesting {len(symbols)} symbol(s)...\n", flush=True)
    report = run(symbols, horizon=args.horizon, step=args.step, vrp=args.vrp,
                 lookback_days=args.lookback_days)
    print(report, flush=True)
    # The Moomoo SDK leaves non-daemon network threads running, so the process
    # won't exit on its own after the report is done -- force it, or the run
    # hangs forever with output already flushed. ponytail: os._exit skips
    # atexit/GC, fine for a one-shot CLI that has printed everything it needs.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
