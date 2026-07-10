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
GARCH_REFIT_EVERY = 3       # backtest: refit GARCH every N entries, reuse between (see backtest_symbol)


@dataclass(frozen=True)
class ManageRule:
    """A per-structure exit rule for the early-management overlay. `profit_target`
    closes when unrealized P&L reaches that fraction of max profit; `stop_to_be`
    arms a breakeven stop once P&L reaches that fraction (never let a decent
    winner round-trip to a loss). None/None = hold to expiry."""
    profit_target: float | None = None
    stop_to_be: float | None = None
    label: str = "hold to expiry"


# Not one-size-fits-all (the user's point): premium SELLERS decay reliably and
# carry short-gamma tail risk into expiry, so take profit early; premium BUYERS
# need the move to keep going, so ride but protect gains with a breakeven stop.
# These thresholds are HYPOTHESES the backtest tests, not received wisdom.
_CREDIT_RULE = ManageRule(profit_target=0.70, stop_to_be=None, label="take 70% of max credit")
_DEBIT_RULE = ManageRule(profit_target=0.85, stop_to_be=0.50, label="ride; BE-stop once +50%")
MANAGEMENT: dict[str, ManageRule] = {
    "Bull Put Spread (credit)": _CREDIT_RULE,
    "Bear Call Spread (credit)": _CREDIT_RULE,
    "Iron Condor": _CREDIT_RULE,
    "Cash-Secured Put": _CREDIT_RULE,
    "Covered Call": _CREDIT_RULE,
    "Call Debit Spread": _DEBIT_RULE,
    "Put Debit Spread": _DEBIT_RULE,
    "Long Straddle": ManageRule(profit_target=1.0, stop_to_be=0.5, label="ride; BE-stop once +50%"),
    "Collar": ManageRule(label="hold (protective)"),   # a hedge, not a P&L trade
}
DEFAULT_RULE = ManageRule()


def managed_exit(strategy, entry_spot: float, path_spots: np.ndarray, horizon: int):
    """Walk the realized daily close path (entry+1 .. expiry) repricing the
    position at constant entry IV -- spot moves (real) and theta decay drive the
    mark; IV drift isn't modeled (conservative for sellers, who'd also bank vol
    crush). Returns (pnl_per_share, exit_reason, days_held) applying the
    per-structure ManageRule. With no rule triggered it equals hold-to-expiry."""
    rule = MANAGEMENT.get(strategy.name, DEFAULT_RULE)
    dtes = horizon - np.arange(1, len(path_spots) + 1)   # remaining trading-day tenor
    pnl_path = np.zeros(len(path_spots))
    for leg in strategy.legs:
        p0 = leg.price or 0.0
        prices = options_math.bsm_price_path(path_spots, leg.strike, leg.iv_pct or 0.0, dtes, leg.right)
        pnl_path += (prices - p0) if leg.action == "Buy" else (p0 - prices)

    base = strategy.max_profit
    if base is None and (strategy.net_debit_credit or 0) > 0:
        base = strategy.net_debit_credit          # credit structures: max profit = credit
    if rule.profit_target is None and rule.stop_to_be is None or not base or base <= 0:
        return float(pnl_path[-1]), "expiry", len(path_spots)

    armed = False
    for k, pnl in enumerate(pnl_path, start=1):
        if rule.profit_target is not None and pnl >= rule.profit_target * base:
            return float(pnl), "profit_target", k
        if rule.stop_to_be is not None:
            if pnl >= rule.stop_to_be * base:
                armed = True
            elif armed and pnl <= 0.0:
                return 0.0, "stop_be", k          # exit ~breakeven, protecting the prior gain
    return float(pnl_path[-1]), "expiry", len(path_spots)


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
    regime: str | None = None   # market regime at entry ("bull"/"bear"/None)
    managed_pnl: float | None = None    # P&L under the early-management overlay
    exit_reason: str | None = None      # profit_target | stop_be | expiry
    days_held: int | None = None
    # capital at risk per share (the structure's max loss at entry). P&L/share is
    # not comparable across a $60 stock and a $1000 one; P&L / risk is.
    risk_per_share: float | None = None


@dataclass
class StratStats:
    n: int = 0
    wins: int = 0
    pnl_sum: float = 0.0
    pop_sum: float = 0.0
    pop_n: int = 0
    m_wins: int = 0            # managed-overlay wins
    m_pnl_sum: float = 0.0     # managed-overlay P&L
    held_sum: int = 0          # trading days held under management
    early_closes: int = 0      # trades exited before expiry by the overlay
    ror_sum: float = 0.0       # sum of per-trade P&L / capital-at-risk
    ror_n: int = 0

    def add(self, t: Trade) -> None:
        self.n += 1
        self.wins += 1 if t.win else 0
        self.pnl_sum += t.pnl_per_share
        if t.pop_pct is not None:
            self.pop_sum += t.pop_pct
            self.pop_n += 1
        if t.risk_per_share and t.risk_per_share > 0:
            self.ror_sum += t.pnl_per_share / t.risk_per_share
            self.ror_n += 1
        if t.managed_pnl is not None:
            self.m_pnl_sum += t.managed_pnl
            self.m_wins += 1 if t.managed_pnl > 0 else 0
            self.held_sum += t.days_held or 0
            self.early_closes += 1 if t.exit_reason in ("profit_target", "stop_be") else 0

    @property
    def win_rate(self) -> float | None:
        return round(100.0 * self.wins / self.n, 1) if self.n else None

    @property
    def avg_pnl(self) -> float | None:
        return round(self.pnl_sum / self.n, 3) if self.n else None

    @property
    def predicted_pop(self) -> float | None:
        return round(self.pop_sum / self.pop_n, 1) if self.pop_n else None

    @property
    def m_win_rate(self) -> float | None:
        return round(100.0 * self.m_wins / self.n, 1) if self.n else None

    @property
    def m_avg_pnl(self) -> float | None:
        return round(self.m_pnl_sum / self.n, 3) if self.n else None

    @property
    def avg_days_held(self) -> float | None:
        return round(self.held_sum / self.n, 1) if self.n else None

    @property
    def avg_return_on_risk_pct(self) -> float | None:
        """Mean per-trade P&L as % of that trade's capital at risk -- the
        comparable-across-stocks unit that per-share P&L is not."""
        return round(100.0 * self.ror_sum / self.ror_n, 1) if self.ror_n else None


def regime_map_from_bench(bench_bars: pd.DataFrame | None) -> dict:
    """{date -> "bull"/"bear"} from the benchmark's close vs its 200-day SMA --
    the same signal options.benchmark_regime computes live, vectorized over
    history so each backtest entry date gets the regime as it stood THEN
    (no lookahead: the SMA at date t uses only bars up to t)."""
    if bench_bars is None or bench_bars.empty or len(bench_bars) < options_engine.REGIME_SMA:
        return {}
    close = bench_bars["close"]
    sma = close.rolling(options_engine.REGIME_SMA).mean()
    out = {}
    for ts, c, m in zip(bench_bars.index, close, sma):
        if np.isfinite(c) and np.isfinite(m):
            out[ts.date()] = "bull" if c >= m else "bear"
    return out


def backtest_symbol(
    code: str, name: str, bars: pd.DataFrame, *,
    horizon: int = DEFAULT_HORIZON, step: int = DEFAULT_STEP, vrp: float = DEFAULT_VRP,
    ppy: float = 252.0, regime_map: dict | None = None,
) -> list[Trade]:
    """Walk `bars` forward, entering strategies every `step` trading days and
    marking them to the realized close `horizon` bars later. `bars` is an
    ascending OHLCV frame (normalize.bars_from_kline output). `regime_map`
    (date -> "bull"/"bear") enables the strategist's counter-regime gate,
    exactly as the live path does; None leaves the gate off."""
    trades: list[Trade] = []
    n = len(bars)
    closes = bars["close"].to_numpy()
    idx = bars.index
    i = MIN_TRAIL_BARS
    entry_no = 0
    fv_cache: float | None = None   # reused GARCH forecast between refits
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
        regime = (regime_map or {}).get(idx[i].date())
        # GARCH is the per-entry bottleneck but barely moves over `step` days --
        # refit every GARCH_REFIT_EVERY entries and reuse in between (the live
        # path still refits every call; this speedup is backtest-only).
        if entry_no % GARCH_REFIT_EVERY == 0:
            fv_cache = options_math.forecast_vol_garch(trail, horizon)
        entry_no += 1
        result = options_engine.build_analysis(
            code=code, name=name, as_of=str(idx[i].date()), spot=spot,
            decision=ta.decision, score=ta.score, bars=trail,
            contracts=chain, expiry=str(idx[i + horizon].date()), dte=horizon,
            holds=False, shares=0.0, confidence=ta.confidence,
            market_regime=regime, forecast_vol_pct=fv_cache if fv_cache is not None else -1.0,
        )
        exit_spot = float(closes[i + horizon])
        path_spots = closes[i + 1 : i + horizon + 1]   # realized daily closes, entry+1..expiry
        for s in result.strategies:
            if any(leg.price is None for leg in s.legs):
                continue
            pnl = float(options_math.payoff_at_expiry(s.legs, np.array([exit_spot]))[0])
            m_pnl, reason, held = managed_exit(s, spot, path_spots, horizon)
            trades.append(Trade(
                date=str(idx[i].date()), symbol=code, strategy=s.name, direction=s.direction,
                decision=ta.decision.value, pop_pct=s.pop_pct, ev_per_share=s.ev_per_share,
                entry_spot=round(spot, 2), exit_spot=round(exit_spot, 2),
                pnl_per_share=round(pnl, 3), win=pnl > 0, regime=regime,
                managed_pnl=round(m_pnl, 3), exit_reason=reason, days_held=held,
                risk_per_share=s.max_loss if (s.max_loss or 0) > 0 else None,
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


def _table(stats: dict[str, StratStats]) -> list[str]:
    lines = [
        f"{'Strategy':<26}{'n':>6}{'win%':>8}{'pred POP':>10}{'avg P&L/sh':>12}{'ret/risk':>10}",
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
        ror = f"{s.avg_return_on_risk_pct:+.1f}%" if s.avg_return_on_risk_pct is not None else "  -"
        lines.append(
            f"{label[:26]:<26}{s.n:>6}{s.win_rate:>7.1f}%{pop:>10}{s.avg_pnl:>12.3f}{ror:>10}"
        )
    return lines


def format_report(trades: list[Trade], meta: dict) -> str:
    """Plain-text calibration report. The headline column is predicted-POP vs
    realized win-rate: if they track, the strategist's probability model is
    calibrated; a large persistent gap means it isn't. When trades carry a
    market regime, per-regime tables show how structures behaved in bull vs
    bear tape -- the whole point of the counter-regime gate."""
    lines = [
        "=" * 78,
        "OPTIONS STRATEGIST -- SYNTHETIC BACKTEST",
        "=" * 78,
        f"Symbols: {meta['n_symbols']}   Entries: every {meta['step']} trading days   "
        f"Tenor: {meta['horizon']} trading days   VRP: {meta['vrp']:.0%}   "
        f"Regime gate: {'ON' if meta.get('regime') else 'OFF'}",
        f"Drift bull/bear: {options_engine.DRIFT_BULL_PCT:+.0f}%/{options_engine.DRIFT_BEAR_PCT:+.0f}%   "
        f"EV gate: {'ON' if options_engine.EV_GATE else 'OFF'}   "
        f"Bear->credit: {'ON' if options_engine.BEAR_PREFERS_CREDIT else 'OFF'}   "
        f"Bear-conf: {options_engine.BEAR_DIRECTIONAL_CONFIDENCE:.0%}   "
        f"Non-S&P: drift {options_engine.DRIFT_BULL_NONSP_PCT:+.0f}%, "
        f"credit-pref {'ON' if options_engine.NONSP_PREFERS_CREDIT else 'OFF'}",
        f"Modeled IV = trailing Yang-Zhang realized vol x (1+VRP); marks hold-to-expiry.",
        "-" * 78,
    ]
    lines += _table(aggregate(trades))
    for regime in ("bull", "bear"):
        sub = [t for t in trades if t.regime == regime]
        if not sub:
            continue
        span = f"{min(t.date for t in sub)} .. {max(t.date for t in sub)}"
        lines += ["", f"--- {regime.upper()} regime only ({len(sub)} trades, {span}) " + "-" * 20]
        lines += _table(aggregate(sub))
    if any(t.managed_pnl is not None for t in trades):
        lines += ["", "--- EARLY MANAGEMENT: hold-to-expiry vs managed exit " + "-" * 26]
        lines += _manage_table(aggregate(trades))
    lines += [
        "-" * 78,
        "win% = fraction of entries with P&L > 0 at expiry on the REAL price path.",
        "pred POP = strategist's mean predicted probability of profit at entry.",
        "managed = per-structure early exit (credit: take 70%; debit: ride + BE-stop).",
        "Caveat: entry premiums are MODELED (no historical chains exist), so this",
        "validates structure/strike/POP calibration, NOT the real IV-timing edge.",
        "=" * 78,
    ]
    return "\n".join(lines)


def _manage_table(stats: dict[str, StratStats]) -> list[str]:
    lines = [
        f"{'Strategy':<26}{'n':>5}{'hold P&L':>10}{'mgd P&L':>10}{'chg':>8}"
        f"{'mgd win%':>10}{'~days':>7}",
        "-" * 78,
    ]
    order = sorted((k for k in stats if k != "__ALL__"), key=lambda k: stats[k].n, reverse=True)
    for k in order + ["__ALL__"]:
        s = stats[k]
        if not s.n or s.m_avg_pnl is None:
            continue
        label = "ALL" if k == "__ALL__" else k
        delta = s.m_avg_pnl - s.avg_pnl
        lines.append(
            f"{label[:26]:<26}{s.n:>5}{s.avg_pnl:>10.3f}{s.m_avg_pnl:>10.3f}{delta:>+8.3f}"
            f"{s.m_win_rate:>9.1f}%{s.avg_days_held:>7.1f}"
        )
    return lines


IBKR_PACING_S = 10.0   # IB allows ~60 historical requests / 10 min -> 1 per 10s


def _cache_deep_enough(bars, lookback_days: int, min_bars: int) -> bool:
    """A cached series is reusable for a backtest if it has enough rows AND spans
    most of the requested lookback -- the training cache is shallow (~750 bars to
    2023), so this correctly rejects it and forces a deep re-fetch, while a
    previously deep-fetched series is reused for free."""
    if bars is None or len(bars) < min_bars:
        return False
    span_days = (bars.index.max() - bars.index.min()).days
    return span_days >= lookback_days * 0.9


def _fetch_all(symbols: list[str], lookback_days: int, horizon: int, use_ibkr: bool = True):
    """Get every symbol's bars UP FRONT (plus the benchmark), then the compute
    phase needs no broker -- so a dropped connection can't waste hours of walking.
    Per symbol: a deep-enough cache is reused as-is; else Moomoo (free for names
    already in its 7-day quota set); else IBKR (no quota wall, reaches the ~half
    of the S&P 500 the Moomoo cache lacks). Anything fetched is saved back to the
    cache, so the expensive IBKR pull is a one-time cost."""
    import time
    from app.data import normalize
    from app.ml import data_store
    from app.services.analysis_service import BENCHMARK_CODE, service

    KLINE_MIN_INTERVAL_S = 0.55  # Moomoo 60-calls/30s cap, same as train.py
    min_bars = MIN_TRAIL_BARS + horizon
    store = data_store.BarStore()

    ibkr = None
    if use_ibkr:
        try:
            from app.brokers.ibkr_client import IBKRClient
            ibkr = IBKRClient(client_id=1150).connect()
            print("  IBKR fallback: connected", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  IBKR fallback unavailable ({e}) -- Moomoo only", flush=True)

    def _get(code: str):
        """(bars, source) or (None, None). Cache -> Moomoo -> IBKR."""
        cached = store.load(code)
        if _cache_deep_enough(cached, lookback_days, min_bars):
            return cached, "cache"
        try:
            time.sleep(KLINE_MIN_INTERVAL_S)
            b = store.update(code, service._client, service._lock, lookback_days=lookback_days)
            if b is not None and len(b) >= min_bars:
                return b, "moomoo"
        except Exception:  # noqa: BLE001 - quota / no permission -> try IBKR
            pass
        if ibkr is not None:
            try:
                time.sleep(IBKR_PACING_S)
                raw = ibkr.get_history_kline(code, ktype="day", duration="7 Y")
                b = normalize.bars_from_kline(raw) if raw is not None else None
                if b is not None and len(b) >= min_bars:
                    store.save(code, b)   # persist -> reused free on the next run
                    return b, "ibkr"
            except Exception:  # noqa: BLE001
                pass
        return None, None

    bench_bars, _ = _get(BENCHMARK_CODE)
    if bench_bars is None:
        print(f"  benchmark {BENCHMARK_CODE} unavailable -- regime gate will be off", flush=True)

    bars_by_code: dict[str, pd.DataFrame] = {}
    src_counts = {"cache": 0, "moomoo": 0, "ibkr": 0}
    fetch_list = [c for c in symbols if c != BENCHMARK_CODE]
    for i, code in enumerate(fetch_list, 1):
        bars, src = _get(code)
        if bars is None:
            print(f"  fetch [{i}/{len(fetch_list)}] {code}: SKIPPED", flush=True)
            continue
        bars_by_code[code] = bars
        src_counts[src] += 1
        if i % 25 == 0:
            print(f"  fetch [{i}/{len(fetch_list)}] ... {src_counts}", flush=True)
    if BENCHMARK_CODE in symbols and bench_bars is not None:
        bars_by_code[BENCHMARK_CODE] = bench_bars
    print(f"  fetched {len(bars_by_code)}/{len(symbols)} symbols  sources={src_counts}", flush=True)
    if ibkr is not None:
        try:
            ibkr.close()
        except Exception:  # noqa: BLE001
            pass
    return bars_by_code, bench_bars


def _walk_all(bars_by_code: dict, *, horizon: int, step: int, vrp: float, regime_map: dict | None):
    trades: list[Trade] = []
    for j, (code, bars) in enumerate(bars_by_code.items(), 1):
        t = backtest_symbol(code, code, bars, horizon=horizon, step=step, vrp=vrp,
                            regime_map=regime_map)
        trades.extend(t)
        if j % 10 == 0 or j == len(bars_by_code):
            print(f"    walked {j}/{len(bars_by_code)} symbols, {len(trades)} trades so far", flush=True)
    return trades


def run(symbols: list[str], *, horizon: int, step: int, vrp: float, lookback_days: int,
        use_regime: bool = True, ab: bool = False) -> str:
    """Fetch real Moomoo equity history once, then backtest. `ab=True` runs the
    walk twice from that single fetch -- gate ON and gate OFF -- for a same-data
    A/B of the regime gate (no double fetch, no double quota hit)."""
    bars_by_code, bench_bars = _fetch_all(symbols, lookback_days, horizon)
    if not bars_by_code:
        return "No symbols returned usable history -- nothing to backtest."
    regime_map = regime_map_from_bench(bench_bars) if bench_bars is not None else {}
    print(f"  regime map: {len(regime_map)} dated reads", flush=True)

    meta = {"n_symbols": len(bars_by_code), "horizon": horizon, "step": step, "vrp": vrp}
    if ab:
        print("  == walk A: gate ON ==", flush=True)
        on = _walk_all(bars_by_code, horizon=horizon, step=step, vrp=vrp, regime_map=regime_map or None)
        print("  == walk B: gate OFF ==", flush=True)
        off = _walk_all(bars_by_code, horizon=horizon, step=step, vrp=vrp, regime_map=None)
        rep_on = format_report(on, {**meta, "regime": True})
        rep_off = format_report(off, {**meta, "regime": False})
        return rep_on + "\n\n" + rep_off + "\n\n" + _ab_delta(on, off)

    regime = regime_map if use_regime else {}
    trades = _walk_all(bars_by_code, horizon=horizon, step=step, vrp=vrp, regime_map=regime or None)
    return format_report(trades, {**meta, "regime": bool(regime)})


def _ab_delta(on: list[Trade], off: list[Trade]) -> str:
    """One-line-per-metric summary of the gate's effect, computed from the same
    underlying bars (the only thing that differs is the gate)."""
    def tot(ts):
        n = len(ts)
        wins = sum(1 for t in ts if t.win)
        pnl = sum(t.pnl_per_share for t in ts)
        return n, (100.0 * wins / n if n else 0.0), pnl, (pnl / n if n else 0.0)
    n_on, wr_on, p_on, avg_on = tot(on)
    n_off, wr_off, p_off, avg_off = tot(off)
    return "\n".join([
        "=" * 78,
        "A/B: REGIME GATE ON vs OFF (same bars, same period)",
        "-" * 78,
        f"{'':<16}{'trades':>10}{'win%':>10}{'total P&L':>14}{'avg/trade':>12}",
        f"{'gate OFF':<16}{n_off:>10}{wr_off:>9.1f}%{p_off:>14.1f}{avg_off:>12.3f}",
        f"{'gate ON':<16}{n_on:>10}{wr_on:>9.1f}%{p_on:>14.1f}{avg_on:>12.3f}",
        f"{'delta':<16}{n_on - n_off:>+10}{wr_on - wr_off:>+9.1f}%{p_on - p_off:>+14.1f}{avg_on - avg_off:>+12.3f}",
        "=" * 78,
    ])


def main() -> None:
    import argparse
    from app.ml import data_store, universe

    ap = argparse.ArgumentParser(description="Synthetic backtest of the options strategist.")
    ap.add_argument("--symbols", default=None, help="comma-separated codes (e.g. US.AAPL,US.MSFT)")
    ap.add_argument("--universe", default=None,
                    help="cached | holdings | sp500 | smallcap | <file> (ignored if --symbols given)")
    ap.add_argument("--horizon", type=int, default=DEFAULT_HORIZON, help="tenor in trading days")
    ap.add_argument("--step", type=int, default=DEFAULT_STEP, help="roll cadence in trading days")
    ap.add_argument("--vrp", type=float, default=DEFAULT_VRP, help="vol risk premium (IV = realized x (1+vrp))")
    ap.add_argument("--lookback-days", type=int, default=1095)
    ap.add_argument("--max-symbols", type=int, default=20)
    ap.add_argument("--no-regime", action="store_true",
                    help="disable the counter-regime gate")
    ap.add_argument("--ab", action="store_true",
                    help="run gate ON and OFF from one fetch and print an A/B delta")
    ap.add_argument("--drift-bull", type=float, default=options_engine.DRIFT_BULL_PCT,
                    help="annualized drift fed to POP/EV in a bull regime (%%)")
    ap.add_argument("--drift-bear", type=float, default=options_engine.DRIFT_BEAR_PCT,
                    help="annualized drift fed to POP/EV in a bear regime (%%)")
    ap.add_argument("--no-ev-gate", action="store_true",
                    help="disable withholding negative-model-EV structures")
    ap.add_argument("--no-bear-credit", action="store_true",
                    help="disable the bear-regime preference for credit structures")
    ap.add_argument("--bear-conf", type=float, default=options_engine.BEAR_DIRECTIONAL_CONFIDENCE,
                    help="confidence needed for a directional bearish trade in a bear tape (0 disables)")
    ap.add_argument("--drift-nonsp", type=float, default=options_engine.DRIFT_BULL_NONSP_PCT,
                    help="bull-regime drift for non-S&P names (%%; size preset)")
    ap.add_argument("--no-nonsp-credit", action="store_true",
                    help="disable the non-S&P preference for credit structures")
    args = ap.parse_args()
    options_engine.DRIFT_BULL_PCT = args.drift_bull
    options_engine.DRIFT_BEAR_PCT = args.drift_bear
    options_engine.EV_GATE = not args.no_ev_gate
    options_engine.BEAR_PREFERS_CREDIT = not args.no_bear_credit
    options_engine.BEAR_DIRECTIONAL_CONFIDENCE = args.bear_conf
    options_engine.DRIFT_BULL_NONSP_PCT = args.drift_nonsp
    options_engine.NONSP_PREFERS_CREDIT = not args.no_nonsp_credit
    # Windows consoles default to cp1252, which can't encode non-ASCII; force
    # UTF-8 so the report (and any stray unicode) never crashes the final print.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass

    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.universe == "cached":
        # US equities already in the bar-store cache -> guaranteed inside Moomoo's
        # 7-day distinct-stock quota set, so a deep re-fetch costs no fresh quota.
        import glob
        files = glob.glob(str(data_store.bars_dir() / "US.*.parquet"))
        symbols = sorted(os.path.basename(f)[:-8] for f in files)[: args.max_symbols]
    elif args.universe:
        symbols = [c for c in universe.resolve_universe(source=args.universe)][: args.max_symbols]
    else:
        ap.error("pass --symbols or --universe")
    print(f"Backtesting {len(symbols)} symbol(s)...\n", flush=True)
    report = run(symbols, horizon=args.horizon, step=args.step, vrp=args.vrp,
                 lookback_days=args.lookback_days, use_regime=not args.no_regime, ab=args.ab)
    print(report, flush=True)
    # The Moomoo SDK leaves non-daemon network threads running, so the process
    # won't exit on its own after the report is done -- force it, or the run
    # hangs forever with output already flushed. ponytail: os._exit skips
    # atexit/GC, fine for a one-shot CLI that has printed everything it needs.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
