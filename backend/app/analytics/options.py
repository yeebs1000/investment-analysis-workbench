"""OptionsStrategist: from a chain (with Greeks) + the technical read + holdings
to concrete, defined strategies with strikes, tenor, and economics.

Like the rest of the app this is deterministic: IV regime, strike selection (by
delta), and trade economics are all computed here. An LLM may later narrate it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics import options_math
from app.data.models import (
    Decision,
    OptionLeg,
    OptionsAnalysis,
    OptionStrategy,
)

TARGET_DELTA = 0.30          # standard short-strike delta for income/spreads
WING_DELTA = 0.15            # protective long wing for credit spreads
CONDOR_SHORT_DELTA = 0.20    # narrower than TARGET_DELTA -- market convention for condors
SKEW_DELTA = 0.25            # wing delta used for the 25-delta put/call skew read
IV_ELEVATED_RATIO = 1.25     # ATM IV / realized vol above this -> "Elevated" (sell premium)
IV_CHEAP_RATIO = 0.85        # ... below this -> "Cheap" (buy premium)
CONTRACT_SIZE = 100          # shares per standard equity option contract

# --- context/risk screens ----------------------------------------------------
LIQ_MAX_SPREAD_PCT = 10.0    # bid/ask spread over mid above this -> illiquid-leg warning
LIQ_MIN_OI = 50              # open interest below this -> thin-contract warning
DEFAULT_RISK_BUDGET_FRAC = 0.01   # max loss per trade as a fraction of book value (1%)
LOW_CONFIDENCE_THRESHOLD = 0.45   # technical confidence below this -> directional read is
                                   # flagged and neutral structures are offered alongside

# --- market-regime gate ------------------------------------------------------
# Counter-regime directional trades (bearish structures in a bull market and
# vice versa) need MORE conviction than with-regime ones: the 35d synthetic
# backtest showed counter-trend debit spreads were the single worst bucket
# (Put Debit Spread: 29% win rate in a bull tape). Below this confidence a
# counter-regime read is demoted to neutral -- protective structures for
# holders (collar) are exempt, protection is never gated away.
REGIME_SMA = 200                  # benchmark close vs its 200-day SMA: the boring classic
REGIME_OVERRIDE_CONFIDENCE = 0.60 # confidence needed to trade AGAINST the regime
SKEW_NOTE_THRESHOLD_PTS = 3.0     # |25d put IV - 25d call IV| above this earns a skew note
TERM_NOTE_THRESHOLD_PTS = 2.0     # front-vs-next ATM IV gap above this earns a term note


def _realized_vol_close_close(close: pd.Series, window: int = 30) -> float | None:
    """Fallback estimator (simple close-to-close stdev) used only when the
    Yang-Zhang range estimator can't run (e.g. insufficient OHLC history)."""
    rets = np.log(close / close.shift(1)).dropna()
    if len(rets) < window:
        window = len(rets)
    if window < 5:
        return None
    return float(rets.tail(window).std(ddof=0) * np.sqrt(252) * 100.0)


def benchmark_regime(bench_close: pd.Series | None) -> str | None:
    """"bull" / "bear" from the benchmark's close vs its 200-day SMA, or None
    when there isn't enough history to say. Deterministic and computable at any
    historical date, unlike the live-only FRED macro read -- so the SAME signal
    the backtest validates is the one the live strategist uses."""
    if bench_close is None or len(bench_close) < REGIME_SMA:
        return None
    sma = float(bench_close.rolling(REGIME_SMA).mean().iloc[-1])
    last = float(bench_close.iloc[-1])
    if not (np.isfinite(sma) and np.isfinite(last)):
        return None
    return "bull" if last >= sma else "bear"


def _bias(decision: Decision | None) -> str:
    if decision in (Decision.STRONG_BUY, Decision.BUY, Decision.ACCUMULATE):
        return "bullish"
    if decision in (Decision.SELL, Decision.REDUCE):
        return "bearish"
    return "neutral"


def _leg_from(row: pd.Series, action: str, dte: int, expiry: str) -> OptionLeg:
    return OptionLeg(
        action=action,
        right="Call" if str(row["right"]).upper() == "CALL" else "Put",
        strike=float(row["strike"]),
        expiry=expiry,
        dte=dte,
        delta=_num(row.get("delta")),
        iv_pct=_num(row.get("iv")),
        price=_num(row.get("price")),
        bid=_num(row.get("bid")),
        ask=_num(row.get("ask")),
        oi=_num(row.get("oi")),
        code=str(row.get("code")) if row.get("code") is not None else None,
    )


def _num(x) -> float | None:
    try:
        f = float(x)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _side(contracts: pd.DataFrame, right: str) -> pd.DataFrame:
    return contracts[contracts["right"].str.upper() == right.upper()].copy()


def _by_delta(df: pd.DataFrame, target: float) -> pd.Series | None:
    d = df.dropna(subset=["delta"])
    d = d[d["delta"].abs() > 0]
    if d.empty:
        return None
    idx = (d["delta"].abs() - target).abs().idxmin()
    return d.loc[idx]


def _by_strike(df: pd.DataFrame, strike: float) -> pd.Series | None:
    if df.empty:
        return None
    return df.loc[(df["strike"] - strike).abs().idxmin()]


def _atm(df: pd.DataFrame, spot: float) -> pd.Series | None:
    return _by_strike(df, spot)


def build_analysis(
    *,
    code: str,
    name: str,
    as_of: str | None,
    spot: float,
    decision: Decision | None,
    score: float | None,
    bars: pd.DataFrame,
    contracts: pd.DataFrame,
    expiry: str,
    dte: int,
    holds: bool,
    shares: float,
    analyst: dict | None = None,
    stock_target: float | None = None,
    stock_stop: float | None = None,
    confidence: float | None = None,
    earnings: dict | None = None,
    book_value_usd: float | None = None,
    risk_budget_frac: float = DEFAULT_RISK_BUDGET_FRAC,
    next_atm_iv_pct: float | None = None,
    next_expiry: str | None = None,
    market_regime: str | None = None,   # "bull" | "bear" | None=unknown (see benchmark_regime)
) -> OptionsAnalysis:
    out = OptionsAnalysis(
        code=code, name=name, as_of=as_of, spot=spot,
        technical_decision=decision, technical_score=score,
        expiry_used=expiry, dte=dte, holds_underlying=holds, shares_held=shares,
        analyst_consensus=analyst,
    )
    if contracts is None or contracts.empty:
        out.error = "No option contracts with quotes were available for this expiry."
        return out

    calls, puts = _side(contracts, "CALL"), _side(contracts, "PUT")

    # --- earnings context: the single biggest knowable event risk in a 30-45d
    # tenor. None means UNKNOWN (coverage skews US), never "no earnings coming".
    days_to_earnings: int | None = None
    if earnings and earnings.get("date"):
        out.earnings_date = str(earnings["date"])
        try:
            from datetime import date as _date
            days_to_earnings = (_date.fromisoformat(out.earnings_date) - _date.today()).days
            out.days_to_earnings = days_to_earnings
        except ValueError:
            days_to_earnings = None
    earnings_within_tenor = days_to_earnings is not None and 0 <= days_to_earnings <= dte

    # --- IV regime: ATM IV vs a vol BASELINE ---
    # Trailing Yang-Zhang realized vol (for display + fallback)...
    rv = options_math.realized_vol_yang_zhang(bars)
    out.vol_estimator = "yang_zhang"
    if rv is None:
        rv = _realized_vol_close_close(bars["close"])
        out.vol_estimator = "close_to_close"
    out.realized_vol_pct = round(rv, 1) if rv else None
    # ...and a GARCH(1,1) FORECAST over the option's actual tenor, which is the
    # statistically correct baseline: implied vol prices variance over the
    # contract's life, so 'rich/cheap' should be IV vs forward vol for that
    # horizon, not vs what already happened. Falls back to realized on any miss.
    fv = options_math.forecast_vol_garch(bars, dte)
    out.forecast_vol_pct = round(fv, 1) if fv else None
    baseline = fv if (fv and fv > 0) else rv
    out.iv_regime_basis = "garch_forecast" if (fv and fv > 0) else "realized"

    atm_ivs = []
    for side in (calls, puts):
        a = _atm(side, spot)
        if a is not None and _num(a.get("iv")) and a["iv"] > 0:
            atm_ivs.append(float(a["iv"]))
    atm_iv = float(np.mean(atm_ivs)) if atm_ivs else None
    out.atm_iv_pct = round(atm_iv, 1) if atm_iv else None
    if atm_iv and baseline and baseline > 0:
        ratio = atm_iv / baseline
        out.iv_vs_realized = round(ratio, 2)
        out.iv_regime = "Elevated" if ratio >= IV_ELEVATED_RATIO else "Cheap" if ratio <= IV_CHEAP_RATIO else "Normal"
    bias = _bias(decision)

    # --- market-regime gate: counter-regime directional reads need conviction.
    # raw_bias survives for PROTECTIVE structures (collar) -- a holder's hedge
    # on a bearish read is never gated away, only speculative direction is.
    raw_bias = bias
    counter_regime = (
        (bias == "bearish" and market_regime == "bull")
        or (bias == "bullish" and market_regime == "bear")
    )
    if counter_regime and (confidence is None or confidence < REGIME_OVERRIDE_CONFIDENCE):
        bias = "neutral"
        out.notes.append(
            f"⚖ Regime gate: the {raw_bias} read fights a {market_regime} market "
            f"(benchmark vs its {REGIME_SMA}-day average) without high conviction"
            + (f" ({confidence:.0%} < {REGIME_OVERRIDE_CONFIDENCE:.0%})" if confidence is not None else "")
            + f" — directional {raw_bias} structures are withheld; neutral structures shown instead."
        )
    elif counter_regime:
        out.notes.append(
            f"⚖ Counter-regime trade: this {bias} read fights a {market_regime} market — "
            f"kept because conviction is high ({confidence:.0%}), but size accordingly."
        )

    iv_high = out.iv_regime == "Elevated"
    iv_low = out.iv_regime == "Cheap"

    # Low technical conviction -> don't pretend the directional read is solid:
    # flag it, and ALSO offer neutral structures alongside the directional ones.
    low_conviction = (
        bias != "neutral" and confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD
    )

    if atm_iv and baseline:
        if out.iv_regime_basis == "garch_forecast":
            iv_detail = (
                f" (ATM IV {out.atm_iv_pct}% vs {out.forecast_vol_pct}% GARCH forecast "
                f"over {dte}d; trailing realized {out.realized_vol_pct}%)."
            )
        else:
            iv_detail = f" (ATM IV {out.atm_iv_pct}% vs {out.realized_vol_pct}% trailing realized)."
    else:
        iv_detail = "."
    out.notes.append(
        f"View: {bias} (technical {decision.value if decision else 'n/a'}"
        + (f", confidence {confidence:.0%}" if confidence is not None else "")
        + f"). IV regime: {out.iv_regime or 'unknown'}" + iv_detail
    )
    if low_conviction:
        out.notes.append(
            f"⚠ The directional read is LOW confidence ({confidence:.0%}) — treat the "
            f"{bias} structures as tentative; neutral structures are shown alongside them."
        )

    # --- 25-delta skew: what the market is paying up for, from the same chain ---
    skew = _skew_25d(calls, puts)
    if skew is not None:
        out.skew_25d_pts = round(skew, 1)
        if skew >= SKEW_NOTE_THRESHOLD_PTS:
            out.notes.append(
                f"Put skew: 25Δ puts trade {skew:.1f} vol pts over 25Δ calls — downside "
                f"protection is expensive (put credit spreads collect relatively more; "
                f"long puts cost relatively more)."
            )
        elif skew <= -SKEW_NOTE_THRESHOLD_PTS:
            out.notes.append(
                f"Call skew: 25Δ calls trade {-skew:.1f} vol pts over 25Δ puts — upside "
                f"speculation is bid (covered calls/call credit spreads collect relatively more)."
            )

    # --- term structure: front tenor vs a longer one (both ATM) ---
    if next_atm_iv_pct is not None and atm_iv is not None:
        out.next_expiry_used = next_expiry
        out.next_atm_iv_pct = round(next_atm_iv_pct, 1)
        gap = atm_iv - next_atm_iv_pct
        if gap >= TERM_NOTE_THRESHOLD_PTS:
            out.notes.append(
                f"Term structure INVERTED: this expiry's ATM IV ({atm_iv:.0f}%) is above the "
                f"longer tenor ({next_atm_iv_pct:.0f}% at {next_expiry}) — the market is "
                f"pricing a near-term event into this expiry specifically."
            )
        elif gap <= -TERM_NOTE_THRESHOLD_PTS:
            out.notes.append(
                f"Term structure upward-sloping (this expiry {atm_iv:.0f}% vs "
                f"{next_atm_iv_pct:.0f}% at {next_expiry}) — normal contango, no event "
                f"premium concentrated in this tenor."
            )

    if out.earnings_date and days_to_earnings is not None:
        where = "WITHIN this option tenor" if earnings_within_tenor else "after this expiry"
        out.notes.append(
            f"Next earnings: {out.earnings_date} (in {days_to_earnings}d — {where})."
        )
    elif earnings is None:
        out.notes.append(
            "Earnings date unknown (no calendar coverage for this symbol) — verify the "
            "next report date yourself before selling premium."
        )

    strategies: list[OptionStrategy] = []

    # --- Covered call: holds >=100 shares, not bearish ---
    if holds and shares >= CONTRACT_SIZE and bias in ("neutral", "bullish"):
        c = _by_delta(calls, TARGET_DELTA)
        if c is not None and (_num(c.get("price")) or 0.0) > 0:
            prem = _num(c.get("price")) or 0.0
            strategies.append(OptionStrategy(
                name="Covered Call",
                direction="Income",
                legs=[_leg_from(c, "Sell", dte, expiry)],
                tenor_dte=dte,
                net_debit_credit=round(prem, 2),
                max_profit=round(c["strike"] - spot + prem, 2),   # stock appreciates to strike + premium
                max_loss=round(spot - prem, 2),                    # stock to zero, cushioned by premium
                breakeven=round(spot - prem, 2),
                rationale=(
                    f"You hold {int(shares)} shares. Selling the ~{TARGET_DELTA:.0%}-delta "
                    f"{c['strike']:.0f} call collects ~${prem:.2f}/share of premium and gives a "
                    f"small downside cushion. You keep the stock unless it closes above "
                    f"{c['strike']:.0f} at expiry (then it's called away)."
                ),
                suited_when="You're neutral-to-mildly-bullish and happy to cap upside for income — best when IV is elevated.",
            ))

    # --- Collar: holds >=100 shares AND (bearish read OR earnings inside the
    # tenor). THE structure for a long-term holder who wants to carry a core
    # position through a rough patch/binary event without selling: buy an OTM
    # put (hard floor), pay for it by selling an OTM call (capped upside) —
    # often near zero net cost.
    # raw_bias, not bias: a regime-demoted bearish read still justifies HEDGING
    # an existing position -- the gate withholds speculation, never protection.
    if holds and shares >= CONTRACT_SIZE and (raw_bias == "bearish" or earnings_within_tenor):
        put_leg = _by_delta(puts, TARGET_DELTA)
        call_leg = _by_delta(calls, TARGET_DELTA)
        if (
            put_leg is not None and call_leg is not None
            and (_num(put_leg.get("price")) or 0) > 0 and (_num(call_leg.get("price")) or 0) > 0
            and put_leg["strike"] < spot < call_leg["strike"]
        ):
            call_prem = _num(call_leg.get("price")) or 0.0
            put_cost = _num(put_leg.get("price")) or 0.0
            net = call_prem - put_cost          # + = collar pays you, - = costs you
            k_put, k_call = float(put_leg["strike"]), float(call_leg["strike"])
            why = ("the technical read is bearish" if raw_bias == "bearish"
                   else f"earnings land inside this tenor ({out.earnings_date})")
            strategies.append(OptionStrategy(
                name="Collar",
                direction="Protective",
                legs=[_leg_from(put_leg, "Buy", dte, expiry), _leg_from(call_leg, "Sell", dte, expiry)],
                tenor_dte=dte,
                net_debit_credit=round(net, 2),
                max_loss=round(spot - k_put - net, 2),     # floor: stock to put strike, net premium adjusts
                max_profit=round(k_call - spot + net, 2),  # cap: stock to call strike, net premium adjusts
                breakeven=round(spot - net, 2),
                rationale=(
                    f"You hold {int(shares)} shares and {why} — a collar lets you keep the "
                    f"position with a hard floor. Buy the {k_put:.0f} put (worst case you sell "
                    f"there), sell the {k_call:.0f} call to pay for it "
                    f"({'net credit' if net >= 0 else 'net cost'} ~${abs(net):.2f}/share). "
                    f"Downside is capped at {k_put:.0f}; upside is capped at {k_call:.0f} until expiry."
                ),
                suited_when=(
                    "Long-term holder facing a bearish stretch or a binary event — you want to "
                    "stay invested but sleep at night. Cheapest when call skew is bid."
                ),
            ))

    # --- Bullish ---
    if bias == "bullish":
        if iv_high:
            short = _by_delta(puts, TARGET_DELTA)
            long = _by_delta(puts, WING_DELTA)
            if short is not None and long is not None and long["strike"] < short["strike"] \
                    and 0 < ((_num(short.get("price")) or 0) - (_num(long.get("price")) or 0)) < (short["strike"] - long["strike"]):
                credit = (_num(short.get("price")) or 0) - (_num(long.get("price")) or 0)
                width = short["strike"] - long["strike"]
                strategies.append(OptionStrategy(
                    name="Bull Put Spread (credit)",
                    direction="Bullish",
                    legs=[_leg_from(short, "Sell", dte, expiry), _leg_from(long, "Buy", dte, expiry)],
                    tenor_dte=dte,
                    net_debit_credit=round(credit, 2),
                    max_profit=round(credit, 2),
                    max_loss=round(width - credit, 2),
                    breakeven=round(short["strike"] - credit, 2),
                    rationale=(
                        f"IV is elevated, so selling premium is favoured. Sell the {short['strike']:.0f} "
                        f"put and buy the {long['strike']:.0f} put for ~${credit:.2f} credit. You profit "
                        f"if {name} stays above {short['strike']:.0f}; risk is capped."
                    ),
                    suited_when="Bullish and want defined risk while IV is rich.",
                ))
            csp = _by_delta(puts, TARGET_DELTA)
            if csp is not None and (_num(csp.get("price")) or 0.0) > 0:
                prem = _num(csp.get("price")) or 0.0
                strategies.append(OptionStrategy(
                    name="Cash-Secured Put",
                    direction="Bullish",
                    legs=[_leg_from(csp, "Sell", dte, expiry)],
                    tenor_dte=dte,
                    net_debit_credit=round(prem, 2),
                    max_loss=round(csp["strike"] - prem, 2),
                    breakeven=round(csp["strike"] - prem, 2),
                    rationale=(
                        f"Get paid ~${prem:.2f}/share to agree to buy {name} at {csp['strike']:.0f}. "
                        f"If it stays above {csp['strike']:.0f} you keep the premium; if it falls you buy "
                        f"at an effective {csp['strike'] - prem:.2f}. Needs ~${csp['strike']*100:,.0f} cash set aside."
                    ),
                    suited_when="You'd happily own the stock lower and IV is elevated.",
                ))
        else:
            long = _atm(calls, spot)
            short = _by_delta(calls, TARGET_DELTA)
            if long is not None and short is not None and short["strike"] > long["strike"] \
                    and 0 < ((_num(long.get("price")) or 0) - (_num(short.get("price")) or 0)) < (short["strike"] - long["strike"]):
                debit = (_num(long.get("price")) or 0) - (_num(short.get("price")) or 0)
                width = short["strike"] - long["strike"]
                strategies.append(OptionStrategy(
                    name="Call Debit Spread",
                    direction="Bullish",
                    legs=[_leg_from(long, "Buy", dte, expiry), _leg_from(short, "Sell", dte, expiry)],
                    tenor_dte=dte,
                    net_debit_credit=round(-debit, 2),
                    max_loss=round(debit, 2),
                    max_profit=round(width - debit, 2),
                    breakeven=round(long["strike"] + debit, 2),
                    rationale=(
                        f"IV is {'cheap' if iv_low else 'reasonable'}, so buying directional exposure is "
                        f"efficient. Buy the {long['strike']:.0f} call and sell the {short['strike']:.0f} "
                        f"call for ~${debit:.2f}. Cheaper than a naked call; profit grows up to {short['strike']:.0f}."
                    ),
                    suited_when="Bullish with a target near the short strike; want defined cost.",
                ))

    # --- Bearish ---
    if bias == "bearish":
        if iv_high:
            short = _by_delta(calls, TARGET_DELTA)
            long = _by_delta(calls, WING_DELTA)
            if short is not None and long is not None and long["strike"] > short["strike"] \
                    and 0 < ((_num(short.get("price")) or 0) - (_num(long.get("price")) or 0)) < (long["strike"] - short["strike"]):
                credit = (_num(short.get("price")) or 0) - (_num(long.get("price")) or 0)
                width = long["strike"] - short["strike"]
                strategies.append(OptionStrategy(
                    name="Bear Call Spread (credit)",
                    direction="Bearish",
                    legs=[_leg_from(short, "Sell", dte, expiry), _leg_from(long, "Buy", dte, expiry)],
                    tenor_dte=dte,
                    net_debit_credit=round(credit, 2),
                    max_profit=round(credit, 2),
                    max_loss=round(width - credit, 2),
                    breakeven=round(short["strike"] + credit, 2),
                    rationale=(
                        f"IV is elevated — sell premium. Sell the {short['strike']:.0f} call and buy the "
                        f"{long['strike']:.0f} call for ~${credit:.2f} credit. Profit if {name} stays below "
                        f"{short['strike']:.0f}; risk capped."
                    ),
                    suited_when="Bearish/neutral and want to collect premium with defined risk.",
                ))
        else:
            long = _atm(puts, spot)
            short = _by_delta(puts, TARGET_DELTA)
            if long is not None and short is not None and short["strike"] < long["strike"] \
                    and 0 < ((_num(long.get("price")) or 0) - (_num(short.get("price")) or 0)) < (long["strike"] - short["strike"]):
                debit = (_num(long.get("price")) or 0) - (_num(short.get("price")) or 0)
                width = long["strike"] - short["strike"]
                strategies.append(OptionStrategy(
                    name="Put Debit Spread",
                    direction="Bearish",
                    legs=[_leg_from(long, "Buy", dte, expiry), _leg_from(short, "Sell", dte, expiry)],
                    tenor_dte=dte,
                    net_debit_credit=round(-debit, 2),
                    max_loss=round(debit, 2),
                    max_profit=round(width - debit, 2),
                    breakeven=round(long["strike"] - debit, 2),
                    rationale=(
                        f"IV is {'cheap' if iv_low else 'reasonable'} — buy directional downside. Buy the "
                        f"{long['strike']:.0f} put, sell the {short['strike']:.0f} put for ~${debit:.2f}. "
                        f"Defined cost; profit grows down to {short['strike']:.0f}."
                    ),
                    suited_when="Bearish with a downside target near the short strike.",
                ))

    # --- Neutral (no clear directional edge): sell a range if IV is rich,
    # buy volatility if IV is cheap. Also offered when a directional read is
    # low-confidence, so a weak signal doesn't force a directional-only menu.
    if bias == "neutral" or low_conviction:
        if iv_high:
            short_call = _by_delta(calls, CONDOR_SHORT_DELTA)
            long_call = _by_delta(calls, WING_DELTA)
            short_put = _by_delta(puts, CONDOR_SHORT_DELTA)
            long_put = _by_delta(puts, WING_DELTA)
            if (
                short_call is not None and long_call is not None and long_call["strike"] > short_call["strike"]
                and short_put is not None and long_put is not None and long_put["strike"] < short_put["strike"]
            ):
                call_credit = (_num(short_call.get("price")) or 0) - (_num(long_call.get("price")) or 0)
                put_credit = (_num(short_put.get("price")) or 0) - (_num(long_put.get("price")) or 0)
                credit = call_credit + put_credit
                call_width = long_call["strike"] - short_call["strike"]
                put_width = short_put["strike"] - long_put["strike"]
            else:
                credit, call_width, put_width = 0.0, 0.0, 0.0
            # guard against degenerate/stale mid quotes: a "credit" condor whose
            # credit is non-positive (or exceeds the wider wing) is a data
            # artifact, not a trade
            if 0 < credit < max(call_width, put_width) and earnings_within_tenor:
                out.notes.append(
                    f"Iron Condor deliberately NOT offered: earnings ({out.earnings_date}) land "
                    f"inside this tenor, and selling a price range through a binary event is a "
                    f"convention-breaking trade regardless of how rich IV looks."
                )
            elif 0 < credit < max(call_width, put_width):
                strategies.append(OptionStrategy(
                    name="Iron Condor",
                    direction="Neutral",
                    legs=[
                        _leg_from(short_put, "Sell", dte, expiry), _leg_from(long_put, "Buy", dte, expiry),
                        _leg_from(short_call, "Sell", dte, expiry), _leg_from(long_call, "Buy", dte, expiry),
                    ],
                    tenor_dte=dte,
                    net_debit_credit=round(credit, 2),
                    max_profit=round(credit, 2),
                    max_loss=round(max(call_width, put_width) - credit, 2),
                    rationale=(
                        f"IV is elevated and the read is neutral — sell both wings. Sell the "
                        f"{short_put['strike']:.0f}/{short_call['strike']:.0f} strangle, buy the "
                        f"{long_put['strike']:.0f}/{long_call['strike']:.0f} wings for protection, for "
                        f"~${credit:.2f} credit. Profit if {name} stays between {short_put['strike']:.0f} "
                        f"and {short_call['strike']:.0f} at expiry; risk capped both sides."
                    ),
                    suited_when="Neutral view, IV rich, and you want a defined-risk range trade.",
                ))
        else:
            long_call = _atm(calls, spot)
            long_put = _atm(puts, spot)
            if (
                long_call is not None and long_put is not None
                and (_num(long_call.get("price")) or 0) > 0 and (_num(long_put.get("price")) or 0) > 0
            ):
                debit = (_num(long_call.get("price")) or 0) + (_num(long_put.get("price")) or 0)
                strategies.append(OptionStrategy(
                    name="Long Straddle",
                    direction="Neutral",
                    legs=[_leg_from(long_call, "Buy", dte, expiry), _leg_from(long_put, "Buy", dte, expiry)],
                    tenor_dte=dte,
                    net_debit_credit=round(-debit, 2),
                    max_loss=round(debit, 2),
                    rationale=(
                        f"IV is {'cheap' if iv_low else 'reasonable'} and the read is neutral, but a big move "
                        f"looks plausible — buy the {long_call['strike']:.0f} call and put for ~${debit:.2f} "
                        f"total. Profit outside roughly {long_call['strike'] - debit:.2f}-"
                        f"{long_call['strike'] + debit:.2f} at expiry; direction doesn't matter, magnitude does."
                    ),
                    suited_when=(
                        "Expecting a large move (earnings, catalyst) but unsure which way — "
                        "best when IV is CHEAP going in, so you're not overpaying for the move."
                    ),
                ))

    # Attach profit-taking / stop / roll management, net Greeks, probability of
    # profit, event/liquidity warnings, and risk-budgeted sizing to every structure.
    for s in strategies:
        _attach_management(s, dte)
        _attach_risk_metrics(s, spot, dte, out.atm_iv_pct)
        _attach_warnings(s, earnings_within_tenor, out.earnings_date, days_to_earnings)
        _attach_sizing(s, shares, book_value_usd, risk_budget_frac)

    if not strategies:
        out.notes.append(
            "No high-conviction options structure: the signal is mixed or strikes/Greeks were unavailable. "
            "Holding stock (or staying flat) is reasonable here."
        )
    out.strategies = strategies

    # Topside / downside context for conviction (technical levels + analysts).
    if stock_target or stock_stop:
        parts = []
        if stock_target:
            parts.append(f"technical target ~{stock_target:.2f} ({(stock_target/spot-1)*100:+.1f}% topside)")
        if stock_stop:
            parts.append(f"stop ~{stock_stop:.2f} ({(stock_stop/spot-1)*100:+.1f}%)")
        out.notes.append("Underlying levels: " + ", ".join(parts) + ".")
    if analyst:
        out.notes.append(
            f"Analyst consensus: {analyst.get('label')} "
            f"({analyst.get('strong_buy',0)+analyst.get('buy',0)} buy / {analyst.get('hold',0)} hold / "
            f"{analyst.get('sell',0)+analyst.get('strong_sell',0)} sell of {analyst.get('total',0)})."
        )
    out.notes.append(
        f"Tenor chosen: {dte} days (expiry {expiry}). Premiums use the live bid/ask midpoint "
        f"where a two-sided quote exists (falls back to last trade on thin contracts); per share "
        f"(×100 per contract)."
    )
    return out


def _attach_management(s: OptionStrategy, dte: int) -> None:
    """Fill take-profit / stop-loss / roll guidance per structure type (standard
    options-desk management: ~50% profit on credit, ~21-DTE roll, etc.)."""
    is_credit = (s.net_debit_credit or 0) > 0
    roll_dte = max(7, min(21, dte // 2))
    if s.name == "Covered Call":
        s.take_profit = "Buy back the call near ~50-65% of the credit (let theta do the work)."
        s.stop_loss = "No hard stop — it's covered; your risk is the stock falling. Watch the underlying's stop level."
        s.manage = (f"If price pushes through the strike, roll the call up-and-out (~{roll_dte} DTE) to defer "
                    f"assignment and keep collecting premium — or let the shares be called away for the gain.")
    elif s.name == "Iron Condor":
        s.take_profit = "Close at ~50% of max credit — don't hold defined-risk condors into the final week."
        s.stop_loss = "Exit if either short strike is breached on a closing basis, or the loss hits ~2× the credit received."
        s.manage = (f"If one side is tested, consider rolling that side out (and further away) around {roll_dte} DTE; "
                    f"the untested side can often be closed early to reduce risk.")
    elif s.name == "Collar":
        s.take_profit = ("If the stock rallies toward the call strike, the collar did its job — either "
                         "let the shares be called away or roll the whole collar up.")
        s.stop_loss = ("None needed — the long put IS the stop: your worst case is selling at the put "
                       "strike, known in advance.")
        s.manage = (f"Around {roll_dte} DTE, roll both legs out (and re-center around the new price) if "
                    f"you still want protection; let the collar expire if the risk has passed.")
    elif s.name == "Long Straddle":
        s.take_profit = "Take profits into a sharp move rather than waiting for max theoretical gain — theta decay works against you."
        s.stop_loss = "Cut if IV crushes post-event with no move, or at ~50% of the debit paid."
        s.manage = "Once the move happens, consider selling against it (into a strangle) to harvest the remaining premium instead of holding to expiry."
    elif is_credit:  # CSP, bull put, bear call
        s.take_profit = "Close at ~50% of max profit (buy the spread back for ~half the credit)."
        s.stop_loss = "Exit if the loss hits ~2× the credit received, or the short strike is breached on a closing basis."
        s.manage = (f"Roll out (and away from price) around {roll_dte} DTE if still tested, to reset theta; "
                    f"don't carry defined-risk credit spreads into expiry week (gamma/pin risk).")
    else:  # debit spreads
        s.take_profit = "Take ~50-75% of max profit rather than holding for the last few %."
        s.stop_loss = "Cut at ~50% of the debit paid, or if the underlying's stop/invalidation level breaks."
        s.manage = (f"Roll up-and-out to lock gains if the target is hit early; avoid the final week "
                    f"(gamma decay) — close or roll by ~{roll_dte} DTE.")


def _attach_risk_metrics(s: OptionStrategy, spot: float, dte: int, atm_iv_pct: float | None) -> None:
    """Fill net position Greeks (native BSM -- the broker only supplies a
    per-leg delta, nothing else), probability of profit, and expected value
    per share (both from the generic payoff-curve engine, so they work for any
    leg combination without per-strategy-type special-casing).

    Covered Call and Collar are the structures whose P&L includes the
    underlying: their payoff/Greeks get a synthetic long-stock component (per
    share, basis = spot), otherwise they'd be scored as naked option legs --
    wrong sign on delta and a wrong profit region entirely."""
    includes_stock = s.name in _STOCK_INCLUSIVE_NAMES

    rate = options_math.RISK_FREE_RATE_PCT / 100.0
    net_delta = net_theta = net_vega = 0.0
    have_greeks = False
    for leg in s.legs:
        g = options_math.bsm_greeks(spot, leg.strike, leg.iv_pct, leg.dte or dte, leg.right, rate=rate)
        if g["delta"] is None:
            continue
        have_greeks = True
        sign = 1.0 if leg.action == "Buy" else -1.0
        net_delta += sign * g["delta"]
        net_theta += sign * g["theta"]
        net_vega += sign * g["vega"]
    if have_greeks:
        if includes_stock:
            net_delta += 1.0   # long stock: delta 1 per share, no theta/vega
        s.net_delta = round(net_delta, 3)
        s.net_theta = round(net_theta, 3)
        s.net_vega = round(net_vega, 3)

    # POP/EV are premium-sensitive: with any leg missing a price the payoff
    # curve is fiction, so report nothing rather than a confident wrong number.
    if any(leg.price is None for leg in s.legs):
        return
    grid = options_math.default_price_grid(spot)
    payoff = options_math.payoff_at_expiry(s.legs, grid)
    if includes_stock:
        payoff = payoff + (grid - spot)   # long stock P&L per share
    tenor = s.tenor_dte or dte
    pop = options_math.probability_of_profit(spot, atm_iv_pct, tenor, payoff, grid)
    if pop is not None:
        s.pop_pct = round(pop, 1)
    ev = options_math.expected_pnl(spot, atm_iv_pct, tenor, payoff, grid)
    if ev is not None:
        s.ev_per_share = round(ev, 2)


def _skew_25d(calls: pd.DataFrame, puts: pd.DataFrame) -> float | None:
    """25-delta risk reversal read: (25Δ put IV) - (25Δ call IV), in vol points.
    Positive = puts richer than calls (the usual equity 'crash premium');
    strongly negative = call skew (upside speculation bid). None if either
    25-delta wing has no usable IV."""
    p = _by_delta(puts, SKEW_DELTA)
    c = _by_delta(calls, SKEW_DELTA)
    if p is None or c is None:
        return None
    put_iv, call_iv = _num(p.get("iv")), _num(c.get("iv"))
    if put_iv is None or call_iv is None or put_iv <= 0 or call_iv <= 0:
        return None
    return float(put_iv - call_iv)


# Structures whose P&L includes the held shares (get a synthetic stock leg in
# payoff/Greeks, and are sized by shares held rather than the risk budget).
_STOCK_INCLUSIVE_NAMES = {"Covered Call", "Collar"}

# Structures that SELL net premium -- these are the ones an earnings event
# inside the tenor works against (IV crush is your friend, but the gap risk of
# a binary move dwarfs the credit). Debit/long-vol structures are the opposite.
_SHORT_PREMIUM_NAMES = {
    "Covered Call", "Cash-Secured Put", "Bull Put Spread (credit)",
    "Bear Call Spread (credit)", "Iron Condor",
}


def _attach_warnings(
    s: OptionStrategy, earnings_within_tenor: bool, earnings_date: str | None,
    days_to_earnings: int | None,
) -> None:
    """Per-structure event-risk + liquidity flags. These populate `s.warnings`
    (shown as ⚠ chips) -- distinct from the always-present management text."""
    w: list[str] = []

    # Earnings inside the tenor: direction depends on whether you're long or
    # short premium. Short-premium sellers face a binary gap; long-vol buyers
    # are paying inflated IV that will crush regardless of the move.
    if earnings_within_tenor and earnings_date:
        if s.name == "Collar":
            w.append(
                f"Earnings {earnings_date} (in {days_to_earnings}d) fall inside this tenor — which is "
                f"exactly what a collar is for: the put floor holds through the report. Expect the "
                f"protection to look 'expensive' pre-event (IV is inflated on both legs, so the short "
                f"call offsets much of it)."
            )
        elif s.name in _SHORT_PREMIUM_NAMES:
            w.append(
                f"Earnings {earnings_date} (in {days_to_earnings}d) fall inside this tenor — "
                f"you're SHORT premium into a binary event; a gap through your short strike "
                f"can exceed the credit. Prefer an expiry BEFORE earnings, or size for the gap."
            )
        else:  # long straddle / debit spreads
            w.append(
                f"Earnings {earnings_date} (in {days_to_earnings}d) fall inside this tenor — "
                f"IV is inflated and will crush post-report; the underlying must move MORE than "
                f"the priced-in expected move just to break even on the vega bleed."
            )

    # Liquidity: wide bid/ask or thin OI on any leg makes fills expensive and
    # exits unreliable -- the difference between a paper edge and a real one.
    wide = [l for l in s.legs if _spread_pct(l) is not None and _spread_pct(l) > LIQ_MAX_SPREAD_PCT]
    thin = [l for l in s.legs if l.oi is not None and l.oi < LIQ_MIN_OI]
    if wide:
        worst = max(_spread_pct(l) for l in wide)
        w.append(
            f"Wide markets: {len(wide)} leg(s) have a bid/ask spread over {LIQ_MAX_SPREAD_PCT:.0f}% "
            f"of mid (worst ~{worst:.0f}%) — use limit orders near mid; the modelled premium is "
            f"optimistic versus a real fill."
        )
    if thin:
        w.append(
            f"Thin open interest: {len(thin)} leg(s) below {LIQ_MIN_OI} contracts — harder to exit "
            f"without slippage; verify a two-sided market before trading."
        )
    s.warnings = w


def _spread_pct(leg: OptionLeg) -> float | None:
    if leg.bid is None or leg.ask is None or leg.bid <= 0 or leg.ask <= 0 or leg.ask < leg.bid:
        return None
    mid = (leg.bid + leg.ask) / 2.0
    return (leg.ask - leg.bid) / mid * 100.0 if mid > 0 else None


def _attach_sizing(
    s: OptionStrategy, shares: float, book_value_usd: float | None, risk_budget_frac: float,
) -> None:
    """Size the position so its worst-case dollar loss fits a per-trade risk
    budget (default 1% of book). Turns an abstract 'max loss $3.80/share' into
    'buy N contracts, risking $X' — the last mile between analysis and a
    decision. Covered Call and Collar are special-cased: contracts are capped
    by shares held (you can't write/hedge more than you can cover)."""
    if book_value_usd is None or book_value_usd <= 0 or s.max_loss is None or s.max_loss <= 0:
        return
    budget = book_value_usd * risk_budget_frac
    loss_per_contract = s.max_loss * CONTRACT_SIZE
    if loss_per_contract <= 0:
        return
    n = int(budget // loss_per_contract)

    if s.name in _STOCK_INCLUSIVE_NAMES:
        coverable = int(shares // CONTRACT_SIZE)
        n = coverable if coverable > 0 else 0

    if n < 1:
        s.warnings = [*s.warnings, (
            f"Even one contract risks more than your {risk_budget_frac:.0%} per-trade budget "
            f"(~${budget:,.0f}) — this structure is too large for the account at this size."
        )]
        return
    s.suggested_contracts = n
    # Capital required: for stock-inclusive structures the shares are already
    # owned, so incremental capital is just the net option cost (0 if a credit);
    # CSP posts the full strike as collateral; long structures pay the debit;
    # defined-risk spreads post the width (max loss) as margin.
    if s.name in _STOCK_INCLUSIVE_NAMES:
        s.capital_required_usd = round(max(0.0, -(s.net_debit_credit or 0.0)) * CONTRACT_SIZE * n, 2)
    elif s.name == "Cash-Secured Put" and s.legs:
        s.capital_required_usd = round(s.legs[0].strike * CONTRACT_SIZE * n, 2)
    elif (s.net_debit_credit or 0) < 0:
        s.capital_required_usd = round(abs(s.net_debit_credit) * CONTRACT_SIZE * n, 2)
    else:
        s.capital_required_usd = round(loss_per_contract * n, 2)
