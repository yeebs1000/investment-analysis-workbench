"""TechnicalAnalyst: from OHLCV bars to a structured, number-backed signal.

Five weighted dimensions (trend / momentum / volatility / volume / levels) each
produce a score in [-1, +1] plus plain-English, number-tagged reasons. The blend
yields a 0-100 score, a discrete Decision, and a confidence. No LLM involved —
this is the deterministic source of truth.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from app.analytics import indicators as ind
from app.analytics.scoring import blend, clamp, score_to_decision
from app.data.models import MLSignal, SignalComponent, TechnicalAnalysis

# Six weighted dimensions. "Quant" carries the systematic models that quant/CTA
# desks lean on (time-series momentum, MA-cross trend filter, Donchian breakout,
# Connors RSI2 mean-reversion) plus risk-adjusted context (Sharpe, drawdown).
#
# "ml" is the optional 7th component (see `ml_signal` param below). It stays 0.0
# until a human reviews `data_store/reports/calibration_*.md` and raises it --
# never auto-applied. Raised to 0.05 after the 2026-07-07 run: the h20 model
# passed gating (OOS AUC 0.53, IC 0.10, CI clears 0.50, stable across
# sub-periods) on a 232-name universe. Kept small -- reliability is ~0.49
# ("tentative"), so it lightly nudges rather than drives the score. blend()
# normalizes by the weight sum, so no need to rescale the others.
WEIGHTS = {
    "trend": 0.24,
    "momentum": 0.20,
    "quant": 0.23,
    "volume": 0.13,
    "volatility": 0.10,
    "levels": 0.10,
    "ml": 0.05,
}

# Kelly win-probability slope (see the Kelly sizing block near the end of
# analyze()). Recalibrated 2026-07-02 from data_store/reports/calibration_2026-07-02.md
# (python -m app.ml.train, 50-symbol universe, 6 purged walk-forward folds,
# Aug 2023-Jun 2026): both the 10-bar and 20-bar horizons independently
# estimated slope ~0.2-0.36 vs. the prior hardcoded 0.7 -- the old value was
# sizing positions ~2-3x too aggressively relative to the observed
# score-to-hit-rate relationship. Set to 0.3, near the low end of that range
# (conservative default; the ML gate itself did NOT activate -- see the report
# for caveats on regime narrowness / survivorship bias before trusting this
# too far either).
KELLY_SLOPE = 0.3

MIN_BARS = 30

# Entry-risk ("too late to chase") thresholds. All ATR-relative so a volatile
# small cap and a quiet megacap are judged by their OWN normal swing size, and
# keyed on VELOCITY (how far price traveled recently) plus STRETCH (how far it
# sits from its 20-EMA) — never on proximity to highs, which the momentum
# components correctly reward. Symmetric: a parabolic ramp flags chase risk on
# entries, a capitulation flush flags panic-sell risk on exits.
ENTRY_RISK_LOOKBACK = 10                 # bars over which the move is measured
ENTRY_RISK_MOVE_ATR_CAUTION = 4.0        # >= this many ATRs in the lookback -> caution
ENTRY_RISK_MOVE_ATR_HIGH = 5.5           # >= this -> high (parabolic/capitulation)
ENTRY_RISK_STRETCH_CAUTION = 2.0         # ATRs from the 20-EMA (same sign as move)
ENTRY_RISK_STRETCH_HIGH = 2.5
ENTRY_RISK_GAP_ATR = 1.5                 # an open gap this large marks an event-driven move


def trend_score(bars: pd.DataFrame) -> tuple[float, str]:
    """Lightweight directional read used for higher-timeframe confirmation.

    Returns (score in [-1, 1], plain summary). Based on price vs EMA20/50, the
    20/50 stack, and EMA20 slope — cheap enough to run on a second timeframe.
    """
    if bars is None or len(bars) < 20:
        return 0.0, "not enough higher-timeframe history"
    close = bars["close"]
    ema20, ema50 = ind.ema(close, 20), ind.ema(close, 50)
    price, e20, e50 = _last(close), _last(ema20), _last(ema50)
    parts: list[float] = []
    if price is not None and e20 is not None:
        parts.append(0.35 if price > e20 else -0.35)
    if e20 is not None and e50 is not None:
        parts.append(0.35 if e20 > e50 else -0.35)
    slope = ind.slope_pct(ema20, 5)
    parts.append(clamp(slope / 4.0))
    s = clamp(sum(parts))
    word = "uptrend" if s > 0.15 else "downtrend" if s < -0.15 else "sideways / no clear trend"
    return s, word


def assess_entry_risk(
    bars: pd.DataFrame,
    price: float | None,
    v_atr: float | None,
    v_ema20: float | None,
    v_rsi: float | None,
) -> dict | None:
    """Categorical "is this a bad moment to act" read, separate from the score.

    Returns None (no flag) for anything inside normal behaviour. Fires only on
    velocity+stretch extremes: a parabolic ramp (chase risk for buyers) or a
    capitulation flush (panic-sell risk for holders). Deliberately NOT blended
    into the 0-100 score — a stock can be a great business ripping on real news
    and still be a poor entry TODAY; conflating those two reads is how tools
    end up recommending tops. The service layer refines `advice` with earnings
    attribution and whether the user actually holds the name.
    """
    n = len(bars)
    lb = ENTRY_RISK_LOOKBACK
    if price is None or not v_atr or v_atr <= 0 or v_ema20 is None or n < lb + 5:
        return None
    close, open_ = bars["close"], bars["open"]
    ref = _last(close.iloc[: n - lb])          # close `lb` bars ago
    if ref is None:
        return None
    move_atr = (price - ref) / v_atr           # ATRs traveled over the lookback
    stretch = (price - v_ema20) / v_atr        # ATRs away from the 20-EMA
    up = move_atr > 0

    level = None
    if move_atr >= ENTRY_RISK_MOVE_ATR_HIGH and stretch >= ENTRY_RISK_STRETCH_HIGH:
        level = "high"
    elif move_atr >= ENTRY_RISK_MOVE_ATR_CAUTION and stretch >= ENTRY_RISK_STRETCH_CAUTION:
        level = "caution"
    elif move_atr <= -ENTRY_RISK_MOVE_ATR_HIGH and stretch <= -ENTRY_RISK_STRETCH_HIGH:
        level = "high"
    elif move_atr <= -ENTRY_RISK_MOVE_ATR_CAUTION and stretch <= -ENTRY_RISK_STRETCH_CAUTION:
        level = "caution"
    if level is None:
        return None

    # Event attribution input: did the move start with a large same-direction
    # open gap inside the lookback? (Earnings/news moves gap; narrative chases
    # mostly grind.) Each gap is judged against the ATR *at that bar* — the
    # end-of-window ATR is already inflated by the burst itself and would
    # under-detect a gap that launched it. The service layer pairs this with
    # the EPS-surprise data.
    event_gap = False
    atr_series = ind.atr(bars["high"], bars["low"], close, 14)
    for o, pc, a in zip(open_.tail(lb), close.shift(1).tail(lb),
                        atr_series.shift(1).tail(lb)):
        if pd.isna(o) or pd.isna(pc) or pd.isna(a) or a <= 0:
            continue
        gap = float(o) - float(pc)
        if abs(gap) >= ENTRY_RISK_GAP_ATR * float(a) and (gap > 0) == up:
            event_gap = True
            break

    pct = (price - ref) / abs(ref) * 100.0 if ref else 0.0
    reasons = [
        f"{pct:+.1f}% in {lb} bars ({move_atr:+.1f}× ATR)",
        f"price {stretch:+.1f} ATR from its 20-EMA",
    ]
    if v_rsi is not None and (v_rsi >= 78 or v_rsi <= 22):
        reasons.append(f"RSI {v_rsi:.0f}")
    if event_gap:
        reasons.append("move started with a large open gap (event-driven)")

    if up:
        label = "Parabolic — chase risk" if level == "high" else "Extended — chase risk"
        advice = (f"Stretched entry: if buying, stage in or wait for a pullback "
                  f"toward the 20-EMA (~{v_ema20:,.2f}) rather than chasing.")
    else:
        label = ("Capitulation flush — panic-sell risk" if level == "high"
                 else "Sharp flush — panic-sell risk")
        advice = ("Selling into a flush locks in the worst prints: wait for "
                  "stabilization, and let the stop level — not the panic — decide the exit.")

    return {
        "level": level,
        "direction": "up" if up else "down",
        "label": label,
        "move_atr_10": round(move_atr, 1),
        "stretch_atr": round(stretch, 1),
        "event_gap": event_gap,
        "attribution": None,      # service layer sets "earnings" when supported
        "reasons": reasons,
        "advice": advice,
    }


def _last(series: pd.Series, default: float | None = None) -> float | None:
    s = series.dropna()
    if s.empty:
        return default
    v = float(s.iloc[-1])
    return default if math.isnan(v) else v


def _pct(a: float | None, b: float | None) -> float | None:
    if a is None or b is None or b == 0:
        return None
    return (a - b) / abs(b) * 100.0


def _fmt(x: float | None, dp: int = 2) -> str:
    return "n/a" if x is None else f"{x:,.{dp}f}"


def _tag(text: str, score: float) -> str:
    mark = "+" if score > 0.05 else "-" if score < -0.05 else "~"
    return f"{text} ({mark})"


def analyze(
    code: str,
    name: str,
    bars: pd.DataFrame,
    snapshot: dict | None = None,
    htf: dict | None = None,
    ppy: float = 252.0,
    analyst: dict | None = None,
    bench: dict | None = None,
    ml_signal: MLSignal | None = None,
) -> TechnicalAnalysis:
    """Run the full technical read on an ascending OHLCV frame.

    `htf` (optional) carries a higher-timeframe trend read
    {"label": str, "score": float, "summary": str} used for confirmation.
    `ppy` is bars-per-year for the timeframe (annualizes Sharpe/vol/return).
    `analyst` (optional) is the Finnhub consensus dict (adds conviction).
    `bench` (optional) {"rel_strength_pct", "beta", "alpha_pct"} vs a benchmark.
    `ml_signal` (optional) is the trained model's read from `app.ml.inference`
    -- appended as a 7th component (weight 0.0 until a human raises it, see
    WEIGHTS above) and fed into confidence at a reliability-discounted rate.
    """
    snapshot = snapshot or {}
    n = len(bars)
    if n < MIN_BARS:
        return TechnicalAnalysis(
            code=code, name=name, score=50.0,
            decision=score_to_decision(50.0), confidence=0.0,
            confidence_label="Low", bars_used=n,
            error=f"Insufficient history ({n} bars, need >= {MIN_BARS}).",
        )

    close, high, low, vol = bars["close"], bars["high"], bars["low"], bars["volume"]
    price = _last(close)
    as_of = str(bars.index[-1].date()) if hasattr(bars.index[-1], "date") else str(bars.index[-1])

    # --- indicator series ---
    ema20, ema50, ema200 = ind.ema(close, 20), ind.ema(close, 50), ind.ema(close, 200)
    rsi = ind.rsi(close, 14)
    macd_line, macd_sig, macd_hist = ind.macd(close)
    _mid, bb_up, bb_low, pct_b, bandwidth = ind.bollinger(close, 20, 2)
    atr = ind.atr(high, low, close, 14)
    adx_, plus_di, minus_di = ind.adx(high, low, close, 14)
    obv = ind.obv(close, vol)
    stoch_k, stoch_d = ind.stochastic(high, low, close)
    vwap = ind.rolling_vwap(high, low, close, vol, 20)

    v_ema20, v_ema50, v_ema200 = _last(ema20), _last(ema50), _last(ema200)
    v_rsi = _last(rsi)
    v_macd, v_sig, v_hist = _last(macd_line), _last(macd_sig), _last(macd_hist)
    v_pctb, v_bw = _last(pct_b), _last(bandwidth)
    v_atr = _last(atr)
    v_adx, v_pdi, v_mdi = _last(adx_), _last(plus_di), _last(minus_di)
    v_k, v_d = _last(stoch_k), _last(stoch_d)
    v_vwap = _last(vwap)
    vol_avg20 = _last(vol.rolling(20).mean())
    v_vol = _last(vol)
    vol_ratio = (v_vol / vol_avg20) if (v_vol and vol_avg20) else None
    ema20_slope = ind.slope_pct(ema20, 5)
    rsi_slope = ind.slope_pct(rsi, 3)
    roc20 = ind.roc(close, 20)
    div_dir, div_msg = ind.divergence(close, rsi, lookback=40)

    # --- quant-model inputs (systematic / risk-adjusted) ---
    sma50, sma200 = ind.sma(close, 50), ind.sma(close, 200)
    v_sma50, v_sma200 = _last(sma50), _last(sma200)
    rsi2 = ind.rsi(close, 2)
    v_rsi2 = _last(rsi2)
    don_n = min(55, n - 1)
    don_up_s, don_lo_s = ind.donchian(high, low, don_n)
    v_don_up, v_don_lo = _last(don_up_s), _last(don_lo_s)
    tsm_n = min(252, n - 1)                 # ~12-month horizon on daily bars
    tsmom = ind.roc(close, tsm_n)           # time-series (absolute) momentum
    sharpe_v = ind.sharpe(close, ppy)
    maxdd = ind.max_drawdown_pct(close, lookback=min(252, n))
    annvol = ind.ann_vol_pct(close, ppy)
    annret = ind.ann_return_pct(close, ppy)

    components: list[SignalComponent] = []

    # ============ TREND ============
    parts, reasons = [], []
    if price is not None and v_ema20 is not None:
        s = 0.25 if price > v_ema20 else -0.25
        parts.append(s)
        reasons.append(_tag(f"Price {_fmt(_pct(price, v_ema20))}% vs 20-day EMA", s))
    if price is not None and v_ema50 is not None:
        s = 0.25 if price > v_ema50 else -0.25
        parts.append(s)
        reasons.append(_tag(f"Price {_fmt(_pct(price, v_ema50))}% vs 50-day EMA", s))
    if price is not None and v_ema200 is not None:
        s = 0.25 if price > v_ema200 else -0.25
        parts.append(s)
        reasons.append(_tag(f"Price {_fmt(_pct(price, v_ema200))}% vs 200-day EMA (long-term)", s))
    if v_ema20 is not None and v_ema50 is not None:
        s = 0.25 if v_ema20 > v_ema50 else -0.25
        parts.append(s)
        reasons.append(_tag("20-EMA above 50-EMA — uptrend structure" if s > 0
                            else "20-EMA below 50-EMA — downtrend structure", s))
    if abs(ema20_slope) > 0.05:
        s = clamp(ema20_slope / 4.0)
        parts.append(s * 0.5)
        reasons.append(_tag(
            f"20-day EMA slope {_fmt(ema20_slope,2)}% — "
            f"{'rising' if ema20_slope > 0 else 'falling'}", s))
    trend_raw = float(np.clip(sum(parts), -1, 1)) if parts else 0.0
    # Scale by trend strength (ADX): choppy/weak trends carry less conviction.
    if v_adx is not None:
        if v_adx < 18:
            trend_raw *= 0.6
            reasons.append(_tag(f"ADX {_fmt(v_adx,1)} — weak/!trending (low conviction)", 0.0))
        elif v_adx >= 25:
            di_aligns = (v_pdi or 0) > (v_mdi or 0)
            reasons.append(_tag(
                f"ADX {_fmt(v_adx,1)} — strong trend, "
                f"{'+DI > -DI' if di_aligns else '-DI > +DI'}",
                0.3 if di_aligns else -0.3))
    # Higher-timeframe confirmation: trade with the larger trend, flag fighting it.
    if htf and abs(htf.get("score", 0.0)) > 0.1:
        hs = float(htf["score"])
        aligned = (hs > 0) == (trend_raw > 0) if trend_raw != 0 else None
        trend_raw = clamp(trend_raw + (0.15 if aligned else -0.15)) if aligned is not None else trend_raw
        verb = "confirms" if aligned else "conflicts with" if aligned is False else "frames"
        reasons.append(_tag(
            f"Higher timeframe ({htf.get('label','?')}) is in a {htf.get('summary','?')} "
            f"— {verb} this read", hs))
    components.append(SignalComponent(
        name="Trend", score=clamp(trend_raw), weight=WEIGHTS["trend"],
        summary=_trend_summary(trend_raw, v_adx), reasons=reasons,
        metrics={"ema20": v_ema20, "ema50": v_ema50, "ema200": v_ema200, "adx": v_adx},
    ))

    # ============ MOMENTUM ============
    parts, reasons = [], []
    if v_rsi is not None:
        s = clamp((v_rsi - 50.0) / 20.0)
        parts.append(s)
        note = "overbought" if v_rsi >= 70 else "oversold" if v_rsi <= 30 else \
               "bullish momentum" if v_rsi > 50 else "bearish momentum"
        reasons.append(_tag(f"RSI {_fmt(v_rsi,1)} — {note}", s))
    if v_macd is not None and v_sig is not None:
        cross = 0.4 if v_macd > v_sig else -0.4
        rising = ind.slope_pct(macd_hist, 3)
        s = clamp(cross + (0.2 if rising > 0 else -0.2))
        parts.append(s)
        reasons.append(_tag(
            f"MACD {'above' if v_macd > v_sig else 'below'} signal, histogram "
            f"{'expanding' if rising > 0 else 'contracting'}", s))
    if v_k is not None and v_d is not None:
        s = clamp((v_k - 50.0) / 50.0 * 0.6 + (0.2 if v_k > v_d else -0.2))
        parts.append(s)
        extreme = " (overbought)" if v_k >= 80 else " (oversold)" if v_k <= 20 else ""
        reasons.append(_tag(f"Stochastic %K {_fmt(v_k,0)}{extreme}", s))
    if v_macd is not None:
        s = 0.15 if v_macd > 0 else -0.15
        parts.append(s)
        reasons.append(_tag(
            f"MACD line {'above' if v_macd > 0 else 'below'} zero — "
            f"{'bullish' if v_macd > 0 else 'bearish'} regime", s))
    if abs(rsi_slope) > 1.0:
        s = clamp(rsi_slope / 15.0)
        parts.append(s * 0.5)
        reasons.append(_tag(f"RSI {'rising' if rsi_slope > 0 else 'falling'} "
                            f"({_fmt(rsi_slope,1)}% over 3 bars)", s))
    if div_dir != 0:
        s = 0.5 * div_dir
        parts.append(s)
        reasons.append(_tag(div_msg, s))
    if roc20 is not None:
        reasons.append(_tag(f"Price {_fmt(roc20,1)}% over last 20 bars", clamp(roc20 / 20.0)))
    mom_raw = float(np.mean(parts)) if parts else 0.0
    components.append(SignalComponent(
        name="Momentum", score=clamp(mom_raw), weight=WEIGHTS["momentum"],
        summary=_generic_summary(mom_raw, "momentum"), reasons=reasons,
        metrics={"rsi": v_rsi, "macd": v_macd, "macd_signal": v_sig,
                 "macd_hist": v_hist, "stoch_k": v_k, "stoch_d": v_d},
    ))

    # ============ VOLATILITY ============
    parts, reasons = [], []
    if v_pctb is not None:
        if v_pctb > 1.0:
            s = 0.2
            reasons.append(_tag(f"%B {_fmt(v_pctb)} — above upper band, stretched", s))
        elif v_pctb < 0.0:
            s = -0.2
            reasons.append(_tag(f"%B {_fmt(v_pctb)} — below lower band, oversold", s))
        else:
            s = clamp((v_pctb - 0.5) * 1.4)
            reasons.append(_tag(f"%B {_fmt(v_pctb)} — "
                                f"{'upper' if v_pctb > 0.5 else 'lower'} half of Bollinger range", s))
        parts.append(s)
    atr_pct = (v_atr / price * 100.0) if (v_atr and price) else None
    if v_bw is not None:
        bw_series = bandwidth.dropna().tail(126)
        if len(bw_series) > 20 and v_bw <= bw_series.quantile(0.15):
            reasons.append(_tag("Bollinger bandwidth near 6-month low — squeeze, breakout setup", 0.0))
    vol_raw = float(np.mean(parts)) if parts else 0.0
    components.append(SignalComponent(
        name="Volatility", score=clamp(vol_raw), weight=WEIGHTS["volatility"],
        summary=_generic_summary(vol_raw, "volatility position"), reasons=reasons,
        metrics={"pct_b": v_pctb, "bandwidth": v_bw, "atr": v_atr, "atr_pct": atr_pct},
    ))

    # ============ VOLUME ============
    parts, reasons = [], []
    obv_slope = ind.slope_pct(obv, 10)
    if abs(obv_slope) > 0:
        s = 0.5 if obv_slope > 0 else -0.5
        parts.append(s)
        reasons.append(_tag(f"OBV {'rising' if obv_slope > 0 else 'falling'} over 10 days — "
                            f"{'accumulation' if obv_slope > 0 else 'distribution'}", s))
    if price is not None and v_vwap is not None:
        s = 0.3 if price > v_vwap else -0.3
        parts.append(s)
        reasons.append(_tag(f"Price {'above' if price > v_vwap else 'below'} 20-day VWAP", s))
    if vol_ratio is not None:
        if vol_ratio >= 1.2:
            reasons.append(_tag(f"Volume {_fmt(vol_ratio)}× 20-day average — strong participation", 0.2))
        elif vol_ratio <= 0.7:
            reasons.append(_tag(f"Volume {_fmt(vol_ratio)}× 20-day average — light participation", 0.0))
    # Volume-by-price: a heavy point-of-control is a defended shelf. Price above a
    # high-volume node = support beneath (bullish); below it = overhead supply.
    poc_price, poc_share = ind.volume_profile_poc(high, low, close, vol, lookback=min(120, n))
    if poc_price is not None and poc_share is not None and price is not None and poc_share >= 0.10:
        near = abs(price - poc_price) / price <= 0.02
        if near:
            s = 0.0
            reasons.append(_tag(f"At a high-volume node ~{_fmt(poc_price)} ({_fmt(poc_share*100,0)}% of volume) — pivotal level", s))
        else:
            s = 0.25 if price > poc_price else -0.25
            parts.append(s)
            reasons.append(_tag(
                f"Price {'above' if s>0 else 'below'} the volume point-of-control ~{_fmt(poc_price)} "
                f"({_fmt(poc_share*100,0)}% of volume) — {'support below' if s>0 else 'overhead supply'}", s))
    volm_raw = float(np.mean(parts)) if parts else 0.0
    # Amplify conviction when a move is backed by above-average volume.
    if vol_ratio is not None and vol_ratio >= 1.2:
        volm_raw = clamp(volm_raw * 1.25)
    components.append(SignalComponent(
        name="Volume", score=clamp(volm_raw), weight=WEIGHTS["volume"],
        summary=_generic_summary(volm_raw, "volume flow"), reasons=reasons,
        metrics={"obv_slope_pct": obv_slope, "vwap": v_vwap, "volume_ratio": vol_ratio},
    ))

    # ============ LEVELS (support / resistance) ============
    parts, reasons = [], []
    win = min(20, n - 1)
    high20 = float(high.tail(win).max())
    low20 = float(low.tail(win).min())
    rng = high20 - low20
    if rng > 0 and price is not None:
        pos = (price - low20) / rng  # 0 at support, 1 at resistance
        s = clamp((pos - 0.5) * 1.6)
        parts.append(s)
        reasons.append(_tag(
            f"In {win}-day range: support ~{_fmt(low20)}, resistance ~{_fmt(high20)} "
            f"(at {_fmt(pos*100,0)}% of range)", s))
    hi52 = _f(snapshot.get("highest52weeks_price"))
    lo52 = _f(snapshot.get("lowest52weeks_price"))
    if hi52 and price:
        d = _pct(price, hi52)
        reasons.append(_tag(f"{_fmt(d)}% from 52-week high", clamp((d or 0) / 20.0)))
    lvl_raw = float(np.mean(parts)) if parts else 0.0
    components.append(SignalComponent(
        name="Levels", score=clamp(lvl_raw), weight=WEIGHTS["levels"],
        summary=_generic_summary(lvl_raw, "position in range"), reasons=reasons,
        metrics={"support": low20, "resistance": high20, "high_52w": hi52, "low_52w": lo52},
    ))

    # ============ QUANT MODELS ============
    # Systematic signals used by quant/CTA desks, each in [-1, 1].
    parts, reasons = [], []
    # 1) Time-series (absolute) momentum — AQR/Moskowitz: long-horizon return sign.
    if tsmom is not None:
        s = clamp(tsmom / 25.0)
        parts.append(s)
        reasons.append(_tag(f"{tsm_n}-bar time-series momentum {_fmt(tsmom,1)}%", s))
    # 2) Trend filter — golden/death cross (50 vs 200 SMA).
    if v_sma50 is not None and v_sma200 is not None:
        s = 0.5 if v_sma50 > v_sma200 else -0.5
        parts.append(s)
        reasons.append(_tag(
            "Golden cross: 50-SMA above 200-SMA" if s > 0
            else "Death cross: 50-SMA below 200-SMA", s))
    # 3) Donchian breakout (Turtle trend-following).
    if price is not None and v_don_up and v_don_lo and v_don_up > v_don_lo:
        if price >= v_don_up * 0.999:
            s = 0.6
            reasons.append(_tag(f"Donchian breakout: at/above {don_n}-bar high {_fmt(v_don_up)}", s))
        elif price <= v_don_lo * 1.001:
            s = -0.6
            reasons.append(_tag(f"Donchian breakdown: at/below {don_n}-bar low {_fmt(v_don_lo)}", s))
        else:
            pos = (price - v_don_lo) / (v_don_up - v_don_lo)
            s = clamp((pos - 0.5) * 1.2)
            reasons.append(_tag(f"Inside {don_n}-bar Donchian channel ({_fmt(pos*100,0)}% of range)", s))
        parts.append(s)
    # 4) Connors RSI(2) mean-reversion, filtered by the 200-SMA trend.
    if v_rsi2 is not None and v_sma200 is not None and price is not None:
        if price > v_sma200 and v_rsi2 < 10:
            s = 0.5
            parts.append(s)
            reasons.append(_tag(f"RSI(2) {_fmt(v_rsi2,0)} oversold in an uptrend — Connors long setup", s))
        elif price < v_sma200 and v_rsi2 > 90:
            s = -0.5
            parts.append(s)
            reasons.append(_tag(f"RSI(2) {_fmt(v_rsi2,0)} overbought in a downtrend — mean-reversion short", s))
        elif v_rsi2 >= 98:
            s = -0.2
            parts.append(s)
            reasons.append(_tag(f"RSI(2) {_fmt(v_rsi2,0)} extremely overbought — near-term pullback risk", s))
        elif v_rsi2 <= 2:
            s = 0.2
            parts.append(s)
            reasons.append(_tag(f"RSI(2) {_fmt(v_rsi2,0)} extremely oversold — near-term bounce potential", s))
    # Risk-adjusted context (informs conviction, not direction).
    if sharpe_v is not None:
        reasons.append(_tag(f"Sharpe {_fmt(sharpe_v,2)} annualized (risk-adjusted return)", clamp(sharpe_v / 2.0)))
    if maxdd is not None:
        reasons.append(_tag(f"Max drawdown {_fmt(maxdd,1)}% over window", 0.0))
    quant_raw = float(np.mean(parts)) if parts else 0.0
    components.append(SignalComponent(
        name="Quant", score=clamp(quant_raw), weight=WEIGHTS["quant"],
        summary=_generic_summary(quant_raw, "systematic models"), reasons=reasons,
        metrics={"ts_momentum_pct": tsmom, "sma50": v_sma50, "sma200": v_sma200,
                 "rsi2": v_rsi2, "donchian_high": v_don_up, "donchian_low": v_don_lo,
                 "sharpe": sharpe_v, "max_drawdown_pct": maxdd,
                 "ann_vol_pct": annvol, "ann_return_pct": annret},
    ))

    # Institutional + relative-strength context (conviction, not raw direction).
    analyst_score = float(analyst["score"]) if analyst else None
    if analyst:
        buys = analyst.get("strong_buy", 0) + analyst.get("buy", 0)
        sells = analyst.get("sell", 0) + analyst.get("strong_sell", 0)
        components[-1].reasons.append(_tag(
            f"Analyst consensus {analyst.get('label')}: {buys} buy / {analyst.get('hold',0)} hold / "
            f"{sells} sell of {analyst.get('total',0)} analysts", analyst_score or 0.0))
    if bench and bench.get("rel_strength_pct") is not None:
        rs = bench["rel_strength_pct"]
        components[-1].reasons.append(_tag(
            f"Relative strength {_fmt(rs,1)}% vs benchmark over window", clamp(rs / 20.0)))

    # --- blend with real-world corroboration signals ---
    # NOTE: computed over the original 6 components, BEFORE the optional ML
    # component is appended below. This keeps `extra_confirms` (an
    # un-discounted confidence bump) from double-counting the ML read -- its
    # ONLY path into confidence is the dedicated, reliability-discounted
    # `ml_signal`/`ml_reliability` params on blend(), same cap as analyst_score.
    total_w = sum(c.weight for c in components) or 1.0
    prelim_net = sum(c.score * c.weight for c in components) / total_w
    htf_aligned: bool | None = None
    if htf and abs(htf.get("score", 0.0)) > 0.1 and abs(prelim_net) > 0.05:
        htf_aligned = (float(htf["score"]) > 0) == (prelim_net > 0)
    vol_confirm = bool(vol_ratio and vol_ratio >= 1.2)
    # count independent signals pointing the same way as the net read
    confirms = sum(1 for c in components
                   if c.score != 0 and (c.score > 0) == (prelim_net > 0) and abs(c.score) >= 0.25)
    if div_dir != 0 and (div_dir > 0) == (prelim_net > 0):
        confirms += 1

    # Optional 7th component: the trained ML forecast (see app/ml/). Weight is
    # 0.0 until a human raises it after reviewing a calibration report, but the
    # read is visible in `reasons` as soon as a model exists -- transparency
    # without borrowed conviction. Appended AFTER prelim_net/confirms above so
    # it can never move the score or the un-discounted confirms count.
    if ml_signal is not None:
        components.append(SignalComponent(
            name="ML Forecast", score=clamp(ml_signal.score), weight=WEIGHTS.get("ml", 0.0),
            summary=_generic_summary(ml_signal.score, "ML forecast"), reasons=list(ml_signal.reasons),
            metrics={"probability": ml_signal.probability, "reliability": ml_signal.reliability},
        ))

    score, confidence, conf_label = blend(
        components, htf_aligned=htf_aligned, adx=v_adx,
        vol_confirm=vol_confirm, extra_confirms=confirms, sharpe=sharpe_v,
        analyst_score=analyst_score,
        ml_signal=(ml_signal.score if ml_signal is not None else None),
        ml_reliability=(ml_signal.reliability if ml_signal is not None else 0.0),
    )
    decision = score_to_decision(score)

    # Aggregate the strongest reasons (largest |contribution|) for a quick read.
    ranked = sorted(components, key=lambda c: abs(c.score) * c.weight, reverse=True)
    top_reasons: list[str] = []
    for c in ranked:
        if c.reasons:
            top_reasons.append(c.reasons[0])
    top_reasons = top_reasons[:5]

    # Structure-aware stop/target (long-framed): anchored to the 20-day
    # support/resistance the Levels analyst already computed, with ATR bounds —
    # a stop is never noise-tight (< 1 ATR below) nor unbounded (> 3 ATR), a
    # target never trivially close (< 1.5 ATR) nor fantasy-far (> 4.5 ATR).
    # This makes reward:risk vary with actual chart structure; the old fixed
    # price±(2,3)×ATR multiples collapsed R:R to a constant 1.5 for every
    # stock, which also fed Kelly a constant payoff ratio.
    stop = target = None
    if price and v_atr and v_atr > 0:
        stop_structural = (low20 - 0.5 * v_atr) if rng > 0 else (price - 2.0 * v_atr)
        stop = max(min(stop_structural, price - 1.0 * v_atr), price - 3.0 * v_atr)
        near_breakout = rng > 0 and price >= high20 - 0.25 * v_atr
        if near_breakout:
            # at/near the range high, the range top is no target — look to the
            # 52-week high if it's meaningfully above, else a plain ATR multiple
            target_structural = hi52 if (hi52 and hi52 > price + 1.5 * v_atr) else price + 3.0 * v_atr
        else:
            target_structural = high20 if rng > 0 else (price + 3.0 * v_atr)
        target = min(max(target_structural, price + 1.5 * v_atr), price + 4.5 * v_atr)
        stop, target = round(stop, 2), round(target, 2)

    # Categorical entry-risk flag (chase / capitulation), separate from the score.
    entry_risk = assess_entry_risk(bars, price, v_atr, v_ema20, v_rsi)

    # Reward:risk and Kelly sizing (half-Kelly, capped). Win-prob is proxied from
    # the score and nudged by analyst consensus; payoff ratio from target/stop.
    reward_risk = None
    if price and stop and target and price > stop:
        reward_risk = round((target - price) / (price - stop), 2)
    kelly_f = kelly_sizing = None
    if reward_risk and reward_risk > 0:
        p = 0.5 + (score - 50.0) / 100.0 * KELLY_SLOPE
        if analyst_score is not None:
            p += analyst_score * 0.05
        p = max(0.10, min(0.85, p))
        kelly_f = round(p - (1.0 - p) / reward_risk, 3)
        kelly_sizing = round(min(max(0.0, kelly_f) / 2.0 * 100.0, 15.0), 1)  # half-Kelly, 15% cap

    return TechnicalAnalysis(
        code=code, name=name, as_of=as_of, price=price,
        score=score, decision=decision, confidence=confidence,
        confidence_label=conf_label,
        higher_tf=(htf or {}).get("label"),
        higher_tf_trend=_r((htf or {}).get("score")) if htf else None,
        higher_tf_summary=(htf or {}).get("summary"),
        components=components, reasons=top_reasons,
        stop=stop, target=target, atr_pct=round(atr_pct, 2) if atr_pct else None,
        reward_risk=reward_risk, kelly_fraction=kelly_f, kelly_sizing_pct=kelly_sizing,
        entry_risk=entry_risk,
        analyst_consensus=analyst,
        rel_strength_pct=_r((bench or {}).get("rel_strength_pct")),
        beta=_r((bench or {}).get("beta")), alpha_pct=_r((bench or {}).get("alpha_pct")),
        indicators={
            "rsi": _r(v_rsi), "rsi_slope_pct": _r(rsi_slope), "macd": _r(v_macd),
            "macd_signal": _r(v_sig), "macd_hist": _r(v_hist), "ema20": _r(v_ema20),
            "ema20_slope_pct": _r(ema20_slope), "ema50": _r(v_ema50), "ema200": _r(v_ema200),
            "adx": _r(v_adx), "plus_di": _r(v_pdi), "minus_di": _r(v_mdi),
            "stoch_k": _r(v_k), "stoch_d": _r(v_d), "pct_b": _r(v_pctb), "atr": _r(v_atr),
            "vwap": _r(v_vwap), "volume_ratio": _r(vol_ratio), "roc_20bar_pct": _r(roc20),
            "rsi_divergence": float(div_dir), "support_20d": _r(low20), "resistance_20d": _r(high20),
            "sma50": _r(v_sma50), "sma200": _r(v_sma200), "rsi2": _r(v_rsi2),
            "ts_momentum_pct": _r(tsmom), "donchian_high": _r(v_don_up), "donchian_low": _r(v_don_lo),
            "sharpe": _r(sharpe_v), "max_drawdown_pct": _r(maxdd),
            "ann_vol_pct": _r(annvol), "ann_return_pct": _r(annret),
        },
        bars_used=n,
    )


# --- small helpers -------------------------------------------------------
def _r(x: float | None, dp: int = 2) -> float | None:
    return None if x is None else round(float(x), dp)


def _f(value) -> float | None:
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _trend_summary(score: float, adx: float | None) -> str:
    strength = "strong " if (adx or 0) >= 25 else "weak " if (adx or 99) < 18 else ""
    if score > 0.15:
        return f"{strength}uptrend".capitalize()
    if score < -0.15:
        return f"{strength}downtrend".capitalize()
    return "Sideways / no clear trend"


def _generic_summary(score: float, what: str) -> str:
    if score > 0.15:
        return f"Bullish {what}"
    if score < -0.15:
        return f"Bearish {what}"
    return f"Neutral {what}"
