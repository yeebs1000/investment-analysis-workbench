"""Turn weighted analyst components into a transparent score, decision, and
confidence. Kept separate so the decision logic is easy to read and tune."""
from __future__ import annotations

from app.data.models import Decision, SignalComponent


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def blend(
    components: list[SignalComponent],
    htf_aligned: bool | None = None,
    adx: float | None = None,
    vol_confirm: bool = False,
    extra_confirms: int = 0,
    sharpe: float | None = None,
    analyst_score: float | None = None,
    ml_signal: float | None = None,
    ml_reliability: float = 0.0,
) -> tuple[float, float, str]:
    """Return (score_0_100, confidence_0_1, confidence_label).

    Score is the weighted mean of component scores ([-1,1]) mapped onto 0-100.
    Confidence ("conviction") reflects not just internal agreement but real-world
    corroboration: a strong, trending setup that the higher timeframe and volume
    both confirm earns high conviction; a weak, mixed, counter-trend read does not.
    """
    if not components:
        return 50.0, 0.0, "Low"

    total_w = sum(c.weight for c in components) or 1.0
    net = sum(c.score * c.weight for c in components) / total_w  # [-1, 1]
    score = round((net + 1.0) * 50.0, 1)

    # Agreement: fraction of weight whose sign matches the net direction.
    if net == 0:
        agreement = 0.0
    else:
        aligned = sum(c.weight for c in components if (c.score > 0) == (net > 0) and c.score != 0)
        agreement = aligned / total_w

    # Base conviction: signal strength + internal agreement.
    conf = abs(net) * 0.50 + agreement * 0.25
    # Trend quality: a strong ADX means the directional read is more trustworthy.
    if adx is not None:
        conf += min(adx, 40.0) / 40.0 * 0.12
    # Higher-timeframe corroboration is the single biggest conviction driver.
    if htf_aligned is True:
        conf += 0.13
    elif htf_aligned is False:
        conf -= 0.10
    # Volume confirmation and the number of independent confirming signals.
    if vol_confirm:
        conf += 0.05
    conf += min(max(extra_confirms, 0), 4) * 0.0125
    # Risk-adjusted corroboration: a meaningful Sharpe that agrees with the
    # directional read adds conviction (the trend has actually been "paying").
    if sharpe is not None and net != 0 and (sharpe > 0) == (net > 0):
        conf += min(abs(sharpe), 2.0) / 2.0 * 0.07
    # Institutional corroboration: analyst consensus agreeing with the read.
    if analyst_score is not None and net != 0 and (analyst_score > 0) == (net > 0):
        conf += min(abs(analyst_score), 1.0) * 0.08
    # ML forecast corroboration -- discounted by `ml_reliability` (sample-size/
    # fold-count derived at training time, see app/ml/train.py), so a thin or
    # noisy model can't move confidence much even with an extreme point score.
    # Same cap as the analyst term above: a maximally-reliable ML read can't
    # dominate over the deterministic engine's own agreement/trend/htf signals.
    if ml_signal is not None and net != 0 and (ml_signal > 0) == (net > 0):
        conf += min(abs(ml_signal), 1.0) * clamp(ml_reliability, 0.0, 1.0) * 0.08

    confidence = round(clamp(conf, 0.0, 1.0), 2)
    label = "High" if confidence >= 0.66 else "Medium" if confidence >= 0.4 else "Low"
    return score, confidence, label


def two_axis_verdict(
    quality: dict | None,
    score: float,
    entry_risk: dict | None,
    stop: float | None = None,
    ema20: float | None = None,
) -> dict | None:
    """Combine the two lenses the app already keeps deliberately separate —
    business quality (fundamental_quality) and entry timing (technical score +
    entry_risk) — into one stated quadrant, WITHOUT blending the numbers.

    "Is it good" and "should you act here" are independent questions: a great
    business can be a terrible entry (parabolic), a weak business a fine trade.
    Returns None when the quality axis is unknown (no honest quadrant exists).
    """
    if not quality or quality.get("score_0_100") is None:
        return None
    q_label = str(quality.get("label", ""))
    good = q_label in ("High quality", "Solid")

    if entry_risk and entry_risk.get("direction") == "up":
        timing = "extended"
    elif entry_risk and entry_risk.get("direction") == "down":
        timing = "flush"
    elif score >= 60:
        timing = "favorable"
    elif score >= 46:
        timing = "neutral"
    else:
        timing = "weak"

    lvl = f"~{ema20:,.2f}" if ema20 else "support"
    stp = f"{stop:,.2f}" if stop else "the suggested stop"
    if good:
        guidance = {
            "favorable": "Good business and a supportive tape — the two lenses agree.",
            "neutral": "Good business, tape undecided — a patient accumulation zone, no urgency either way.",
            "weak": (f"Good business in a weak tape — for a long-term holder this argues for "
                     f"patience, not a panic exit; {stp} is the invalidation."),
            "extended": (f"Good business, stretched entry — wait for a pullback toward "
                         f"{lvl} rather than chasing."),
            "flush": "Good business in a capitulation flush — the worst moment to sell; wait for stabilization.",
        }[timing]
    else:
        guidance = {
            "favorable": (f"Tape is strong but business quality is {q_label.lower()} — treat it as "
                          f"a trade with the stop at {stp}, not a core holding."),
            "neutral": "Neither the business nor the tape is compelling — capital likely works harder elsewhere.",
            "weak": "Weak business and a weak tape — neither lens supports owning this.",
            "extended": f"Extended tape on a {q_label.lower()}-quality business — chasing here has the worst of both.",
            "flush": f"Falling hard and quality is {q_label.lower()} — any bounce is a trade at best, not an investment.",
        }[timing]

    return {
        "quality_axis": q_label,
        "timing_axis": timing,
        "quadrant": f"{q_label} × {timing} timing",
        "guidance": guidance,
    }


def score_to_decision(score: float) -> Decision:
    if score >= 72:
        return Decision.STRONG_BUY
    if score >= 60:
        return Decision.BUY
    if score >= 54:
        return Decision.ACCUMULATE
    if score >= 46:
        return Decision.HOLD
    if score >= 36:
        return Decision.REDUCE
    return Decision.SELL
