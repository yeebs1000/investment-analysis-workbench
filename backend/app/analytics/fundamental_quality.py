"""Fundamental quality score — a Buffett/Munger-style read on business quality
from the curated Finnhub metrics (profitability, returns on capital, growth,
financial strength).

DELIBERATELY SEPARATE from the technical score and the ML signal:
- It is computed from *current* fundamentals (point-in-time now). That's fine
  for scoring a live decision, but it would be lookahead-leakage to feed into
  the walk-forward-trained ML on historical rows -- so it never touches
  app/ml/. It is a standalone quality lens, shown alongside the technical read.
- It answers a different question on a different horizon: technicals =
  "how is this trading over days/weeks", quality = "is this a good business to
  own for years". A long-term holder wants both, kept distinct, not blended
  into one number that hides which lens is talking.

Every threshold here is a named constant, and every metric that's missing is
reported (never silently treated as zero) so the score's coverage is honest.
"""
from __future__ import annotations

# (metric_key, weight, [(threshold, points)] descending) -- points in [0,1].
# Weights sum to 1.0 across metrics we have; missing metrics are renormalized
# out so a US name with full coverage and a thin-coverage name are both scored
# on whatever IS available, with coverage disclosed.
_RUBRIC: dict[str, tuple[float, list[tuple[float, float]]]] = {
    # Returns on capital -- the core Buffett quality signal (durable moat proxy).
    "roe_pct":       (0.22, [(20, 1.0), (15, 0.8), (10, 0.55), (5, 0.3), (0, 0.1)]),
    "roic_pct":      (0.20, [(15, 1.0), (10, 0.8), (7, 0.55), (3, 0.3), (0, 0.1)]),
    # Profitability / pricing power.
    "net_margin_pct":   (0.15, [(20, 1.0), (12, 0.8), (7, 0.55), (3, 0.3), (0, 0.1)]),
    "gross_margin_pct": (0.10, [(50, 1.0), (40, 0.8), (30, 0.55), (20, 0.3), (0, 0.1)]),
    # Growth.
    "revenue_growth_yoy_pct": (0.15, [(15, 1.0), (8, 0.8), (3, 0.55), (0, 0.3), (-100, 0.05)]),
    # Financial strength (downside protection). Note: LOWER debt is better, so
    # these rubrics are keyed on the inverse in _score_metric below.
    "debt_to_equity":  (0.10, [(0.3, 1.0), (0.6, 0.8), (1.0, 0.55), (2.0, 0.3), (999, 0.05)]),
    "current_ratio":   (0.08, [(2.0, 1.0), (1.5, 0.8), (1.2, 0.6), (1.0, 0.4), (0, 0.1)]),
}
# Metrics where a LOWER value scores higher (thresholds are ascending ceilings).
_LOWER_IS_BETTER = {"debt_to_equity"}

STRONG_QUALITY = 70.0    # >= this -> "High quality"
DECENT_QUALITY = 50.0    # >= this -> "Solid"
WEAK_QUALITY = 35.0      # >= this -> "Mixed"; below -> "Weak"
MIN_METRICS_FOR_SCORE = 3   # below this, coverage too thin for a meaningful score


def _score_metric(key: str, value: float) -> float:
    """Map a raw metric to [0,1] via its rubric."""
    _w, rubric = _RUBRIC[key]
    if key in _LOWER_IS_BETTER:
        # ascending ceilings: first threshold the value is <= wins
        for ceiling, pts in rubric:
            if value <= ceiling:
                return pts
        return rubric[-1][1]
    # descending floors: first threshold the value is >= wins
    for floor, pts in rubric:
        if value >= floor:
            return pts
    return rubric[-1][1]


def score_quality(fundamentals: dict | None) -> dict | None:
    """Return {score_0_100, label, coverage, reasons, missing} or None if there
    isn't enough coverage to say anything honest."""
    if not fundamentals:
        return None
    present: list[tuple[str, float, float, float]] = []  # key, value, sub[0,1], weight
    missing: list[str] = []
    for key, (weight, _rubric) in _RUBRIC.items():
        val = fundamentals.get(key)
        if isinstance(val, (int, float)):
            present.append((key, float(val), _score_metric(key, float(val)), weight))
        else:
            missing.append(key)

    if len(present) < MIN_METRICS_FOR_SCORE:
        return None

    total_w = sum(w for *_rest, w in present)
    score01 = sum(sub * w for _k, _v, sub, w in present) / total_w
    score = round(score01 * 100.0, 1)
    label = (
        "High quality" if score >= STRONG_QUALITY
        else "Solid" if score >= DECENT_QUALITY
        else "Mixed" if score >= WEAK_QUALITY
        else "Weak"
    )

    # Human-readable drivers: the 3 strongest and any genuine red flag.
    ranked = sorted(present, key=lambda t: t[2], reverse=True)
    reasons: list[str] = []
    for key, val, sub, _w in ranked[:3]:
        reasons.append(f"{_pretty(key)} {_fmt_val(key, val)} ({_grade(sub)})")
    flags = [t for t in present if t[2] <= 0.3]
    for key, val, _sub, _w in flags:
        note = f"⚠ {_pretty(key)} {_fmt_val(key, val)}"
        if note not in reasons:
            reasons.append(note)

    return {
        "score_0_100": score,
        "label": label,
        "coverage": f"{len(present)}/{len(_RUBRIC)} metrics",
        "reasons": reasons,
        "missing": missing,
    }


# --- size / growth-stage tilt -------------------------------------------------
# A conviction TILT (not a quality score): where a name sits on the
# size-vs-growth map. Same honesty bar as score_quality -- computed from CURRENT
# fundamentals for a live decision, never fed into the historical ML (leak).
# Thesis (user-directed): a mega-cap is unlikely to double, so underweight it; a
# small-cap still in its growth phase (and, as a price proxy, on a roll vs the
# market) has the most room to compound, so overweight it hardest.
MEGA_CAP_MUSD = 200_000    # >= ~$200B: law of large numbers bites hardest
LARGE_CAP_MUSD = 50_000    # >= ~$50B
MID_CAP_MUSD = 10_000      # >= ~$10B
SMALL_CAP_MUSD = 2_000     # >= ~$2B; below this = micro/nano
HIGH_GROWTH_YOY = 20.0     # revenue YoY % that counts as growth-stage
HOT_REL_STRENGTH = 10.0    # relative strength vs SPY % to count as "on a roll"


def size_growth_tilt(fundamentals: dict | None, rel_strength_pct: float | None = None) -> dict | None:
    """A [-1, +1] conviction tilt from size + growth stage + price momentum, with
    a suggested position-size multiplier and human reasons. None if there's no
    market cap to anchor on (degrade honestly, don't guess size)."""
    if not fundamentals:
        return None
    mcap = fundamentals.get("market_cap_musd")
    if not isinstance(mcap, (int, float)) or mcap <= 0:
        return None
    mcap = float(mcap)
    growth = fundamentals.get("revenue_growth_yoy_pct")
    growth = float(growth) if isinstance(growth, (int, float)) else None

    reasons: list[str] = []
    # 1. size base tilt
    if mcap >= MEGA_CAP_MUSD:
        tilt, size_class = -0.6, "mega-cap"
        reasons.append(f"Mega-cap (~${mcap/1000:.0f}B) — law of large numbers caps upside; "
                       f"doubling from here is a multi-trillion move.")
    elif mcap >= LARGE_CAP_MUSD:
        tilt, size_class = -0.3, "large-cap"
        reasons.append(f"Large-cap (~${mcap/1000:.0f}B) — mature; unlikely to beat the market by a wide margin.")
    elif mcap >= MID_CAP_MUSD:
        tilt, size_class = 0.0, "mid-cap"
    elif mcap >= SMALL_CAP_MUSD:
        tilt, size_class = 0.3, "small-cap"
        reasons.append(f"Small-cap (~${mcap/1000:.1f}B) — far more room to compound than a mega-cap.")
    else:
        tilt, size_class = 0.25, "micro-cap"
        reasons.append(f"Micro-cap (~${mcap:.0f}M) — high growth potential, but thinner and more volatile.")

    small_ish = mcap < MID_CAP_MUSD
    # 2. growth-stage boost -- small + high-growth is the strongest-overweight bucket
    if growth is not None:
        if growth >= HIGH_GROWTH_YOY and small_ish:
            tilt += 0.35
            reasons.append(f"Growth-stage: revenue +{growth:.0f}% YoY on a small base — the strongest overweight.")
        elif growth >= HIGH_GROWTH_YOY:
            tilt += 0.10
            reasons.append(f"High revenue growth (+{growth:.0f}% YoY), though size limits the runway.")
        elif growth < 0 and not small_ish:
            tilt -= 0.20
            reasons.append(f"Revenue shrinking ({growth:.0f}% YoY) at scale — value-trap risk.")

    # 3. "hot" / on-a-roll -- price-driven proxy: relative strength vs the market
    if rel_strength_pct is not None and rel_strength_pct >= HOT_REL_STRENGTH:
        tilt += 0.15
        reasons.append(f"On a roll: +{rel_strength_pct:.0f}% relative strength vs the market lately.")

    tilt = max(-1.0, min(1.0, tilt))
    label = (
        "Strong overweight" if tilt >= 0.5 else
        "Overweight" if tilt >= 0.2 else
        "Neutral" if tilt > -0.2 else
        "Underweight" if tilt > -0.5 else
        "Strong underweight"
    )
    return {
        "tilt": round(tilt, 2),
        "label": label,
        "size_class": size_class,
        # strong tilt scales the suggested position ~0.4x (mega) .. 1.6x (small grower)
        "sizing_multiplier": round(1.0 + 0.6 * tilt, 2),
        "reasons": reasons,
    }


def _pretty(key: str) -> str:
    return {
        "roe_pct": "ROE", "roic_pct": "ROIC", "net_margin_pct": "Net margin",
        "gross_margin_pct": "Gross margin", "revenue_growth_yoy_pct": "Rev growth",
        "debt_to_equity": "Debt/equity", "current_ratio": "Current ratio",
    }.get(key, key)


def _fmt_val(key: str, val: float) -> str:
    if key in ("debt_to_equity", "current_ratio"):
        return f"{val:.2f}"
    return f"{val:.0f}%"


def _grade(sub: float) -> str:
    return "excellent" if sub >= 0.9 else "good" if sub >= 0.7 else "ok" if sub >= 0.45 else "weak"
