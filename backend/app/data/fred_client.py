"""FRED (Federal Reserve Economic Data) provider, read-only: a small set of
macro series that define the market *regime* — the yield-curve slope, a
high-yield credit spread, and the VIX.

Purpose is NOT stock prediction. It's a regime gate: a book arguably shouldn't
be sized identically in a calm, upward-sloping-curve regime and an inverted-
curve, blown-out-spreads, high-VIX one. This module reports the regime; callers
decide how (or whether) to lean on it.

Free API key (settings.fred_api_key). Absent or erroring -> every function
returns None and the feature degrades cleanly, exactly like the LLM/Finnhub
layers. All calls short-cached.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

from app.config import settings

_BASE = "https://api.stlouisfed.org/fred"
_TIMEOUT = 12
_TTL = 3600.0   # macro series update daily at most; an hour is plenty
_cache: dict[str, tuple[float, object]] = {}

# The series we read, and the regime thresholds. Sourced from standard macro
# convention, each a named constant so the regime logic is auditable.
SERIES = {
    "t10y2y": "T10Y2Y",     # 10yr minus 2yr Treasury (yield-curve slope), percent
    "hy_oas": "BAMLH0A0HYM2",  # ICE BofA US High Yield option-adjusted spread, percent
    "vix": "VIXCLS",        # CBOE VIX close
}
CURVE_INVERTED_BELOW = 0.0      # 10y-2y < 0 -> inverted (classic recession lead)
HY_SPREAD_STRESS_ABOVE = 5.0    # HY OAS > 5% -> credit stress
HY_SPREAD_CALM_BELOW = 3.5      # HY OAS < 3.5% -> credit calm
VIX_ELEVATED_ABOVE = 22.0       # VIX > 22 -> elevated fear
VIX_CALM_BELOW = 15.0           # VIX < 15 -> complacent/calm


def available() -> bool:
    return bool(settings.fred_api_key)


def _latest(series_id: str) -> float | None:
    """Most recent non-missing observation for a FRED series."""
    if not available():
        return None
    params = {
        "series_id": series_id,
        "api_key": settings.fred_api_key,
        "file_type": "json",
        "sort_order": "desc",
        "limit": 5,   # a few, in case the very latest is a "." placeholder
    }
    url = f"{_BASE}/series/observations?{urllib.parse.urlencode(params)}"
    now = time.time()
    hit = _cache.get(url)
    if hit and now - hit[0] < _TTL:
        return hit[1]  # type: ignore[return-value]
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return None
    val: float | None = None
    for obs in (data.get("observations") or []):
        raw = obs.get("value")
        try:
            val = float(raw)
            break
        except (TypeError, ValueError):
            continue  # "." = missing; try the next-most-recent
    _cache[url] = (now, val)
    return val


def macro_regime() -> dict | None:
    """Current macro regime read, or None if FRED is unavailable / all series
    failed. Returns the raw levels plus a synthesized regime label and the
    risk-posture implication."""
    if not available():
        return None
    curve = _latest(SERIES["t10y2y"])
    hy = _latest(SERIES["hy_oas"])
    vix = _latest(SERIES["vix"])
    if curve is None and hy is None and vix is None:
        return None

    flags: list[str] = []
    stress_score = 0   # 0 = calm, higher = more stressed
    if curve is not None and curve < CURVE_INVERTED_BELOW:
        flags.append(f"yield curve inverted ({curve:.2f}pp)")
        stress_score += 1
    if hy is not None:
        if hy > HY_SPREAD_STRESS_ABOVE:
            flags.append(f"credit spreads wide ({hy:.1f}%)")
            stress_score += 1
        elif hy < HY_SPREAD_CALM_BELOW:
            flags.append(f"credit spreads calm ({hy:.1f}%)")
    if vix is not None:
        if vix > VIX_ELEVATED_ABOVE:
            flags.append(f"VIX elevated ({vix:.0f})")
            stress_score += 1
        elif vix < VIX_CALM_BELOW:
            flags.append(f"VIX calm ({vix:.0f})")

    regime = "risk-off" if stress_score >= 2 else "cautious" if stress_score == 1 else "risk-on"
    implication = {
        "risk-off": "Multiple stress signals — favour smaller sizing, defined risk, and cash buffer.",
        "cautious": "One stress signal — normal sizing, but keep the invalidation levels tight.",
        "risk-on": "No major macro stress signals — the environment supports normal risk-taking.",
    }[regime]

    return {
        "regime": regime,
        "curve_10y2y": round(curve, 2) if curve is not None else None,
        "hy_spread_pct": round(hy, 2) if hy is not None else None,
        "vix": round(vix, 1) if vix is not None else None,
        "flags": flags,
        "implication": implication,
    }
