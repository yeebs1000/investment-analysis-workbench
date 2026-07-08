"""Finnhub data provider (read-only): company/symbol search, analyst
recommendation consensus, and basic fundamentals. The free tier covers
search, quotes, recommendation trends, and `stock/metric` fundamentals;
price targets AND upgrade/downgrade events are premium (403, verified live)
so we use the analyst rating distribution as the institutional-conviction
signal instead.

All calls are short-cached. Network/HTTP errors degrade to None so the core
(Moomoo-driven) analysis never depends on this. Coverage skews heavily to
US-listed names -- HK/SG/JP tickers commonly return partial or no fundamentals;
callers must surface that plainly, never silently substitute or guess.
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request

from app.config import settings

_BASE = "https://finnhub.io/api/v1"
_TIMEOUT = 12
_cache: dict[str, tuple[float, object]] = {}
_TTL = 600.0


def available() -> bool:
    return bool(settings.finnhub_api_key)


def _get(path: str, params: dict) -> object | None:
    if not available():
        return None
    params = {**params, "token": settings.finnhub_api_key}
    url = f"{_BASE}/{path}?{urllib.parse.urlencode(params)}"
    now = time.time()
    hit = _cache.get(url)
    if hit and now - hit[0] < _TTL:
        return hit[1]
    try:
        with urllib.request.urlopen(url, timeout=_TIMEOUT) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:  # noqa: BLE001 - any failure -> treat as no data
        return None
    _cache[url] = (now, data)
    return data


# --- symbol mapping (Finnhub <-> Moomoo "MARKET.CODE") --------------------
_FH_SUFFIX_MARKET = {
    "HK": "HK", "T": "JP", "SS": "SH", "SZ": "SZ", "SI": "SG",
    "L": "UK", "TO": "CA", "AX": "AU",
}
_MOOMOO_FH_SUFFIX = {"HK": "HK", "JP": "T", "SH": "SS", "SZ": "SZ", "SG": "SI"}


def finnhub_to_moomoo(fh_symbol: str) -> str | None:
    """Map a Finnhub symbol (e.g. 'AAPL', '0700.HK', '7203.T') to a Moomoo code."""
    s = (fh_symbol or "").strip().upper()
    if not s:
        return None
    if "." not in s:
        return f"US.{s}"                     # bare ticker -> US
    base, suffix = s.rsplit(".", 1)
    mkt = _FH_SUFFIX_MARKET.get(suffix)
    if not mkt:
        return None                          # market we don't cover
    if mkt == "HK" and base.isdigit():
        base = base.zfill(5)                 # Moomoo HK codes are 5-digit
    return f"{mkt}.{base}"


def moomoo_to_finnhub(code: str) -> str | None:
    """Map a Moomoo 'MARKET.CODE' to a Finnhub symbol for recommendations."""
    if "." not in code:
        return code.upper()
    mkt, sym = code.split(".", 1)
    mkt, sym = mkt.upper(), sym.upper()
    if mkt == "US":
        return sym
    if mkt == "HK":
        return f"{str(int(sym)).zfill(4) if sym.isdigit() else sym}.HK"
    suffix = _MOOMOO_FH_SUFFIX.get(mkt)
    return f"{sym}.{suffix}" if suffix else None


# --- public API -----------------------------------------------------------
def search(query: str, limit: int = 12) -> list[dict]:
    """Resolve a free-text query (ticker or company name) to candidate symbols."""
    data = _get("search", {"q": query})
    out: list[dict] = []
    if not isinstance(data, dict):
        return out
    for r in data.get("result", []):
        fh = r.get("symbol", "")
        code = finnhub_to_moomoo(fh)
        if not code:
            continue
        out.append({
            "code": code,
            "name": r.get("description", ""),
            "finnhub_symbol": fh,
            "type": r.get("type", ""),
        })
        if len(out) >= limit:
            break
    return out


_REC_W = {"strongBuy": 2.0, "buy": 1.0, "hold": 0.0, "sell": -1.0, "strongSell": -2.0}


def recommendation(code: str) -> dict | None:
    """Latest analyst recommendation distribution + a normalized consensus score.

    Returns {strong_buy, buy, hold, sell, strong_sell, total, score_-1_1,
    label, as_of} or None when unavailable.
    """
    fh = moomoo_to_finnhub(code)
    if not fh:
        return None
    data = _get("stock/recommendation", {"symbol": fh})
    if not isinstance(data, list) or not data:
        return None
    latest = data[0]
    counts = {k: int(latest.get(k, 0) or 0) for k in _REC_W}
    total = sum(counts.values())
    if total == 0:
        return None
    weighted = sum(_REC_W[k] * v for k, v in counts.items())
    score = max(-1.0, min(1.0, weighted / (total * 2.0)))
    label = ("Strong Buy" if score >= 0.5 else "Buy" if score >= 0.15
             else "Hold" if score > -0.15 else "Sell" if score > -0.5 else "Strong Sell")
    return {
        "strong_buy": counts["strongBuy"], "buy": counts["buy"], "hold": counts["hold"],
        "sell": counts["sell"], "strong_sell": counts["strongSell"], "total": total,
        "score": round(score, 2), "label": label, "as_of": latest.get("period"),
    }


def next_earnings(code: str, horizon_days: int = 90) -> dict | None:
    """Next confirmed earnings date within `horizon_days`, or None.

    Uses the free-tier /calendar/earnings endpoint. Returns
    {"date": "YYYY-MM-DD", "hour": "bmo"|"amc"|"dmh"|"", "eps_estimate": float|None}
    for the SOONEST upcoming report. None = no earnings found in the window OR
    the data source is unavailable -- callers must treat None as "unknown",
    never as "no earnings coming" (coverage skews US; HK/SG/JP often missing)."""
    import datetime as _dt

    fh = moomoo_to_finnhub(code)
    if not fh:
        return None
    today = _dt.date.today()
    data = _get("calendar/earnings", {
        "symbol": fh,
        "from": today.isoformat(),
        "to": (today + _dt.timedelta(days=horizon_days)).isoformat(),
    })
    if not isinstance(data, dict):
        return None
    rows = data.get("earningsCalendar") or []
    upcoming = sorted(
        (r for r in rows if isinstance(r, dict) and r.get("date") and r["date"] >= today.isoformat()),
        key=lambda r: r["date"],
    )
    if not upcoming:
        return None
    r = upcoming[0]
    eps = r.get("epsEstimate")
    return {
        "date": str(r["date"]),
        "hour": str(r.get("hour") or ""),
        "eps_estimate": float(eps) if isinstance(eps, (int, float)) else None,
    }


def earnings_surprises(code: str, limit: int = 4) -> dict | None:
    """Recent quarterly EPS surprises (actual vs estimate) + a post-earnings-
    announcement-drift (PEAD) read. PEAD is one of the most replicated market
    anomalies: stocks that BEAT tend to drift up for weeks after, missers drift
    down. Returns {beats, misses, avg_surprise_pct, last_surprise_pct,
    last_period, quarters:[...]} or None if unavailable.

    Coverage skews US; returns None (unknown) rather than zero when missing."""
    fh = moomoo_to_finnhub(code)
    if not fh:
        return None
    data = _get("stock/earnings", {"symbol": fh, "limit": limit})
    if not isinstance(data, list) or not data:
        return None
    quarters = []
    surprises_pct = []
    beats = misses = 0
    for q in data:
        if not isinstance(q, dict):
            continue
        actual, est = q.get("actual"), q.get("estimate")
        if not isinstance(actual, (int, float)) or not isinstance(est, (int, float)):
            continue
        # surprise % relative to |estimate|; guard tiny/zero estimates
        surp_pct = ((actual - est) / abs(est) * 100.0) if est not in (0, None) else None
        if surp_pct is not None:
            surprises_pct.append(surp_pct)
            beats += 1 if actual >= est else 0
            misses += 1 if actual < est else 0
        quarters.append({
            "period": q.get("period"),
            "actual": float(actual), "estimate": float(est),
            "surprise_pct": round(surp_pct, 1) if surp_pct is not None else None,
        })
    if not surprises_pct:
        return None
    return {
        "beats": beats, "misses": misses,
        "avg_surprise_pct": round(sum(surprises_pct) / len(surprises_pct), 1),
        "last_surprise_pct": quarters[0]["surprise_pct"] if quarters else None,
        "last_period": quarters[0]["period"] if quarters else None,
        "quarters": quarters,
    }


def insider_sentiment(code: str) -> dict | None:
    """Net insider buying/selling from Finnhub's insider-sentiment endpoint
    (aggregated MSPR + net share change). Modest documented signal: sustained
    net insider BUYING is mildly bullish. Returns {net_mspr, net_change,
    months, direction} or None. MSPR = monthly share purchase ratio, -100..100."""
    fh = moomoo_to_finnhub(code)
    if not fh:
        return None
    import datetime as _dt
    today = _dt.date.today()
    data = _get("stock/insider-sentiment", {
        "symbol": fh,
        "from": (today - _dt.timedelta(days=180)).isoformat(),
        "to": today.isoformat(),
    })
    rows = data.get("data") if isinstance(data, dict) else None
    if not rows:
        return None
    net_mspr = sum(r.get("mspr", 0) or 0 for r in rows if isinstance(r, dict))
    net_change = sum(r.get("change", 0) or 0 for r in rows if isinstance(r, dict))
    if not rows:
        return None
    direction = ("net buying" if net_mspr > 5 else "net selling" if net_mspr < -5 else "neutral")
    return {
        "net_mspr": round(float(net_mspr), 1),
        "net_change": int(net_change),
        "months": len(rows),
        "direction": direction,
    }


# SEC Form 4 transaction codes that are genuine, discretionary open-market
# trades. Excludes 'G' (gift), 'F' (tax withholding), 'M' (option exercise),
# 'A' (grant/award) -- those are administrative, not a signal, and counting
# them as bullish/bearish (a common mistake) would misread routine paperwork
# as conviction.
_DISCRETIONARY_CODES = {"P": "buy", "S": "sell"}


def insider_transactions(code: str, days: int = 180) -> dict | None:
    """Individual insider Form 4 filings, filtered to genuine open-market buys
    /sells (see _DISCRETIONARY_CODES) -- a sharper signal than the aggregated
    MSPR from insider_sentiment(), which can't distinguish a discretionary buy
    from routine option-exercise/tax-withholding activity. Returns
    {open_market_buys, open_market_sells, net_notional_usd, largest_trade,
    window_days} or None if unavailable."""
    fh = moomoo_to_finnhub(code)
    if not fh:
        return None
    import datetime as _dt
    today = _dt.date.today()
    data = _get("stock/insider-transactions", {
        "symbol": fh,
        "from": (today - _dt.timedelta(days=days)).isoformat(),
        "to": today.isoformat(),
    })
    rows = data.get("data") if isinstance(data, dict) else None
    if not rows:
        return None

    buys = sells = 0
    net_notional = 0.0
    largest: dict | None = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        code_txn = r.get("transactionCode")
        side = _DISCRETIONARY_CODES.get(code_txn)
        if side is None:
            continue
        shares = r.get("share")
        price = r.get("transactionPrice")
        if not isinstance(shares, (int, float)):
            continue
        notional = abs(float(shares)) * float(price) if isinstance(price, (int, float)) and price else None
        if side == "buy":
            buys += 1
            net_notional += notional or 0.0
        else:
            sells += 1
            net_notional -= notional or 0.0
        if notional is not None and (largest is None or notional > largest["notional_usd"]):
            largest = {
                "name": r.get("name"), "side": side, "shares": abs(float(shares)),
                "price": float(price), "notional_usd": round(notional, 0),
                "date": r.get("transactionDate"),
            }
    if buys == 0 and sells == 0:
        return None
    return {
        "open_market_buys": buys, "open_market_sells": sells,
        "net_notional_usd": round(net_notional, 0),
        "largest_trade": largest, "window_days": days,
    }


# Curated subset of Finnhub's ~150-field `stock/metric` response -- the fields
# a quality-focused (Buffett/Munger-style) read actually leans on: profitability,
# growth, valuation, financial strength. (metric_key, our_key, round_dp)
_FUNDAMENTAL_FIELDS: list[tuple[str, str, int]] = [
    ("peBasicExclExtraTTM", "pe_ttm", 1),
    ("pbAnnual", "pb", 1),
    ("roeTTM", "roe_pct", 1),
    ("roicTTM", "roic_pct", 1),
    ("grossMarginTTM", "gross_margin_pct", 1),
    ("netProfitMarginTTM", "net_margin_pct", 1),
    ("operatingMarginTTM", "operating_margin_pct", 1),
    ("revenueGrowthTTMYoy", "revenue_growth_yoy_pct", 1),
    ("epsGrowth5Y", "eps_growth_5y_pct", 1),
    ("totalDebt/totalEquityAnnual", "debt_to_equity", 2),
    ("currentRatioAnnual", "current_ratio", 2),
    ("beta", "beta", 2),
    ("52WeekHigh", "week52_high", 2),
    ("52WeekLow", "week52_low", 2),
]


def fundamentals(code: str) -> dict | None:
    """Curated basic fundamentals for a symbol: profitability, growth,
    valuation, financial strength. Returns None if the symbol can't be
    mapped or Finnhub has nothing at all for it; otherwise always returns a
    dict with `available` (fields we got) and `missing` (fields we didn't) so
    callers can state gaps plainly rather than silently degrade."""
    fh = moomoo_to_finnhub(code)
    if not fh:
        return None
    metric_data = _get("stock/metric", {"symbol": fh, "metric": "all"})
    profile = _get("stock/profile2", {"symbol": fh})
    metrics = metric_data.get("metric") if isinstance(metric_data, dict) else None
    if not metrics and not (isinstance(profile, dict) and profile):
        return None  # Finnhub has nothing at all for this symbol

    out: dict = {"finnhub_symbol": fh}
    if isinstance(profile, dict):
        out["name"] = profile.get("name")
        out["sector"] = profile.get("finnhubIndustry")
        out["market_cap_musd"] = profile.get("marketCapitalization")
        out["exchange"] = profile.get("exchange")

    available_fields, missing_fields = [], []
    for src_key, out_key, dp in _FUNDAMENTAL_FIELDS:
        val = (metrics or {}).get(src_key)
        if val is None or not isinstance(val, (int, float)):
            missing_fields.append(out_key)
            continue
        out[out_key] = round(float(val), dp)
        available_fields.append(out_key)

    out["available_fields"] = available_fields
    out["missing_fields"] = missing_fields
    return out
