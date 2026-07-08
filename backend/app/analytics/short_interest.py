"""Short-interest / borrow-availability read -- a market-structure/positioning
lens, separate from fundamental quality and the technical score. Never
blended into the 0-100 score; shown as context (chip + LLM payload), same
treatment as insider sentiment and entry-risk.

Sourced from the Moomoo snapshot dict, which is already fetched for every
analyzed symbol (`AnalysisService._snapshots`) -- these fields were simply
never read before. No extra network call.
"""
from __future__ import annotations

from app.data.normalize import _f

HIGH_SHORT_SELL_RATE_PCT = 40.0   # % of today's volume that was short-sold
# ponytail: a raw share count, not normalized by float or average daily
# volume -- a genuinely thin, low-float name can trip this at any threshold.
# Add an ADV-relative read if this proves too noisy in practice.
LOW_AVAILABLE_SHARES = 50_000


def read(snapshot: dict | None) -> dict | None:
    """{shortable, short_sell_rate_pct, available_shares, label, reasons} or
    None if the snapshot has none of these fields (e.g. IBKR-sourced
    snapshot, or a market Moomoo has no permission for)."""
    snapshot = snapshot or {}
    shortable = snapshot.get("enable_short_sell")
    shortable = shortable if isinstance(shortable, bool) else None
    # snapshot rows come from a pandas DataFrame, where a missing numeric cell
    # is NaN, not None -- _f() (already used elsewhere for the same reason)
    # filters that out so None is the only "missing" sentinel downstream.
    rate = _f(snapshot.get("short_sell_rate"))
    available = _f(snapshot.get("short_available_volume"))
    if shortable is None and rate is None and available is None:
        return None

    reasons: list[str] = []
    hard_to_borrow = available is not None and available < LOW_AVAILABLE_SHARES
    if hard_to_borrow:
        reasons.append(
            f"Hard to borrow -- only {available:,.0f} shares available to short "
            f"(squeeze risk if the price rallies)."
        )
    if rate is not None and rate >= HIGH_SHORT_SELL_RATE_PCT:
        reasons.append(f"Heavy short-selling today: {rate:.0f}% of volume was short-sold.")
    if shortable is False:
        reasons.append("Shorting is disabled for this name right now.")

    if hard_to_borrow:
        label = "Hard to borrow"
    elif reasons:
        label = "Active shorting"
    elif shortable is True or available is not None:
        label = "Normal"
    else:
        label = "Unknown"

    return {
        "shortable": shortable,
        "short_sell_rate_pct": round(rate, 1) if rate is not None else None,
        "available_shares": round(available) if available is not None else None,
        "label": label,
        "reasons": reasons,
    }


def demo() -> None:
    assert read(None) is None
    assert read({}) is None
    normal = read({"enable_short_sell": True, "short_sell_rate": 5.0, "short_available_volume": 5_000_000})
    assert normal["label"] == "Normal", normal
    htb = read({"enable_short_sell": True, "short_sell_rate": 5.0, "short_available_volume": 1_000})
    assert htb["label"] == "Hard to borrow", htb
    active = read({"enable_short_sell": True, "short_sell_rate": 55.0, "short_available_volume": 5_000_000})
    assert active["label"] == "Active shorting", active
    disabled = read({"enable_short_sell": False, "short_sell_rate": None, "short_available_volume": None})
    assert disabled["label"] == "Active shorting" and "disabled" in disabled["reasons"][0], disabled
    print("short_interest demo OK")


if __name__ == "__main__":
    demo()
