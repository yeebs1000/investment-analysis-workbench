"""Self-check for the per-leg execution gate (scripts/log_signals.leg_tradeable).

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/test_leg_gate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.log_signals import (  # noqa: E402
    MIN_OI, WIDE_ABS_SPREAD, WIDE_SPREAD_PCT_CAP, leg_tradeable,
)


def demo() -> None:
    OI = MIN_OI + 10

    # --- the normal %-of-mid path still governs liquid legs -------------------
    assert leg_tradeable(10.00, 10.50, OI)      # 4.9% of mid -> fine
    assert not leg_tradeable(10.00, 12.50, OI)  # 22% on a $11 mid, $2.50 wide -> junk

    # --- the reason this gate exists: cheap wings with tight ABSOLUTE spreads.
    # 20% of mid (would have failed the old 10% rule) but only $0.40 to cross.
    assert leg_tradeable(1.80, 2.20, OI)

    # --- but a cheap leg is not a free pass: the % cap still bites ------------
    assert not leg_tradeable(0.30, 0.62, OI)    # $0.32 wide but 70% of mid -> far-OTM junk

    # --- boundaries ----------------------------------------------------------
    # exactly at the absolute allowance, under the pct cap -> tradeable
    assert leg_tradeable(2.00, 2.00 + WIDE_ABS_SPREAD, OI)
    # a hair over the absolute allowance, and over 10% of mid -> rejected
    assert not leg_tradeable(2.00, 2.00 + WIDE_ABS_SPREAD + 0.01, OI)
    # wide-but-cheap must also respect the pct ceiling
    mid_at_cap = WIDE_ABS_SPREAD / (WIDE_SPREAD_PCT_CAP / 100.0)   # spread == cap% of mid
    assert leg_tradeable(mid_at_cap - WIDE_ABS_SPREAD / 2, mid_at_cap + WIDE_ABS_SPREAD / 2, OI)

    # --- liquidity + garbage input -------------------------------------------
    assert not leg_tradeable(10.00, 10.50, MIN_OI - 1)   # thin OI
    assert not leg_tradeable(0, 0.50, OI)                # no bid
    assert not leg_tradeable(10.00, 9.00, OI)            # crossed market
    assert not leg_tradeable(None, 10.50, OI)
    assert not leg_tradeable(float("nan"), 10.50, OI)
    assert not leg_tradeable(10.00, 10.50, float("nan"))

    # --- the LLY case that motivated the change: genuinely wide quotes stay out
    # (bid 11.15 / ask 15.20 -> $4.05 wide, 31% of mid) -- eligible must NOT mean lax
    assert not leg_tradeable(11.15, 15.20, 789)

    print("leg_tradeable: all checks passed")


if __name__ == "__main__":
    demo()
