"""Self-check for the tail-hedge spread selector (paper_trade._pick_hedge_spread).

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/test_hedge.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import scripts.paper_trade as pt  # noqa: E402


def _puts(strikes, mids):
    return pd.DataFrame({
        "strike": strikes,
        "code": [f"US.SPY-P{k}" for k in strikes],
        "bid": [m - 0.5 for m in mids],
        "ask": [m + 0.5 for m in mids],
    })


def demo() -> None:
    spot = 750.0
    strikes = [640, 650, 660, 670, 680, 690, 700, 710, 720, 730]
    mids = [3, 3.5, 4, 5, 6.5, 8, 10, 12, 15, 19]        # richer toward the money
    puts = _puts(strikes, mids)

    p = pt._pick_hedge_spread(spot, puts, budget=1200.0)
    assert p is not None
    assert p["long_strike"] == 710, p["long_strike"]      # 5.5% OTM of 750 = 708.75 -> 710
    assert p["short_strike"] == 660, p["short_strike"]    # 12% OTM = 660
    assert p["short_strike"] < p["long_strike"]           # real width, never inverted
    assert abs(p["net_debit"] - 8.0) < 1e-9               # mid 12 - mid 4
    assert p["width"] == 50
    assert p["contracts"] == 1                            # floor(1200 / (8*100))

    # payoff rises with the drawdown, then caps at (width - debit) * 100 * n
    assert p["payoff"]["15%"] >= p["payoff"]["10%"]
    cap = (p["width"] - p["net_debit"]) * 100 * p["contracts"]
    assert p["payoff"]["20%"] == cap
    assert p["payoff"]["10%"] > 0                          # -10% already in the money

    # contracts scale with the carry budget
    assert pt._pick_hedge_spread(spot, puts, budget=3200.0)["contracts"] == 4

    # never exceed the hard contract cap
    big = pt._pick_hedge_spread(spot, puts, budget=1_000_000.0)
    assert big["contracts"] == pt.HEDGE_MAX_CONTRACTS

    # degenerate inputs -> None (no naked/garbage hedge)
    bad = _puts([700], [10]); bad["bid"] = 0.0
    assert pt._pick_hedge_spread(spot, bad, 1200.0) is None      # no two-sided quote
    assert pt._pick_hedge_spread(spot, _puts([710], [12]), 1200.0) is None  # can't form width
    assert pt._pick_hedge_spread(spot, None, 1200.0) is None

    print(f"hedge selector: {p['contracts']}x SPY {p['long_strike']:.0f}/{p['short_strike']:.0f}p "
          f"@ ${p['net_debit']}, pays ${p['payoff']['10%']:.0f} at -10% / "
          f"${p['payoff']['15%']:.0f} at -15% -- all checks pass")


if __name__ == "__main__":
    demo()
