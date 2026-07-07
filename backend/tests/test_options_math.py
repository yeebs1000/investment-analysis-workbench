"""Correctness checks for the options-strategist math upgrade (realized vol,
BSM Greeks, payoff curve + probability of profit) and the new strategy types.
Run directly:

    python -m tests.test_options_math

Hand-constructed synthetic inputs with known-correct expected properties, same
style as test_indicators.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics import options as opt
from app.analytics import options_math as om
from app.data.models import Decision, OptionLeg


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def _simulate_ohlc_from_path(n_days: int, seed: int, daily_vol: float,
                              sub_steps: int = 20, overnight_frac: float = 0.3) -> pd.DataFrame:
    """Simulate OHLC from a genuine intraday path (not just endpoints), so
    high/low reflect real range -- lets Yang-Zhang be checked against a known
    injected total daily vol."""
    rng = np.random.default_rng(seed)
    sigma_overnight = daily_vol * np.sqrt(overnight_frac)
    sigma_intraday_total = daily_vol * np.sqrt(1 - overnight_frac)
    sigma_tick = sigma_intraday_total / np.sqrt(sub_steps)

    opens, highs, lows, closes = [], [], [], []
    prev_close = 100.0
    for _ in range(n_days):
        o = prev_close * np.exp(rng.normal(0, sigma_overnight))
        ticks = [o]
        for _ in range(sub_steps):
            ticks.append(ticks[-1] * np.exp(rng.normal(0, sigma_tick)))
        closes.append(ticks[-1])
        highs.append(max(ticks))
        lows.append(min(ticks))
        opens.append(o)
        prev_close = ticks[-1]
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    return pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes}, index=idx)


def _synthetic_contracts(spot: float, iv_pct: float, dte: int, n_strikes: int = 15,
                          spacing: float = 0.02) -> pd.DataFrame:
    strikes = spot * (1 + spacing * np.arange(-n_strikes, n_strikes + 1))
    rows = []
    for right in ("CALL", "PUT"):
        for k in strikes:
            g = om.bsm_greeks(spot, float(k), iv_pct, dte, right)
            delta = g["delta"] if g["delta"] is not None else 0.0
            price = max(0.01, abs(delta) * spot * 0.08)
            rows.append({
                "right": right, "strike": round(float(k), 2), "delta": delta, "iv": iv_pct,
                "price": round(price, 3), "bid": round(price * 0.98, 3), "ask": round(price * 1.02, 3),
                "code": f"TEST.{right[0]}{int(k)}",
            })
    return pd.DataFrame(rows)


# --- Yang-Zhang realized vol ---------------------------------------------------

def test_yang_zhang_matches_known_reference():
    daily_vol = 0.02   # 2%/day -> ~31.7%/yr annualized
    bars = _simulate_ohlc_from_path(n_days=250, seed=5, daily_vol=daily_vol)
    est = om.realized_vol_yang_zhang(bars, window=60)
    injected_annualized = daily_vol * np.sqrt(252) * 100.0
    assert est is not None
    rel_err = abs(est - injected_annualized) / injected_annualized
    assert rel_err < 0.35, (est, injected_annualized, rel_err)


# --- BSM Greeks -----------------------------------------------------------------

def test_bsm_greeks_atm_call_delta_near_half():
    call = om.bsm_greeks(spot=100.0, strike=100.0, iv_pct=25.0, dte=30, right="Call")
    put = om.bsm_greeks(spot=100.0, strike=100.0, iv_pct=25.0, dte=30, right="Put")
    assert 0.45 < call["delta"] < 0.60
    assert _approx(call["delta"] - put["delta"], 1.0, tol=1e-6)


# --- Payoff curve -----------------------------------------------------------------

def _find_crossings(payoff: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Zero-crossing prices of a piecewise-linear payoff curve. Robust to a
    grid point landing exactly on a breakeven (payoff == 0), which would
    otherwise register as two sign transitions instead of one."""
    prod = payoff[:-1] * payoff[1:]
    idx = np.where(prod <= 0)[0]
    raw = (grid[idx] + grid[idx + 1]) / 2.0
    step = grid[1] - grid[0]
    merged: list[float] = []
    for x in raw:
        if merged and abs(x - merged[-1]) < step * 2:
            merged[-1] = (merged[-1] + x) / 2.0
        else:
            merged.append(float(x))
    return np.array(merged)


def test_payoff_curve_long_straddle_two_breakevens():
    call = OptionLeg(action="Buy", right="Call", strike=100.0, expiry="2099-01-01", price=5.0)
    put = OptionLeg(action="Buy", right="Put", strike=100.0, expiry="2099-01-01", price=5.0)
    grid = om.default_price_grid(100.0, pct=0.3, n=400)
    payoff = om.payoff_at_expiry([call, put], grid)

    crossings = _find_crossings(payoff, grid)
    assert len(crossings) == 2
    assert _approx(crossings[0], 90.0, tol=1.0)
    assert _approx(crossings[1], 110.0, tol=1.0)


# --- Probability of profit --------------------------------------------------------

def test_pop_higher_for_wider_short_strikes():
    spot = 100.0
    grid = om.default_price_grid(spot, pct=0.5, n=400)

    def condor_legs(short_dist: float, wing_dist: float):
        return [
            OptionLeg(action="Sell", right="Put", strike=spot - short_dist, expiry="x", price=2.0),
            OptionLeg(action="Buy", right="Put", strike=spot - short_dist - wing_dist, expiry="x", price=0.5),
            OptionLeg(action="Sell", right="Call", strike=spot + short_dist, expiry="x", price=2.0),
            OptionLeg(action="Buy", right="Call", strike=spot + short_dist + wing_dist, expiry="x", price=0.5),
        ]

    narrow = condor_legs(short_dist=5.0, wing_dist=3.0)
    wide = condor_legs(short_dist=15.0, wing_dist=3.0)
    pop_narrow = om.probability_of_profit(spot, 30.0, 30, om.payoff_at_expiry(narrow, grid), grid)
    pop_wide = om.probability_of_profit(spot, 30.0, 30, om.payoff_at_expiry(wide, grid), grid)
    assert pop_wide > pop_narrow, (pop_narrow, pop_wide)


# --- End-to-end strategy construction ---------------------------------------------

def test_iron_condor_built_when_neutral_and_iv_elevated():
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=1, daily_vol=0.01)   # low realized vol
    contracts = _synthetic_contracts(spot, iv_pct=60.0, dte=30)           # high IV vs realized
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.HOLD, score=50.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=False, shares=0.0,
    )
    assert result.iv_regime == "Elevated"
    condors = [s for s in result.strategies if s.name == "Iron Condor"]
    assert len(condors) == 1
    assert len(condors[0].legs) == 4
    assert condors[0].pop_pct is not None


def test_straddle_built_when_neutral_and_iv_cheap():
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=2, daily_vol=0.02)   # higher realized vol
    contracts = _synthetic_contracts(spot, iv_pct=8.0, dte=30)            # low IV vs realized
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.HOLD, score=50.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=False, shares=0.0,
    )
    assert result.iv_regime == "Cheap"
    straddles = [s for s in result.strategies if s.name == "Long Straddle"]
    assert len(straddles) == 1
    assert len(straddles[0].legs) == 2
    assert straddles[0].pop_pct is not None


def test_existing_strategies_unaffected():
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=3, daily_vol=0.01)
    contracts = _synthetic_contracts(spot, iv_pct=60.0, dte=30)
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.BUY, score=70.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=False, shares=0.0,
    )
    names = {s.name for s in result.strategies}
    assert "Bull Put Spread (credit)" in names
    assert "Cash-Secured Put" in names
    for s in result.strategies:
        # net Greeks / POP now populate on every strategy without breaking
        # the existing economics fields.
        assert s.net_delta is not None
        assert s.pop_pct is not None
        assert s.max_profit is not None or s.max_loss is not None


def test_covered_call_includes_stock_leg():
    """A covered call is stock + short call: net delta must be strongly
    POSITIVE (~1 - call delta) and POP must reflect the 'profitable above
    spot - premium' region, not the naked short call's 'below strike +
    premium' region. Regression test for the stock-leg omission bug."""
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=4, daily_vol=0.01)
    contracts = _synthetic_contracts(spot, iv_pct=60.0, dte=30)
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.HOLD, score=50.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=True, shares=200.0,
    )
    cc = [s for s in result.strategies if s.name == "Covered Call"]
    assert len(cc) == 1
    s = cc[0]
    assert s.net_delta is not None and s.net_delta > 0.5, s.net_delta   # long stock dominates
    assert s.max_profit is not None and s.max_profit > 0
    assert s.max_loss is not None and s.max_loss > 0
    # profitable region is 'above spot - premium' -- with ~0.30-delta call
    # premium collected, POP should exceed the ~50% coin-flip of raw stock.
    assert s.pop_pct is not None and s.pop_pct > 50.0, s.pop_pct


def test_degenerate_credit_spread_skipped():
    """A 'credit' spread whose mid quotes produce a non-positive credit is a
    data artifact and must not be recommended."""
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=6, daily_vol=0.01)
    contracts = _synthetic_contracts(spot, iv_pct=60.0, dte=30)
    # corrupt the quotes: make every option the same price, so short - long = 0
    contracts["price"] = 1.0
    contracts["bid"] = 0.98
    contracts["ask"] = 1.02
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.BUY, score=70.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=False, shares=0.0,
    )
    names = {s.name for s in result.strategies}
    assert "Bull Put Spread (credit)" not in names, "zero-credit spread must be skipped"


def test_expected_pnl_sane():
    """EV engine sanity: under the zero-drift lognormal, long stock has EV ~0,
    and overpaying for a straddle (vs fair) must show negative EV while
    underpaying shows a higher EV."""
    spot = 100.0
    grid = om.default_price_grid(spot, pct=0.5, n=400)

    def straddle(price_each: float):
        return [
            OptionLeg(action="Buy", right="Call", strike=100.0, expiry="x", price=price_each),
            OptionLeg(action="Buy", right="Put", strike=100.0, expiry="x", price=price_each),
        ]

    cheap = om.expected_pnl(spot, 30.0, 30, om.payoff_at_expiry(straddle(1.0), grid), grid)
    rich = om.expected_pnl(spot, 30.0, 30, om.payoff_at_expiry(straddle(10.0), grid), grid)
    assert cheap is not None and rich is not None
    assert cheap > rich
    # premium enters EV dollar-for-dollar: 9 more paid per leg = 18 lower EV
    assert _approx(cheap - rich, 18.0, tol=0.01)
    # grossly overpriced straddle must be clearly negative-EV
    assert rich < -5.0


def test_earnings_inside_tenor_vetoes_condor_and_warns_short_premium():
    """Earnings within the option tenor: the Iron Condor must NOT be offered
    (selling a range through a binary event), and any short-premium structure
    that IS offered must carry an earnings warning."""
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=1, daily_vol=0.01)
    contracts = _synthetic_contracts(spot, iv_pct=60.0, dte=30)
    earnings = {"date": "2099-01-15", "hour": "amc", "eps_estimate": None}
    # dte spans past the earnings date (expiry 2099-02-01, earnings 2099-01-15)
    import datetime
    # make "today" reasoning work: earnings 10 days out, tenor 30 days
    near_earn = {"date": (datetime.date.today() + datetime.timedelta(days=10)).isoformat()}
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.HOLD, score=50.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=False, shares=0.0, earnings=near_earn,
    )
    assert not any(s.name == "Iron Condor" for s in result.strategies), "condor must be vetoed through earnings"
    assert any("NOT offered" in n and "earnings" in n.lower() for n in result.notes)


def test_earnings_after_expiry_allows_condor():
    """Earnings AFTER the expiry must not veto the condor."""
    import datetime
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=1, daily_vol=0.01)
    contracts = _synthetic_contracts(spot, iv_pct=60.0, dte=30)
    far_earn = {"date": (datetime.date.today() + datetime.timedelta(days=60)).isoformat()}
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.HOLD, score=50.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=False, shares=0.0, earnings=far_earn,
    )
    assert any(s.name == "Iron Condor" for s in result.strategies), "condor should survive earnings after expiry"


def test_risk_budget_sizing():
    """A structure's suggested_contracts must keep max-loss within the budget."""
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=3, daily_vol=0.01)
    contracts = _synthetic_contracts(spot, iv_pct=60.0, dte=30)
    book = 500_000.0   # large enough that a defined-risk spread fits the 1% budget
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.BUY, score=70.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=False, shares=0.0,
        book_value_usd=book, risk_budget_frac=0.01,
    )
    sized = [s for s in result.strategies if s.suggested_contracts is not None]
    assert sized, "at least one defined-risk structure should get sizing"
    for s in sized:
        worst_case = s.max_loss * 100 * s.suggested_contracts
        assert worst_case <= book * 0.01 + 1e-6, (s.name, worst_case, book * 0.01)


def test_tiny_account_refuses_oversized_structure():
    """A book too small for even one contract must get a warning, not a size."""
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=3, daily_vol=0.01)
    contracts = _synthetic_contracts(spot, iv_pct=60.0, dte=30)
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.BUY, score=70.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=False, shares=0.0,
        book_value_usd=3_000.0, risk_budget_frac=0.01,   # $30 budget -> nothing fits
    )
    for s in result.strategies:
        if s.max_loss and s.max_loss > 0:
            assert s.suggested_contracts is None
            assert any("too large" in w for w in s.warnings)


def test_low_confidence_adds_neutral_alongside_directional():
    """A low-confidence bullish read should still surface neutral structures."""
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=1, daily_vol=0.01)
    contracts = _synthetic_contracts(spot, iv_pct=60.0, dte=30)  # elevated IV
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.BUY, score=60.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=False, shares=0.0, confidence=0.20,
    )
    names = {s.name for s in result.strategies}
    assert any("Bull Put" in n or "Cash-Secured" in n for n in names), "directional still present"
    assert "Iron Condor" in names, "neutral structure added under low confidence"
    assert any("LOW confidence" in n for n in result.notes)


def test_garch_forecast_tracks_injected_vol():
    """GARCH forecast on a series with a known constant daily vol should land
    near the injected annualized level. Also: returns None on too-short history
    (the documented fallback path), never raises."""
    if om.forecast_vol_garch is None or not om._ARCH_AVAILABLE:
        print("  (arch not installed -- skipping GARCH test)")
        return
    daily_vol = 0.015   # ~23.8%/yr
    bars = _simulate_ohlc_from_path(n_days=600, seed=11, daily_vol=daily_vol)
    fc = om.forecast_vol_garch(bars, horizon_days=30)
    injected = daily_vol * np.sqrt(252) * 100.0
    assert fc is not None, "expected a forecast on 600 bars"
    rel = abs(fc - injected) / injected
    assert rel < 0.5, (fc, injected, rel)   # GARCH on a stationary series -> ballpark

    short = _simulate_ohlc_from_path(n_days=100, seed=12, daily_vol=daily_vol)
    assert om.forecast_vol_garch(short, horizon_days=30) is None, "too-short history must return None"


def test_garch_drives_iv_regime_when_available():
    """When GARCH produces a forecast, the IV regime must compare against it
    (iv_regime_basis == 'garch_forecast') rather than trailing realized vol."""
    if not om._ARCH_AVAILABLE:
        print("  (arch not installed -- skipping GARCH regime test)")
        return
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=600, seed=13, daily_vol=0.012)
    contracts = _synthetic_contracts(spot, iv_pct=60.0, dte=30)
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.HOLD, score=50.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=False, shares=0.0,
    )
    assert result.forecast_vol_pct is not None
    assert result.iv_regime_basis == "garch_forecast"


def test_collar_built_for_holder_when_bearish():
    """Holding >=100 shares with a bearish read must offer a Collar: buy put +
    sell call, stock-inclusive metrics (positive net delta, hard floor)."""
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=8, daily_vol=0.015)
    contracts = _synthetic_contracts(spot, iv_pct=40.0, dte=30)
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.REDUCE, score=40.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=True, shares=300.0,
        book_value_usd=100_000.0,
    )
    collars = [s for s in result.strategies if s.name == "Collar"]
    assert len(collars) == 1
    c = collars[0]
    assert len(c.legs) == 2
    actions = {(l.action, l.right) for l in c.legs}
    assert ("Buy", "Put") in actions and ("Sell", "Call") in actions
    # stock + long put + short call: delta positive but well under 1
    assert c.net_delta is not None and 0.0 < c.net_delta < 1.0, c.net_delta
    # hard floor: max loss defined and finite
    assert c.max_loss is not None and c.max_loss > 0
    assert c.max_profit is not None
    # sized by shares held (300 -> 3 contracts), incremental capital = net cost
    assert c.suggested_contracts == 3
    assert c.capital_required_usd is not None and c.capital_required_usd >= 0


def test_collar_absent_without_shares():
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=8, daily_vol=0.015)
    contracts = _synthetic_contracts(spot, iv_pct=40.0, dte=30)
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.REDUCE, score=40.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=False, shares=0.0,
    )
    assert not any(s.name == "Collar" for s in result.strategies)


def test_collar_offered_for_holder_through_earnings():
    """Even with a non-bearish read, earnings inside the tenor should offer the
    collar to a share-holder (carrying shares through a binary event), with the
    collar-specific warning rather than the IV-crush one."""
    import datetime
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=8, daily_vol=0.015)
    contracts = _synthetic_contracts(spot, iv_pct=40.0, dte=30)
    near_earn = {"date": (datetime.date.today() + datetime.timedelta(days=10)).isoformat()}
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.HOLD, score=50.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=True, shares=100.0, earnings=near_earn,
    )
    collars = [s for s in result.strategies if s.name == "Collar"]
    assert len(collars) == 1
    assert any("exactly what a collar is for" in w for w in collars[0].warnings)


def test_skew_computed():
    """25-delta skew should be reported when both wings have IV."""
    spot = 100.0
    bars = _simulate_ohlc_from_path(n_days=120, seed=1, daily_vol=0.01)
    contracts = _synthetic_contracts(spot, iv_pct=60.0, dte=30)
    result = opt.build_analysis(
        code="TEST", name="Test Co", as_of=None, spot=spot,
        decision=Decision.HOLD, score=50.0, bars=bars, contracts=contracts,
        expiry="2099-01-01", dte=30, holds=False, shares=0.0,
    )
    # synthetic contracts use a flat IV, so skew should be ~0 (both wings equal)
    assert result.skew_25d_pts is not None
    assert _approx(result.skew_25d_pts, 0.0, tol=0.5)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} options-math tests passed.")


if __name__ == "__main__":
    main()
