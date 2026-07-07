"""Correctness checks for the risk-aware portfolio optimizer. Run directly:

    python -m tests.test_risk_optimizer

Hand-constructed synthetic inputs with known-correct expected properties, same
style as test_indicators.py.
"""
from __future__ import annotations

import types

import numpy as np
import pandas as pd

from app.analytics import risk
from app.data.models import (
    Account,
    Decision,
    HoldingAnalysis,
    PortfolioAnalysis,
    PortfolioRisk,
    Position,
    TechnicalAnalysis,
)


def _approx(a, b, tol=1e-6):
    return abs(float(a) - float(b)) <= tol


def test_option_positions_held_not_optimized():
    """An option-contract position (e.g. US.IREN260702C44000) must come back
    as HOLD with its value untouched -- never BUY/SELL sized by equity
    technicals -- and its capital must not be re-allocated to other names."""
    codes = ["US.AAA", "US.BBB", "US.CCC", "US.DDD"]
    holdings = [_holding(c, score=70.0, decision=Decision.BUY) for c in codes]
    holdings.append(_holding("US.IREN260702C44000", score=80.0, decision=Decision.STRONG_BUY,
                             price=3.0, qty=100.0))
    pa = _portfolio(holdings, cash=0.0)
    plan = risk.optimize_portfolio(pa, cash_usd=0.0, cap_pct=40.0)

    opt_action = next(a for a in plan.actions if a.code == "US.IREN260702C44000")
    assert opt_action.action == "HOLD"
    assert opt_action.delta_usd == 0.0
    assert "option" in opt_action.reason.lower()
    # equity targets + option value + cash target must not over-allocate the book
    total = plan.total_value_usd
    equity_targets = sum(a.target_pct for a in plan.actions if a.code != "US.IREN260702C44000")
    option_pct = opt_action.current_pct
    assert equity_targets + option_pct <= 100.0 - plan.cash_target_pct + 1.0, (
        equity_targets, option_pct, total)


def _position(code: str, price: float = 100.0, qty: float = 10.0, currency: str = "USD") -> Position:
    return Position(
        code=code, name=code, market="US", currency=currency, broker="moomoo",
        qty=qty, last_price=price, market_value=price * qty, cost_price=price,
    )


def _ta(code: str, score: float, decision: Decision = Decision.BUY) -> TechnicalAnalysis:
    return TechnicalAnalysis(
        code=code, name=code, score=score, decision=decision,
        confidence=0.7, confidence_label="Medium",
    )


def _holding(code: str, score: float, decision: Decision = Decision.BUY,
             price: float = 100.0, qty: float = 10.0) -> HoldingAnalysis:
    p = _position(code, price=price, qty=qty)
    ta = _ta(code, score=score, decision=decision)
    return HoldingAnalysis(position=p, analysis=ta, weight_pct=0.0, action=decision, action_reason="")


def _portfolio(holdings: list[HoldingAnalysis], cash: float = 0.0) -> PortfolioAnalysis:
    account = Account(currency="USD", total_assets=0.0, cash=cash, market_value=0.0, cash_usd=cash)
    return PortfolioAnalysis(account=account, risk=PortfolioRisk(), holdings=holdings)


def _synthetic_bars(n_days: int, seed: int, vol: float = 0.02, start: str = "2023-01-02") -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(start=start, periods=n_days)
    rets = rng.normal(0.0002, vol, n_days)
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.roll(close, 1)
    open_[0] = 100.0
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n_days))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n_days))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": 1_000_000.0},
        index=idx,
    )


def _eight_holding_portfolio_with_bars(n_days: int = 120):
    # 8 names comfortably clears both MIN_HOLDINGS_FOR_RISK_MODEL and the
    # default-cap feasibility floor (8 * ~15.8% > 100%), so a forced solver
    # failure -- not the upfront feasibility check -- is what gets exercised.
    codes = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG", "HHH"]
    holdings = [_holding(c, score=70.0, decision=Decision.BUY) for c in codes]
    pa = _portfolio(holdings, cash=0.0)
    bars_by_code = {c: _synthetic_bars(n_days, seed=i) for i, c in enumerate(codes)}
    return pa, bars_by_code, len(codes)


# --- Stage A: correlation penalty --------------------------------------------

def test_correlation_penalty_splits_uncorrelated_more_evenly():
    codes = ["A", "B", "C"]
    raw = {"A": 1.0, "B": 1.0, "C": 1.0}
    vol = 0.20

    # Scenario 1: A and B highly correlated, C independent.
    corr_hi = np.array([
        [1.00, 0.95, 0.00],
        [0.95, 1.00, 0.00],
        [0.00, 0.00, 1.00],
    ])
    cov_hi = corr_hi * (vol ** 2)
    risk_adj_hi = risk._risk_overlay_weights(raw, cov_hi, corr_hi, codes)

    # Scenario 2: everyone uncorrelated.
    corr_lo = np.eye(3)
    cov_lo = corr_lo * (vol ** 2)
    risk_adj_lo = risk._risk_overlay_weights(raw, cov_lo, corr_lo, codes)

    ab_share_hi = (risk_adj_hi["A"] + risk_adj_hi["B"]) / sum(risk_adj_hi.values())
    ab_share_lo = (risk_adj_lo["A"] + risk_adj_lo["B"]) / sum(risk_adj_lo.values())
    assert ab_share_hi < ab_share_lo, (ab_share_hi, ab_share_lo)

    # Equivalently: C picks up more of the total weight when A/B are correlated.
    c_share_hi = risk_adj_hi["C"] / sum(risk_adj_hi.values())
    c_share_lo = risk_adj_lo["C"] / sum(risk_adj_lo.values())
    assert c_share_hi > c_share_lo, (c_share_hi, c_share_lo)


# --- Ledoit-Wolf shrinkage -----------------------------------------------------

def test_ledoit_wolf_shrinkage_shape_and_psd():
    rng = np.random.default_rng(11)
    n_days, n_assets = 100, 8
    common = rng.normal(0, 0.01, n_days)
    idiosyncratic = rng.normal(0, 0.015, (n_days, n_assets))
    rets = idiosyncratic + common[:, None] * 0.6
    returns = pd.DataFrame(rets, columns=[f"S{i}" for i in range(n_assets)])

    cov, corr, delta = risk._shrunk_covariance(returns)
    assert cov.shape == (n_assets, n_assets)
    assert np.allclose(cov, cov.T, atol=1e-10)
    eigvals = np.linalg.eigvalsh(cov)
    assert eigvals.min() >= -1e-6
    assert np.allclose(np.diagonal(corr), 1.0, atol=1e-8)
    assert 0.0 <= delta <= 1.0


# --- Stage B: constrained solve ------------------------------------------------

def test_solver_respects_cap_and_no_short():
    codes = ["A", "B", "C", "D", "E"]
    risk_adj = np.array([100.0, 1.0, 1.0, 1.0, 1.0])
    cap_frac = 0.30   # 5 assets * 0.30 = 1.5 >= 1.0 -> sum-to-1 is actually reachable
    investable = 10_000.0

    targets, converged, note = risk._solve_risk_adjusted_weights(codes, risk_adj, cap_frac, investable)
    assert converged, note
    total = sum(targets.values())
    assert _approx(total, investable, tol=1.0)
    for v in targets.values():
        assert v >= -1e-6
        assert v <= cap_frac * investable + 1.0


def test_solver_pins_sell_rated_names_to_zero():
    codes = ["A", "B", "C"]
    risk_adj = np.array([5.0, 0.0, 5.0])
    targets, converged, note = risk._solve_risk_adjusted_weights(codes, risk_adj, 0.5, 1000.0)
    assert converged, note
    assert _approx(targets["B"], 0.0, tol=1e-4)


def test_convergence_failure_falls_back_to_heuristic():
    pa, bars_by_code, n = _eight_holding_portfolio_with_bars()

    fake_result = types.SimpleNamespace(success=False, x=np.full(n, np.nan), message="forced failure")
    orig_minimize = risk.minimize
    risk.minimize = lambda *a, **k: fake_result
    try:
        plan = risk.optimize_portfolio(pa, method="risk_aware", bars_by_code=bars_by_code)
    finally:
        risk.minimize = orig_minimize

    assert plan.method_used == "heuristic"
    assert any("forced failure" in n for n in plan.risk_notes)
    assert len(plan.actions) == n


# --- Degenerate-input gating ----------------------------------------------------

def test_below_min_holdings_uses_heuristic():
    holdings = [_holding(c, score=70.0) for c in ("AAA", "BBB", "CCC")]
    pa = _portfolio(holdings)
    plan = risk.optimize_portfolio(pa, method="risk_aware", bars_by_code={})
    assert plan.method_used == "heuristic"
    assert any("at least" in n and "holdings" in n for n in plan.risk_notes)


def test_short_history_holding_gets_neutral_adjustment():
    codes = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF", "GGG"]
    raw = {c: 1.0 for c in codes}
    bars_by_code = {c: _synthetic_bars(120, seed=i) for i, c in enumerate(codes[:-1])}
    bars_by_code["GGG"] = _synthetic_bars(10, seed=99)   # too short: excluded from covariance

    returns, cov_codes = risk._build_returns_frame(bars_by_code)
    assert "GGG" not in cov_codes
    assert set(cov_codes) == set(codes[:-1])

    cov, corr, _delta = risk._shrunk_covariance(returns)
    risk_adj = risk._risk_overlay_weights(raw, cov, corr, cov_codes)

    assert _approx(risk_adj["GGG"], raw["GGG"])
    for c in cov_codes:
        assert abs(risk_adj[c] - raw[c]) > 1e-9


# --- Regression: heuristic path byte-identical to the pre-refactor behavior ---

def test_heuristic_mode_output_byte_identical_to_before():
    aaa = _holding("AAA", score=70.0, decision=Decision.BUY, price=100.0, qty=10.0)
    bbb = _holding("BBB", score=54.0, decision=Decision.BUY, price=100.0, qty=10.0)
    pa = _portfolio([aaa, bbb], cash=0.0)

    plan = risk.optimize_portfolio(pa, cash_usd=0.0, cap_pct=100.0, cash_target_pct=0.0)

    assert plan.method_used == "heuristic"
    assert plan.portfolio_vol_pct is None
    assert plan.covariance_shrinkage is None
    assert plan.risk_notes == []
    assert _approx(plan.total_value_usd, 2000.0)
    assert _approx(plan.buy_usd, 500.0)
    assert _approx(plan.sell_usd, 500.0)
    assert _approx(plan.projected_top_pct, 75.0)

    by_code = {a.code: a for a in plan.actions}
    assert by_code["AAA"].action == "ADD"
    assert _approx(by_code["AAA"].current_pct, 50.0)
    assert _approx(by_code["AAA"].target_pct, 75.0)
    assert _approx(by_code["AAA"].delta_usd, 500.0)
    assert _approx(by_code["AAA"].est_shares, 5.0)
    assert by_code["AAA"].risk_contribution_pct is None

    assert by_code["BBB"].action == "TRIM"
    assert _approx(by_code["BBB"].current_pct, 50.0)
    assert _approx(by_code["BBB"].target_pct, 25.0)
    assert _approx(by_code["BBB"].delta_usd, -500.0)
    assert _approx(by_code["BBB"].est_shares, -5.0)

    assert [a.code for a in plan.actions] == ["BBB", "AAA"]   # TRIM before ADD


# --- Risk contribution decomposition -------------------------------------------

def test_risk_contribution_sums_to_100pct():
    codes = ["A", "B", "C"]
    rng = np.random.default_rng(3)
    corr = np.array([
        [1.00, 0.30, 0.10],
        [0.30, 1.00, 0.20],
        [0.10, 0.20, 1.00],
    ])
    vols = np.array([0.15, 0.25, 0.10])
    cov = corr * np.outer(vols, vols)
    weight_frac = {"A": 0.5, "B": 0.3, "C": 0.2}

    rc, port_vol_pct = risk._risk_contributions(codes, weight_frac, cov, codes)
    assert port_vol_pct is not None and port_vol_pct > 0
    assert _approx(sum(rc.values()), 100.0, tol=1e-3)


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} risk optimizer tests passed.")


if __name__ == "__main__":
    main()
