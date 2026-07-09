"""Offline checks for the synthetic options backtest. Fully deterministic --
synthetic GBM price path, no network/broker -- so it's safe for CI.

    python -m tests.test_options_backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.analytics import options_backtest as bt
from app.analytics import options_math as om


def _gbm_bars(n=400, mu=0.0003, sigma=0.02, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(mu, sigma, n)))
    return pd.DataFrame({
        "open": close, "high": close * 1.01, "low": close * 0.99,
        "close": close, "volume": 1e6,
    }, index=dates)


def test_synth_chain_prices_are_sane():
    ch = bt.synth_chain(100.0, 25.0, 35)
    assert not ch.empty
    assert {"right", "strike", "delta", "iv", "price", "bid", "ask", "oi", "code"} <= set(ch.columns)
    # ATM call: 0 < price < spot; bid < ask
    atm = ch[(ch.right == "Call") & (ch.strike.between(99, 101))]
    assert (atm["price"] > 0).all() and (atm["price"] < 100).all()
    assert (ch["bid"] <= ch["ask"]).all()
    # calls have positive delta, puts negative
    assert (ch[ch.right == "Call"]["delta"] > 0).all()
    assert (ch[ch.right == "Put"]["delta"] < 0).all()


def test_bsm_price_matches_intrinsic_at_zero_vol_limit():
    # tiny vol, short tenor -> price collapses to intrinsic
    itm = om.bsm_price(120.0, 100.0, 1.0, 1, "Call")
    assert abs(itm - 20.0) < 0.5, itm
    otm = om.bsm_price(80.0, 100.0, 1.0, 1, "Call")
    assert otm < 0.5, otm


def test_payoff_mark_matches_manual_bull_put_spread():
    # a bull put spread: sell 95 put @2, buy 90 put @1 -> credit 1.
    # at expiry S=100 (above both): keep full credit = +1/share.
    from app.data.models import OptionLeg
    legs = [
        OptionLeg(action="Sell", right="Put", strike=95, expiry="x", price=2.0),
        OptionLeg(action="Buy", right="Put", strike=90, expiry="x", price=1.0),
    ]
    pnl_above = float(om.payoff_at_expiry(legs, np.array([100.0]))[0])
    assert abs(pnl_above - 1.0) < 1e-9, pnl_above
    # at S=90 (both ITM): loss = width - credit = 5 - 1 = -4.
    pnl_below = float(om.payoff_at_expiry(legs, np.array([90.0]))[0])
    assert abs(pnl_below - (-4.0)) < 1e-9, pnl_below


def test_backtest_runs_and_stats_are_bounded():
    bars = _gbm_bars()
    trades = bt.backtest_symbol("US.TEST", "Test", bars, horizon=21, step=15)
    assert len(trades) > 0
    stats = bt.aggregate(trades)
    assert stats["__ALL__"].n == len(trades)
    for k, s in stats.items():
        if s.n:
            assert 0.0 <= s.win_rate <= 100.0
            assert s.predicted_pop is None or 0.0 <= s.predicted_pop <= 100.0
    # win flag must be consistent with sign of P&L
    for t in trades:
        assert t.win == (t.pnl_per_share > 0)


def test_report_renders():
    bars = _gbm_bars(seed=3)
    trades = bt.backtest_symbol("US.TEST", "Test", bars, horizon=21, step=15)
    txt = bt.format_report(trades, {"n_symbols": 1, "horizon": 21, "step": 15, "vrp": 0.05})
    assert "SYNTHETIC BACKTEST" in txt and "pred POP" in txt


def test_regime_map_and_plumbing():
    bench = _gbm_bars(n=400, mu=0.001, seed=7)   # drifting up -> mostly bull
    rmap = bt.regime_map_from_bench(bench)
    # no reads until the 200-day SMA exists; one read per bar after that
    assert len(rmap) == len(bench) - 199
    assert set(rmap.values()) <= {"bull", "bear"}
    # the map keys align with bar dates and carry into trades
    bars = _gbm_bars(n=400, seed=0)
    trades = bt.backtest_symbol("US.TEST", "Test", bars, horizon=21, step=15, regime_map=rmap)
    assert trades and all(t.regime in ("bull", "bear", None) for t in trades)
    assert any(t.regime is not None for t in trades)
    # per-regime section renders
    txt = bt.format_report(trades, {"n_symbols": 1, "horizon": 21, "step": 15, "vrp": 0.05, "regime": True})
    assert "regime only" in txt


def test_managed_terminal_matches_payoff():
    # With no rule triggered, the path's LAST mark must equal payoff_at_expiry --
    # proves the daily BSM repricing is consistent with the terminal intrinsic.
    from app.data.models import OptionLeg, OptionStrategy
    legs = [
        OptionLeg(action="Buy", right="Call", strike=100, expiry="x", price=3.0, iv_pct=25.0),
        OptionLeg(action="Sell", right="Call", strike=110, expiry="x", price=1.0, iv_pct=25.0),
    ]
    s = OptionStrategy(name="Call Debit Spread", direction="Bullish", legs=legs,
                       net_debit_credit=-2.0, max_loss=2.0, max_profit=8.0)
    horizon = 20
    # flat path ending exactly at entry spot -> known terminal payoff
    path = np.full(horizon, 100.0)
    m_pnl, reason, held = bt.managed_exit(s, 100.0, path, horizon)
    term = float(om.payoff_at_expiry(legs, np.array([100.0]))[0])
    assert reason == "expiry" and held == horizon
    assert abs(m_pnl - term) < 1e-6, (m_pnl, term)


def test_managed_profit_target_triggers_on_credit():
    # a credit spread whose underlying rockets away from the short strike will hit
    # the 70%-of-credit target well before expiry -> early exit.
    from app.data.models import OptionLeg, OptionStrategy
    legs = [
        OptionLeg(action="Sell", right="Put", strike=95, expiry="x", price=2.5, iv_pct=30.0),
        OptionLeg(action="Buy", right="Put", strike=90, expiry="x", price=1.0, iv_pct=30.0),
    ]
    s = OptionStrategy(name="Bull Put Spread (credit)", direction="Bullish", legs=legs,
                       net_debit_credit=1.5, max_profit=1.5, max_loss=3.5)
    horizon = 30
    path = np.linspace(101, 120, horizon)   # rallies hard -> puts decay fast
    m_pnl, reason, held = bt.managed_exit(s, 100.0, path, horizon)
    assert reason == "profit_target", (reason, m_pnl, held)
    assert held < horizon and m_pnl > 0


def test_managed_be_stop_protects_a_faded_winner():
    # debit spread that spikes (arms the BE stop) then fully reverses -> exits ~0
    # instead of riding to a loss.
    from app.data.models import OptionLeg, OptionStrategy
    legs = [
        OptionLeg(action="Buy", right="Call", strike=100, expiry="x", price=3.0, iv_pct=30.0),
        OptionLeg(action="Sell", right="Call", strike=110, expiry="x", price=1.0, iv_pct=30.0),
    ]
    s = OptionStrategy(name="Call Debit Spread", direction="Bullish", legs=legs,
                       net_debit_credit=-2.0, max_profit=8.0, max_loss=2.0)
    horizon = 30
    path = np.concatenate([np.linspace(100, 112, 15), np.linspace(112, 99, 15)])
    m_pnl, reason, held = bt.managed_exit(s, 100.0, path, horizon)
    assert reason == "stop_be", (reason, m_pnl, held)
    assert abs(m_pnl) < 1e-9


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        t()
        print(f"  PASS  {t.__name__}")
        passed += 1
    print(f"\n{passed}/{len(tests)} options-backtest tests passed.")


if __name__ == "__main__":
    main()
