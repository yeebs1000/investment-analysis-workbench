"""Pure-calculation helpers for the options strategist: a better realized-vol
estimator, native Black-Scholes Greeks, and a generic payoff-curve/probability-
of-profit engine. No I/O, no broker/service imports -- mirrors how
indicators.py separates pure math from technical.py's orchestration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import ndtr as _ndtr   # normal CDF without scipy.stats' per-call wrapper overhead

_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


class _FastNorm:
    """Drop-in for scipy.stats.norm's cdf/pdf, ~10-50x faster per call: ndtr is
    the bare normal CDF (what norm.cdf wraps in argsreduce/broadcast machinery),
    and the pdf is the closed form. Works on scalars and arrays. This math is on
    the backtest's hot path (millions of BSM evals) -- it dominated the profile."""
    @staticmethod
    def cdf(x):
        return _ndtr(x)

    @staticmethod
    def pdf(x):
        x = np.asarray(x, dtype=float)
        return np.exp(-0.5 * x * x) * _INV_SQRT_2PI


norm = _FastNorm()

PERIODS_PER_YEAR = 252.0

# Approximate short-tenor risk-free rate used in BSM Greeks (annualized %).
# There is no live rate feed in this app -- update by hand when rates move
# materially. POP/EV deliberately stay zero-drift (retail-platform convention,
# and drift over a 30-45 DTE window is second-order next to the vol input).
RISK_FREE_RATE_PCT = 4.0


def realized_vol_yang_zhang(bars: pd.DataFrame, window: int = 30) -> float | None:
    """Yang-Zhang (2000) range-based realized-vol estimator, annualized percent.

    Combines overnight (close-to-open) variance, open-to-close variance, and a
    drift-independent Rogers-Satchell term -- the most statistically efficient
    of the common range estimators, and the only common one that handles both
    overnight jumps and intraday drift (unlike a plain close-to-close stdev).
    """
    if bars is None or bars.empty or window < 2 or len(bars) < window + 1:
        return None
    for col in ("open", "high", "low", "close"):
        if col not in bars.columns:
            return None
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    prev_c = c.shift(1)

    log_oc = np.log(o / prev_c)   # overnight return
    log_co = np.log(c / o)        # intraday return
    log_ho = np.log(h / o)
    log_lo = np.log(l / o)
    rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)   # Rogers-Satchell term

    sigma_o2 = log_oc.rolling(window).var(ddof=1)
    sigma_c2 = log_co.rolling(window).var(ddof=1)
    sigma_rs2 = rs.rolling(window).mean()

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    sigma_yz2 = (sigma_o2 + k * sigma_c2 + (1 - k) * sigma_rs2).dropna()
    if sigma_yz2.empty:
        return None
    last = float(sigma_yz2.iloc[-1])
    if not np.isfinite(last) or last < 0:
        return None
    return float(np.sqrt(last * PERIODS_PER_YEAR) * 100.0)


# arch is a hard dependency (see requirements.txt) but the import is guarded so
# a missing/broken install degrades to "no forecast" rather than breaking the
# whole options endpoint -- same defensive pattern as the LLM/ML layers.
try:
    from arch import arch_model
    _ARCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _ARCH_AVAILABLE = False

MIN_BARS_FOR_GARCH = 250   # ~1 trading year; GARCH(1,1) is unstable on less


def forecast_vol_garch(bars: pd.DataFrame, horizon_days: int) -> float | None:
    """Annualized % volatility FORECAST over the next `horizon_days`, from a
    GARCH(1,1) fit on daily close-to-close returns.

    This is the statistically correct thing to compare an option's implied vol
    against: implied vol prices the expected variance over the contract's life,
    so the honest 'rich/cheap' test is IV vs a *forward* vol forecast for that
    same horizon -- not vs trailing realized vol (a rear-view mirror). GARCH is
    the standard workhorse here because vol mean-reverts and clusters: after a
    spike it forecasts elevated-but-decaying vol, after calm it forecasts a
    drift back up toward the long-run level.

    Returns None (caller falls back to the realized-vol comparison) if arch is
    unavailable, history is too short, or the fit doesn't converge."""
    if not _ARCH_AVAILABLE or bars is None or bars.empty or horizon_days < 1:
        return None
    if "close" not in bars.columns or len(bars) < MIN_BARS_FOR_GARCH:
        return None
    close = bars["close"].dropna()
    if len(close) < MIN_BARS_FOR_GARCH:
        return None

    # arch expects returns scaled to ~O(1) for numerical stability; percent
    # daily log returns (×100) is the library's own documented convention.
    rets = np.log(close / close.shift(1)).dropna() * 100.0
    if len(rets) < MIN_BARS_FOR_GARCH:
        return None
    try:
        model = arch_model(rets, mean="Constant", vol="GARCH", p=1, q=1, dist="normal")
        res = model.fit(disp="off", show_warning=False)
        fc = res.forecast(horizon=horizon_days, reindex=False)
        # variance forecast is per-day (in %^2); average over the horizon gives
        # the mean daily variance the option "sees", then annualize.
        daily_var = float(np.asarray(fc.variance.iloc[-1]).mean())
    except Exception:  # noqa: BLE001 - non-convergence / singular fit -> no forecast
        return None
    if not np.isfinite(daily_var) or daily_var <= 0:
        return None
    # daily_var is in percent^2 (returns were ×100), so sqrt gives daily % vol;
    # annualize by sqrt(252). Result is already in percent.
    return float(np.sqrt(daily_var * PERIODS_PER_YEAR))


def bsm_greeks(
    spot: float, strike: float, iv_pct: float | None, dte: int | None, right: str, rate: float = 0.0,
) -> dict[str, float | None]:
    """Standard closed-form Black-Scholes-Merton Greeks (European, no dividend
    yield). `rate` defaults to 0 -- this app has no risk-free-rate/dividend
    pipeline, so treat these as an approximation, not a precision pricer.
    Returns {"delta", "gamma", "theta", "vega"} (theta per calendar day, vega
    per 1 vol point)."""
    if not spot or spot <= 0 or not strike or strike <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None}
    if iv_pct is None or iv_pct <= 0 or dte is None or dte <= 0:
        return {"delta": None, "gamma": None, "theta": None, "vega": None}

    sigma = iv_pct / 100.0
    t = dte / 365.0
    sqrt_t = np.sqrt(t)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    pdf_d1 = norm.pdf(d1)
    is_call = right.upper() == "CALL"

    if is_call:
        delta = norm.cdf(d1)
        theta = (
            -(spot * pdf_d1 * sigma) / (2 * sqrt_t)
            - rate * strike * np.exp(-rate * t) * norm.cdf(d2)
        ) / 365.0
    else:
        delta = norm.cdf(d1) - 1.0
        theta = (
            -(spot * pdf_d1 * sigma) / (2 * sqrt_t)
            + rate * strike * np.exp(-rate * t) * norm.cdf(-d2)
        ) / 365.0
    gamma = pdf_d1 / (spot * sigma * sqrt_t)
    vega = spot * pdf_d1 * sqrt_t / 100.0

    return {"delta": float(delta), "gamma": float(gamma), "theta": float(theta), "vega": float(vega)}


def bsm_price(
    spot: float, strike: float, iv_pct: float | None, dte: int | None, right: str, rate: float = 0.0,
) -> float | None:
    """European Black-Scholes-Merton option price (no dividend). Companion to
    bsm_greeks -- the strategist reads real broker quotes, so live code never
    needs this, but the synthetic backtest (no historical chains exist) must
    price modeled contracts off a vol assumption. Same d1/d2 as bsm_greeks."""
    if not spot or spot <= 0 or not strike or strike <= 0:
        return None
    if iv_pct is None or iv_pct <= 0 or dte is None or dte <= 0:
        return None
    sigma = iv_pct / 100.0
    t = dte / 365.0
    sqrt_t = np.sqrt(t)
    d1 = (np.log(spot / strike) + (rate + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    disc = np.exp(-rate * t)
    if right.upper() == "CALL":
        px = spot * norm.cdf(d1) - strike * disc * norm.cdf(d2)
    else:
        px = strike * disc * norm.cdf(-d2) - spot * norm.cdf(-d1)
    return float(max(px, 0.0))


def bsm_price_path(spot, strike: float, iv_pct: float, dte, right: str, rate: float = 0.0):
    """Vectorized BSM price over parallel spot/dte arrays -- reprices one leg
    along a realized price path for the early-management backtest. Where dte<=0
    (expiry) the price collapses to intrinsic, so the path's last point equals
    payoff_at_expiry's premium-free intrinsic (verified in tests)."""
    spot = np.asarray(spot, dtype=float)
    dte = np.asarray(dte, dtype=float)
    is_call = right.upper() == "CALL"
    intrinsic = np.maximum(spot - strike, 0.0) if is_call else np.maximum(strike - spot, 0.0)
    out = intrinsic.copy()
    live = (dte > 0) & (spot > 0) & (iv_pct is not None and iv_pct > 0)
    if np.any(live):
        sigma = iv_pct / 100.0
        t = dte[live] / 365.0
        sqrt_t = np.sqrt(t)
        s = spot[live]
        d1 = (np.log(s / strike) + (rate + 0.5 * sigma ** 2) * t) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        disc = np.exp(-rate * t)
        if is_call:
            px = s * norm.cdf(d1) - strike * disc * norm.cdf(d2)
        else:
            px = strike * disc * norm.cdf(-d2) - s * norm.cdf(-d1)
        out[live] = np.maximum(px, 0.0)
    return out


def default_price_grid(spot: float, pct: float = 0.4, n: int = 200) -> np.ndarray:
    """A dense price grid spanning spot +/- pct, for payoff/POP evaluation."""
    lo = max(spot * (1.0 - pct), 0.01)
    hi = spot * (1.0 + pct)
    return np.linspace(lo, hi, n)


def payoff_at_expiry(legs: list, price_grid: np.ndarray) -> np.ndarray:
    """Vectorized net P&L per share at expiry, summed across legs -- works for
    any leg combination (2-leg spreads, 4-leg condors, etc.), not just the
    strategy types this module currently constructs."""
    total = np.zeros_like(price_grid, dtype=float)
    for leg in legs:
        intrinsic = (
            np.maximum(price_grid - leg.strike, 0.0) if leg.right == "Call"
            else np.maximum(leg.strike - price_grid, 0.0)
        )
        premium = leg.price or 0.0
        if leg.action == "Buy":
            total += intrinsic - premium
        else:
            total += premium - intrinsic
    return total


def _terminal_masses(
    spot: float, iv_pct: float | None, dte: int | None, price_grid: np.ndarray,
) -> np.ndarray | None:
    """Probability mass per grid bin under a zero-drift lognormal terminal
    price distribution (ln(S_T/S0) ~ N(-0.5*sigma^2*T, sigma^2*T)) at the
    given ATM implied vol -- the standard retail-platform convention absent a
    reliable risk-free-rate/dividend pipeline. Bin i covers (-inf, grid[0])
    for i=0, [grid[i-1], grid[i]) for interior bins, [grid[-1], inf) for the
    last -- so the result has len(grid)+1 entries summing to 1."""
    if spot is None or spot <= 0 or iv_pct is None or iv_pct <= 0 or dte is None or dte <= 0:
        return None
    if len(price_grid) < 2:
        return None
    sigma = iv_pct / 100.0
    t = dte / 365.0
    mu = -0.5 * sigma ** 2 * t
    with np.errstate(divide="ignore"):
        d = (np.log(price_grid / spot) - mu) / (sigma * np.sqrt(t))
    cdf = norm.cdf(d)
    edges_cdf = np.concatenate(([0.0], cdf, [1.0]))
    return np.diff(edges_cdf)


def probability_of_profit(
    spot: float, iv_pct: float | None, dte: int | None, payoff: np.ndarray, price_grid: np.ndarray,
) -> float | None:
    """Probability the position is profitable at expiry (see _terminal_masses
    for the distributional assumption). An approximation, not a precise
    probability; treat it as directional."""
    masses = _terminal_masses(spot, iv_pct, dte, price_grid)
    if masses is None or len(payoff) != len(price_grid):
        return None

    # Classify each bin profitable/not using the payoff at its lower (left)
    # edge -- a reasonable approximation given a dense grid.
    profit_flags = np.empty(len(masses), dtype=bool)
    profit_flags[0] = payoff[0] > 0
    profit_flags[-1] = payoff[-1] > 0
    profit_flags[1:-1] = payoff[:-1] > 0

    pop = float(masses[profit_flags].sum() * 100.0)
    return max(0.0, min(100.0, pop))


def expected_pnl(
    spot: float, iv_pct: float | None, dte: int | None, payoff: np.ndarray, price_grid: np.ndarray,
) -> float | None:
    """Probability-weighted P&L per share at expiry under the same zero-drift
    lognormal used for POP. This is the principled companion to POP: a
    high-POP structure can still be negative-EV (small frequent wins, rare
    large losses), and vice versa. Under this flat-vol model, deviations from
    zero mostly reflect market skew/mid-pricing vs the model -- directional,
    not precise. Tail bins beyond the grid use the edge payoff (understates
    unbounded wings slightly)."""
    masses = _terminal_masses(spot, iv_pct, dte, price_grid)
    if masses is None or len(payoff) != len(price_grid):
        return None
    bin_payoff = np.empty(len(masses), dtype=float)
    bin_payoff[0] = payoff[0]
    bin_payoff[-1] = payoff[-1]
    bin_payoff[1:-1] = (payoff[:-1] + payoff[1:]) / 2.0
    return float(np.sum(masses * bin_payoff))
