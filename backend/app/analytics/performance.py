"""Portfolio performance tracking vs the S&P 500 benchmark.

The stated goal is to BEAT SPY every year — and you can't manage what you don't
measure. Nothing else in the app persists your equity over time (every other
read is a live snapshot), so this module keeps a small daily log of
(account equity in USD, SPY close) and computes the accountability stats:
cumulative return vs SPY, alpha/beta, tracking error, information ratio, and
max drawdown.

Honesty constraints:
- History accrues FORWARD from first use — there is no backfill of your past
  equity (the brokers don't expose a clean point-in-time equity history here).
  So early on it plainly says "building history" instead of computing noise
  off two data points.
- One row per calendar day (last write wins), so intraday re-fetches don't
  inflate the sample.
- Equity is the FX-approx USD total already shown elsewhere; stated as approx.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from app.analytics import indicators as ind

MIN_DAYS_FOR_STATS = 20    # below this, ratios are noise -- report "building history"
PERIODS_PER_YEAR = 252.0


def _store_path() -> Path:
    d = Path(__file__).resolve().parents[2] / "data_store"
    d.mkdir(parents=True, exist_ok=True)
    return d / "performance.parquet"


def _load() -> pd.DataFrame:
    path = _store_path()
    if not path.exists():
        return pd.DataFrame(columns=["date", "equity_usd", "spy_close"]).set_index("date")
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df = df.set_index("date")
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def record_snapshot(equity_usd: float | None, spy_close: float | None,
                    on: date | None = None) -> None:
    """Append (or overwrite) today's equity + SPY close. No-ops on bad input so
    a snapshot failure never disturbs the request that triggered it."""
    if not equity_usd or equity_usd <= 0 or not spy_close or spy_close <= 0:
        return
    day = pd.Timestamp(on or date.today()).normalize()
    df = _load()
    df.loc[day, "equity_usd"] = float(equity_usd)
    df.loc[day, "spy_close"] = float(spy_close)
    df = df.sort_index()
    out = df.reset_index().rename(columns={"index": "date"})
    tmp = _store_path().with_suffix(".parquet.tmp")
    out.to_parquet(tmp, engine="pyarrow", index=False)
    tmp.replace(_store_path())


def compute_performance() -> dict:
    """Read the log and compute performance vs SPY. Always returns a dict with a
    `status` so the caller/UI can distinguish 'no data', 'building', 'ready'."""
    df = _load().dropna(subset=["equity_usd", "spy_close"])
    n = len(df)
    if n == 0:
        return {"status": "no_data", "days_tracked": 0,
                "message": "No performance history yet — it starts logging from your first portfolio view."}

    first, last = df.iloc[0], df.iloc[-1]
    acct_ret = (last["equity_usd"] / first["equity_usd"] - 1.0) * 100.0
    spy_ret = (last["spy_close"] / first["spy_close"] - 1.0) * 100.0
    common = {
        "status": "building" if n < MIN_DAYS_FOR_STATS else "ready",
        "days_tracked": n,
        "since": str(df.index[0].date()),
        "as_of": str(df.index[-1].date()),
        "equity_usd": round(float(last["equity_usd"]), 2),
        "account_return_pct": round(float(acct_ret), 2),
        "spy_return_pct": round(float(spy_ret), 2),
        "excess_return_pct": round(float(acct_ret - spy_ret), 2),
        "beating_spy": bool(acct_ret > spy_ret),
    }
    if n < MIN_DAYS_FOR_STATS:
        common["message"] = (
            f"Building history ({n}/{MIN_DAYS_FOR_STATS} days) — cumulative return vs SPY is "
            f"shown, but risk-adjusted ratios need more data to be meaningful."
        )
        return common

    # Risk-adjusted stats off daily returns (reuse the same primitives the
    # per-symbol engine uses, so the math is consistent across the app).
    beta, alpha_pct, _rel = ind.beta_alpha(df["equity_usd"], df["spy_close"], PERIODS_PER_YEAR)
    acct_r = ind.log_returns(df["equity_usd"]).dropna()
    spy_r = ind.log_returns(df["spy_close"]).dropna()
    idx = acct_r.index.intersection(spy_r.index)
    excess_daily = (acct_r.loc[idx] - spy_r.loc[idx])
    tracking_error = float(excess_daily.std(ddof=1) * (PERIODS_PER_YEAR ** 0.5) * 100.0) if len(excess_daily) > 2 else None
    ann_excess = float(excess_daily.mean() * PERIODS_PER_YEAR * 100.0) if len(excess_daily) > 2 else None
    info_ratio = (ann_excess / tracking_error) if (tracking_error and tracking_error > 0) else None

    common.update({
        "beta": round(beta, 2) if beta is not None else None,
        "alpha_pct": round(alpha_pct, 2) if alpha_pct is not None else None,
        "tracking_error_pct": round(tracking_error, 2) if tracking_error is not None else None,
        "information_ratio": round(info_ratio, 2) if info_ratio is not None else None,
        "max_drawdown_pct": round(ind.max_drawdown_pct(df["equity_usd"]) or 0.0, 2),
        "spy_max_drawdown_pct": round(ind.max_drawdown_pct(df["spy_close"]) or 0.0, 2),
    })
    return common
