"""Persisted historical OHLCV store for offline ML training.

The live app fetches bars on-demand with a short in-memory TTL cache (see
`AnalysisService._bars`) -- fine for serving requests, useless for training,
which needs a stable snapshot to build features/labels against across a
training run and across repeated retrains.

IMPORTANT (QFQ drift): Moomoo's `get_history_kline` returns forward-adjusted
(QFQ) prices, which are recomputed relative to "today" on every call -- a bar
for a date 6 months ago will show a different level today than it would have
shown 6 months ago if a split/dividend happened in between. That means this
store must be a periodically REPLACED SNAPSHOT, never an accumulating
append-log: merging old QFQ-adjusted rows with newly-fetched ones would create
a silent level discontinuity across any split boundary, corrupting every
indicator computed across it (SMA200, ATR, etc.). `update()` therefore always
overwrites the full stored series for a symbol.

This module never runs in the request-serving path -- it's driven only by
`app/ml/train.py`.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from app.data import normalize


def store_dir() -> Path:
    """`backend/data_store/`, resolved from this file's location (CWD-independent)."""
    return Path(__file__).resolve().parents[2] / "data_store"


def bars_dir() -> Path:
    d = store_dir() / "bars"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _bars_path(code: str) -> Path:
    # codes contain a '.', safe as a filename component (e.g. "US.AAPL.parquet")
    return bars_dir() / f"{code}.parquet"


class BarStore:
    """Thin, offline-only wrapper around a directory of per-symbol parquet files."""

    def load(self, code: str) -> pd.DataFrame:
        path = _bars_path(code)
        if not path.exists():
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.read_parquet(path)
        return df.set_index("date") if "date" in df.columns else df

    def save(self, code: str, bars: pd.DataFrame) -> None:
        path = _bars_path(code)
        tmp = path.with_suffix(".parquet.tmp")
        out = bars.copy()
        out.index.name = "date"
        out = out.reset_index()
        out.to_parquet(tmp, engine="pyarrow", index=False)
        tmp.replace(path)  # atomic on the same filesystem

    def update(self, code: str, client, lock, lookback_days: int = 430) -> pd.DataFrame:
        """Fetch fresh bars and REPLACE the stored series (see module docstring)."""
        with lock:
            raw = client.get_history_kline(code, ktype="day", lookback_days=lookback_days)
        bars = normalize.bars_from_kline(raw)
        if not bars.empty:
            self.save(code, bars)
        return bars

    def manifest(self, codes: list[str]) -> dict:
        """Bookkeeping snapshot: per-symbol date range / row count, for the training report."""
        m: dict[str, dict] = {}
        for code in codes:
            df = self.load(code)
            if df.empty:
                m[code] = {"rows": 0}
                continue
            m[code] = {
                "rows": len(df),
                "first_date": str(df.index.min().date()),
                "last_date": str(df.index.max().date()),
            }
        m["_generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        (store_dir() / "manifest.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
        return m
