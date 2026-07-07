"""Convert raw Moomoo/Futu DataFrames into typed domain models."""
from __future__ import annotations

import math
import re

import pandas as pd

from app.data.models import Account, Position

# Option-contract codes: underlying + YYMMDD + C/P + strike (e.g.
# "US.IREN260702C44000"). Used to keep derivatives out of equity-only
# pipelines (stock optimizer targets, ML training universe).
_OPTION_CODE_RE = re.compile(r"\d{6}[CP]\d+$")


def is_option_code(code: str | None) -> bool:
    return bool(code) and bool(_OPTION_CODE_RE.search(code.strip().upper()))

# Approximate FX rates -> USD, used only for cross-currency portfolio WEIGHTING
# (not for P&L, which is shown in each position's native currency). Clearly
# labelled "approx" in the UI; can be wired to a live FX feed later.
CURRENCY_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "HKD": 0.1282,
    "SGD": 0.74,
    "JPY": 0.0064,
    "CNY": 0.138,
    "CNH": 0.138,
    "AUD": 0.65,
    "CAD": 0.73,
    "MYR": 0.225,
    "EUR": 1.08,
    "GBP": 1.27,
}

# Map a Moomoo symbol prefix / position_market to a default currency.
MARKET_CURRENCY: dict[str, str] = {
    "US": "USD",
    "HK": "HKD",
    "SG": "SGD",
    "JP": "JPY",
    "SH": "CNY",
    "SZ": "CNY",
    "CN": "CNY",
    "AU": "AUD",
    "CA": "CAD",
    "MY": "MYR",
}


def fx_to_usd(currency: str) -> float:
    return CURRENCY_TO_USD.get((currency or "USD").upper(), 1.0)


def market_of(code: str) -> str:
    return code.split(".", 1)[0].upper() if "." in code else ""


# Map an IBKR contract currency to the Moomoo market prefix so an IBKR holding
# can be analyzed through the same (Moomoo-sourced) market-data pipeline.
IB_CCY_MARKET: dict[str, str] = {
    "USD": "US", "HKD": "HK", "SGD": "SG", "JPY": "JP",
    "CNH": "CN", "CNY": "CN", "AUD": "AU", "CAD": "CA",
}


def ib_to_moomoo_code(symbol: str, currency: str, primary_exchange: str = "") -> str:
    """Best-effort map of an IBKR stock to a Moomoo 'MARKET.SYMBOL' code."""
    mkt = IB_CCY_MARKET.get((currency or "USD").upper(), (currency or "US").upper())
    sym = (symbol or "").upper().replace(" ", ".")
    if mkt == "HK" and sym.isdigit():
        sym = sym.zfill(5)            # Moomoo HK codes are zero-padded, e.g. HK.00700
    return f"{mkt}.{sym}"


def _f(value, default=None) -> float | None:
    """Coerce a cell to float, treating Moomoo's 'N/A'/NaN sentinels as missing."""
    if value is None:
        return default
    if isinstance(value, str):
        if value.strip().upper() in ("N/A", "", "NONE"):
            return default
        try:
            return float(value)
        except ValueError:
            return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(f):
        return default
    return f


def normalize_account(df: pd.DataFrame) -> Account:
    if df is None or len(df) == 0:
        return Account()
    row = df.iloc[0].to_dict()
    by_currency: dict[str, dict[str, float]] = {}
    for ccy, cash_key, asset_key in (
        ("USD", "us_cash", "usd_assets"),
        ("HKD", "hk_cash", "hkd_assets"),
        ("SGD", "sg_cash", "sgd_assets"),
        ("JPY", "jp_cash", "jpy_assets"),
        ("CNH", "cn_cash", "cnh_assets"),
        ("AUD", "au_cash", "aud_assets"),
        ("CAD", "ca_cash", "cad_assets"),
        ("MYR", "my_cash", "myr_assets"),
    ):
        cash = _f(row.get(cash_key))
        assets = _f(row.get(asset_key))
        if (cash and cash != 0) or (assets and assets != 0):
            by_currency[ccy] = {"cash": cash or 0.0, "assets": assets or 0.0}

    return Account(
        currency=str(row.get("currency") or "HKD"),
        total_assets=_f(row.get("total_assets")) or 0.0,
        cash=_f(row.get("cash")) or 0.0,
        market_value=_f(row.get("market_val")) or 0.0,
        available_funds=_f(row.get("available_funds")),
        buying_power=_f(row.get("power")),
        unrealized_pl=_f(row.get("unrealized_pl")),
        realized_pl=_f(row.get("realized_pl")),
        risk_level=(str(row.get("risk_level")) if row.get("risk_level") not in (None, "N/A") else None),
        by_currency=by_currency,
    )


def normalize_positions(df: pd.DataFrame) -> list[Position]:
    if df is None or len(df) == 0:
        return []
    out: list[Position] = []
    for _, r in df.iterrows():
        code = str(r.get("code"))
        market = str(r.get("position_market") or market_of(code))
        currency = str(r.get("currency") or MARKET_CURRENCY.get(market, "USD"))
        out.append(
            Position(
                code=code,
                name=str(r.get("stock_name") or r.get("name") or code),
                market=market,
                currency=currency,
                side=str(r.get("position_side") or "LONG"),
                qty=_f(r.get("qty")) or 0.0,
                cost_price=_f(r.get("cost_price")),
                last_price=_f(r.get("nominal_price")),
                market_value=_f(r.get("market_val")) or 0.0,
                pl_ratio_pct=_f(r.get("pl_ratio")),
                pl_value=_f(r.get("pl_val")),
                today_pl_value=_f(r.get("today_pl_val")),
            )
        )
    return out


def bars_from_kline(df: pd.DataFrame) -> pd.DataFrame:
    """Return an OHLCV frame indexed by timestamp, ascending, numeric, NaNs dropped."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    out = pd.DataFrame(
        {
            "open": pd.to_numeric(df["open"], errors="coerce"),
            "high": pd.to_numeric(df["high"], errors="coerce"),
            "low": pd.to_numeric(df["low"], errors="coerce"),
            "close": pd.to_numeric(df["close"], errors="coerce"),
            "volume": pd.to_numeric(df["volume"], errors="coerce"),
        }
    )
    out.index = pd.to_datetime(df["time_key"], errors="coerce")
    out = out.dropna(subset=["close"]).sort_index()
    return out
