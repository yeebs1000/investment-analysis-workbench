"""Read-only Moomoo (Futu OpenAPI) client.

SAFETY: This module never imports or calls any order-placing/-modifying API
(`place_order`, `modify_order`, `unlock_trade`). The trade context is used
exclusively for read queries — account info, positions, and order history.
That guarantees the whole system is decision-support only.

Run the connectivity smoke test from the `backend/` directory:

    python -m app.brokers.moomoo_client
"""
from __future__ import annotations

import contextlib
import time
from datetime import date, timedelta

import pandas as pd

# The Moomoo-branded SDK and the original Futu SDK expose an identical API.
# Prefer `moomoo`; fall back to `futu` if that is what's installed.
try:  # pragma: no cover - import resolution depends on environment
    from moomoo import (
        AuType,
        KLType,
        OpenQuoteContext,
        OpenSecTradeContext,
        RET_OK,
        SecurityFirm,
        TrdEnv,
        TrdMarket,
        UserSecurityGroupType,
    )

    SDK_NAME = "moomoo"
except ImportError:  # pragma: no cover
    from futu import (
        AuType,
        KLType,
        OpenQuoteContext,
        OpenSecTradeContext,
        RET_OK,
        SecurityFirm,
        TrdEnv,
        TrdMarket,
        UserSecurityGroupType,
    )

    SDK_NAME = "futu"

from app.config import settings


class MoomooError(RuntimeError):
    """Raised when an OpenD query returns a non-OK result."""


_TRD_ENV = {"REAL": TrdEnv.REAL, "SIMULATE": TrdEnv.SIMULATE}

_TRD_MARKET = {
    "US": TrdMarket.US,
    "HK": TrdMarket.HK,
    "CN": TrdMarket.CN,
}

# SecurityFirm members vary slightly by SDK version; resolve defensively.
_SEC_FIRM = {
    name: getattr(SecurityFirm, name)
    for name in ("FUTUSECURITIES", "FUTUINC", "FUTUAU", "FUTUSG")
    if hasattr(SecurityFirm, name)
}

_KTYPE = {
    "1m": KLType.K_1M,
    "5m": KLType.K_5M,
    "15m": KLType.K_15M,
    "30m": KLType.K_30M,
    "60m": KLType.K_60M,
    "day": KLType.K_DAY,
    "week": KLType.K_WEEK,
    "month": KLType.K_MON,
}


def _trd_env():
    return _TRD_ENV.get(settings.trd_env.upper(), TrdEnv.REAL)


def _trd_market():
    return _TRD_MARKET.get(settings.trd_market.upper(), TrdMarket.US)


def _sec_firm():
    return _SEC_FIRM.get(
        settings.security_firm.upper(),
        next(iter(_SEC_FIRM.values())),
    )


class MoomooClient:
    """Thin, read-only wrapper around the OpenD quote + trade contexts."""

    def __init__(self, host: str | None = None, port: int | None = None):
        self.host = host or settings.opend_host
        self.port = port or settings.opend_port
        self._quote: OpenQuoteContext | None = None
        self._trade: OpenSecTradeContext | None = None

    # --- lifecycle -----------------------------------------------------
    def connect(self) -> "MoomooClient":
        if self._quote is None:
            self._quote = OpenQuoteContext(host=self.host, port=self.port)
        if self._trade is None:
            self._trade = OpenSecTradeContext(
                filter_trdmarket=_trd_market(),
                host=self.host,
                port=self.port,
                security_firm=_sec_firm(),
            )
        return self

    def close(self) -> None:
        for ctx in (self._quote, self._trade):
            if ctx is not None:
                with contextlib.suppress(Exception):
                    ctx.close()
        self._quote = self._trade = None

    def __enter__(self) -> "MoomooClient":
        return self.connect()

    def __exit__(self, *exc) -> None:
        self.close()

    # --- helpers -------------------------------------------------------
    @staticmethod
    def _check(ret, data, what: str) -> pd.DataFrame:
        if ret != RET_OK:
            raise MoomooError(f"{what} failed: {data}")
        return data

    @property
    def quote(self) -> OpenQuoteContext:
        if self._quote is None:
            self.connect()
        return self._quote  # type: ignore[return-value]

    @property
    def trade(self) -> OpenSecTradeContext:
        if self._trade is None:
            self.connect()
        return self._trade  # type: ignore[return-value]

    # --- account / positions (READ ONLY) -------------------------------
    def get_account_info(self) -> pd.DataFrame:
        ret, data = self.trade.accinfo_query(trd_env=_trd_env())
        return self._check(ret, data, "accinfo_query")

    def get_positions(self) -> pd.DataFrame:
        ret, data = self.trade.position_list_query(trd_env=_trd_env())
        return self._check(ret, data, "position_list_query")

    def get_today_orders(self) -> pd.DataFrame:
        ret, data = self.trade.order_list_query(trd_env=_trd_env())
        return self._check(ret, data, "order_list_query")

    def get_today_deals(self) -> pd.DataFrame:
        ret, data = self.trade.deal_list_query(trd_env=_trd_env())
        return self._check(ret, data, "deal_list_query")

    # --- market data ---------------------------------------------------
    def get_snapshot(self, symbols: list[str]) -> pd.DataFrame:
        ret, data = self.quote.get_market_snapshot(symbols)
        return self._check(ret, data, "get_market_snapshot")

    def get_history_kline(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        ktype: str = "day",
        lookback_days: int = 430,
        max_count: int = 1000,
    ) -> pd.DataFrame:
        # IMPORTANT: without an explicit window OpenD returns the OLDEST bars in
        # the permitted history, not the most recent ones. Default to a window
        # ending today so we always analyze current data.
        if end is None:
            end = date.today().isoformat()
        if start is None:
            start = (date.fromisoformat(end) - timedelta(days=lookback_days)).isoformat()
        ret, data, _page = self.quote.request_history_kline(
            symbol,
            start=start,
            end=end,
            ktype=_KTYPE.get(ktype, KLType.K_DAY),
            autype=AuType.QFQ,
            max_count=max_count,
        )
        return self._check(ret, data, "request_history_kline")

    # --- options -------------------------------------------------------
    def get_option_expirations(self, code: str) -> pd.DataFrame:
        ret, data = self.quote.get_option_expiration_date(code=code)
        return self._check(ret, data, "get_option_expiration_date")

    def get_option_chain(self, code: str, start: str, end: str) -> pd.DataFrame:
        ret, data = self.quote.get_option_chain(code=code, start=start, end=end)
        return self._check(ret, data, "get_option_chain")

    # --- watchlists ----------------------------------------------------
    def get_watchlist_groups(self) -> pd.DataFrame:
        ret, data = self.quote.get_user_security_group(
            group_type=UserSecurityGroupType.CUSTOM
        )
        return self._check(ret, data, "get_user_security_group")

    def get_watchlist(self, group_name: str) -> pd.DataFrame:
        ret, data = self.quote.get_user_security(group_name)
        return self._check(ret, data, "get_user_security")

    def default_watchlist_group(self) -> str:
        """First custom watchlist group name (used as the default add target)."""
        df = self.get_watchlist_groups()
        if df is None or len(df) == 0:
            raise MoomooError("No watchlist groups found.")
        return str(df.iloc[0]["group_name"])

    def add_to_watchlist(self, group_name: str, code: str) -> None:
        """Add a security to a watchlist group. This is a benign list edit, NOT a
        trade/order — it uses the quote context's user-security write, never the
        trade context. No `place_order`/`unlock_trade` is involved."""
        from moomoo import ModifyUserSecurityOp  # local import; enum name stable
        ret, data = self.quote.modify_user_security(
            group_name, [code], ModifyUserSecurityOp.ADD
        )
        self._check(ret, data, "modify_user_security")

    # --- Level-2 order book --------------------------------------------
    def get_market_depth(self, code: str, num: int = 10) -> dict | None:
        """Level-2 order book snapshot, aggregated to bid/ask totals + imbalance.

        Returns the SAME shape as IBKRClient.get_market_depth so the two are
        interchangeable. Needs the market's L2 quote permission (the user has US
        stocks/ETFs at full 10-level depth); returns None on no permission,
        thin/one-sided book, or any failure — depth is bonus context, never a
        hard dependency. Subscribe→read→unsubscribe so it doesn't hold an
        order-book subscription slot beyond the call."""
        from moomoo import SubType
        ret, _msg = self.quote.subscribe([code], [SubType.ORDER_BOOK], is_first_push=False)
        if ret != RET_OK:
            return None  # e.g. "No permission to subscribe" for this market
        try:
            time.sleep(1.2)  # let the book push arrive
            ret, ob = self.quote.get_order_book(code, num=num)
            if ret != RET_OK or not isinstance(ob, dict):
                return None
            # each level is (price, volume, order_count, detail_dict)
            bids = [(float(l[0]), float(l[1])) for l in ob.get("Bid", []) if len(l) >= 2 and l[1]]
            asks = [(float(l[0]), float(l[1])) for l in ob.get("Ask", []) if len(l) >= 2 and l[1]]
        finally:
            with contextlib.suppress(Exception):
                self.quote.unsubscribe([code], [SubType.ORDER_BOOK])
        if not bids or not asks:
            return None
        bid_vol = sum(v for _, v in bids)
        ask_vol = sum(v for _, v in asks)
        total = bid_vol + ask_vol
        best_bid, best_ask = bids[0][0], asks[0][0]
        return {
            "bid_levels": len(bids), "ask_levels": len(asks),
            "bid_vol": round(bid_vol), "ask_vol": round(ask_vol),
            # % of visible size resting on the bid: >50 = buy-side pressure
            "imbalance_pct": round(bid_vol / total * 100.0, 1) if total else None,
            "best_bid": best_bid, "best_ask": best_ask,
            "spread_pct": round((best_ask - best_bid) / best_ask * 100.0, 3) if best_ask else None,
        }


def _smoke() -> None:
    """Connectivity check against a running OpenD gateway."""
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print(f"[smoke] SDK={SDK_NAME}  host={settings.opend_host}:{settings.opend_port}  "
          f"market={settings.trd_market}  firm={settings.security_firm}  env={settings.trd_env}")
    with MoomooClient() as cli:
        print("\n=== Account info ===")
        try:
            acc = cli.get_account_info()
            print(acc.to_string())
        except Exception as exc:  # noqa: BLE001
            print(f"  unavailable: {exc}")

        print("\n=== Positions ===")
        try:
            pos = cli.get_positions()
            cols = [c for c in ("code", "stock_name", "qty", "cost_price",
                                "nominal_price", "pl_ratio", "pl_val") if c in pos.columns]
            print(pos[cols].to_string() if cols else pos.to_string())
        except Exception as exc:  # noqa: BLE001
            print(f"  unavailable: {exc}")

        print("\n=== Snapshot (US.AAPL) ===")
        try:
            snap = cli.get_snapshot(["US.AAPL"])
            cols = [c for c in ("code", "last_price", "update_time",
                                "volume", "high_price", "low_price") if c in snap.columns]
            print(snap[cols].to_string() if cols else snap.to_string())
        except Exception as exc:  # noqa: BLE001
            print(f"  unavailable: {exc}")

        print("\n=== Watchlist groups ===")
        try:
            groups = cli.get_watchlist_groups()
            print(groups.to_string())
        except Exception as exc:  # noqa: BLE001
            print(f"  unavailable: {exc}")


if __name__ == "__main__":
    _smoke()
