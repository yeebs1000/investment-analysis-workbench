"""Read-only Interactive Brokers (IBKR) client via TWS / IB Gateway.

SAFETY: like the Moomoo client, this module never places or modifies orders. It
connects with `readonly=True` and only reads account values and portfolio
holdings. IBKR holdings are mapped to Moomoo 'MARKET.SYMBOL' codes so they flow
through the existing Moomoo-sourced analytics unchanged. IBKR also serves as a
market-data FALLBACK (bars, snapshot quotes, and now the options chain) used
only when Moomoo is unavailable or unlinked — the analytics layer itself is
still broker-agnostic and doesn't know which source fed it.

ib_async is not thread-safe and binds its IB object to the asyncio loop of the
thread that created it. FastAPI serves sync endpoints from a threadpool, so we
own a dedicated background thread that runs the event loop and the IB
connection; all calls are marshalled onto that loop via run_coroutine_threadsafe.
"""
from __future__ import annotations

import asyncio
import threading
from datetime import date

from app.config import settings
from app.data import normalize
from app.data.models import Account, Position

try:  # pragma: no cover - import resolution depends on environment
    from ib_async import IB, Option, Stock, util
except ImportError:  # pragma: no cover
    IB = None  # type: ignore
    Option = None  # type: ignore
    Stock = None  # type: ignore
    util = None  # type: ignore

import pandas as pd

# Moomoo market prefix -> (IBKR exchange, currency). This is an explicit
# allow-list: a market NOT listed here yields no IBKR contract (see
# _stock_params) rather than being force-mapped to a US contract. Guessing
# SMART/USD for, say, an LSE (GBP, priced in pence) holding would silently
# produce a wrong price -- failing cleanly to "no data" is the honest degrade.
# Add a market here (verify the exchange + price units) if you hold there.
_IB_EXCHANGE = {
    "JP": ("TSEJ", "JPY"),
    "SH": ("SEHKNTL", "CNH"),     # Shanghai-Connect (Northbound)
    "SZ": ("SEHKSZSE", "CNH"),    # Shenzhen-Connect
    "HK": ("SEHK", "HKD"),
    "SG": ("SGX", "SGD"),
    "AU": ("ASX", "AUD"),
    "CA": ("TSX", "CAD"),
    "US": ("SMART", "USD"),
}


def _stock_params(code: str) -> "tuple[str, str, str] | None":
    """(symbol, IBKR exchange, currency) for a MARKET.SYMBOL code, or None if the
    market isn't in the allow-list -- so callers skip IBKR for it instead of
    mis-resolving a foreign listing to a US contract."""
    mkt, _, sym = code.partition(".")
    mkt = mkt.upper()
    if mkt not in _IB_EXCHANGE:
        return None
    exch, ccy = _IB_EXCHANGE[mkt]
    if mkt == "HK" and sym.isdigit():
        sym = str(int(sym))            # IBKR HK uses the unpadded numeric symbol
    return sym, exch, ccy

# tf -> (IBKR durationStr, barSizeSetting)
_IB_BARS = {
    "day": ("2 Y", "1 day"),
    "week": ("5 Y", "1 week"),
    "month": ("10 Y", "1 month"),
    "60m": ("90 D", "1 hour"),
    "30m": ("45 D", "30 mins"),
    "15m": ("25 D", "15 mins"),
    "5m": ("12 D", "5 mins"),
}


class IBKRError(RuntimeError):
    """Raised when the IBKR gateway is unreachable or returns nothing."""


# Account-summary tags we surface.
_TAGS = ("NetLiquidation", "TotalCashValue", "AvailableFunds",
         "BuyingPower", "GrossPositionValue", "UnrealizedPnL", "RealizedPnL")


class IBKRClient:
    """Persistent, read-only connection to IB Gateway / TWS on a private loop."""

    def __init__(self, host: str | None = None, port: int | None = None,
                 client_id: int | None = None):
        self.host = host or settings.ibkr_host
        self.port = port or settings.ibkr_port
        self.client_id = client_id if client_id is not None else settings.ibkr_client_id
        self._ib = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    # --- private loop plumbing ----------------------------------------
    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop and self._thread and self._thread.is_alive():
            return self._loop
        if IB is None:
            raise IBKRError("ib_async is not installed.")
        loop = asyncio.new_event_loop()
        t = threading.Thread(target=loop.run_forever, name="ibkr-loop", daemon=True)
        t.start()
        self._loop, self._thread = loop, t
        return loop

    def _run(self, coro, timeout: float = 30.0):
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)

    async def _connect(self) -> None:
        if self._ib is None:
            self._ib = IB()
        if not self._ib.isConnected():
            await self._ib.connectAsync(
                self.host, self.port, clientId=self.client_id,
                readonly=True, timeout=15,
            )
            # 3 = "delayed if live isn't subscribed": IB automatically uses real-time
            # data when the account has a live subscription, and only falls back to
            # ~15-20min-delayed quotes when it doesn't -- never blocks on error 354.
            self._ib.reqMarketDataType(3)

    # --- public read API (thread-safe) --------------------------------
    def connect(self) -> "IBKRClient":
        with self._lock:
            self._run(self._connect())
        return self

    def is_connected(self) -> bool:
        """Non-blocking: True only if a live socket to the gateway already
        exists. Used by the health probe so it never pays the connect timeout."""
        return self._ib is not None and self._ib.isConnected()

    def close(self) -> None:
        with self._lock:
            if self._ib is not None and self._loop is not None:
                try:
                    self._run(self._disconnect(), timeout=5)
                except Exception:  # noqa: BLE001
                    pass
            if self._loop is not None:
                self._loop.call_soon_threadsafe(self._loop.stop)
            self._ib = None

    async def _disconnect(self) -> None:
        if self._ib is not None and self._ib.isConnected():
            self._ib.disconnect()

    def get_account(self) -> Account:
        with self._lock:
            rows = self._run(self._account_rows())
        return _account_from_summary(rows)

    async def _account_rows(self) -> list[tuple[str, str, str]]:
        await self._connect()
        summary = await self._ib.accountSummaryAsync()
        return [(av.tag, av.value, av.currency) for av in summary if av.tag in _TAGS]

    def get_positions(self) -> list[Position]:
        with self._lock:
            items = self._run(self._portfolio())
        return [p for p in (_position_from_item(it) for it in items) if p is not None]

    async def _portfolio(self) -> list:
        await self._connect()
        return list(self._ib.portfolio())

    # --- market data (for markets Moomoo can't serve, e.g. JP/SZ) ------
    def get_history_kline(self, code: str, ktype: str = "day",
                          lookback_days: int = 430,
                          duration: str | None = None) -> "pd.DataFrame | None":
        if Stock is None:
            return None
        params = _stock_params(code)
        if params is None:
            return None            # market not in the allow-list -> skip IBKR
        sym, exch, ccy = params
        default_duration, bar_size = _IB_BARS.get(ktype, _IB_BARS["day"])
        # duration override (e.g. "7 Y") lets a backtest pull deep history the
        # default per-ktype window (day="2 Y") is too short for.
        duration = duration or default_duration
        with self._lock:
            # Tight timeout: this only runs as a fallback after Moomoo has already
            # failed, so a slow/unpermissioned IBKR market shouldn't stall the request.
            return self._run(self._history(sym, exch, ccy, duration, bar_size), timeout=15)

    def get_market_depth(self, code: str, rows: int = 10) -> dict | None:
        """Level-2 order book snapshot, aggregated to bid/ask totals + imbalance.

        Needs an IBKR deep-book market data subscription and a live market;
        returns None on any failure (no permission, closed market, no gateway) —
        depth is bonus context, never a hard dependency."""
        if Stock is None:
            return None
        params = _stock_params(code)
        if params is None:
            return None
        sym, exch, ccy = params
        try:
            with self._lock:
                return self._run(self._depth(sym, exch, ccy, rows), timeout=15)
        except Exception:  # noqa: BLE001
            return None

    async def _depth(self, sym, exch, ccy, rows) -> dict | None:
        await self._connect()
        contract = Stock(sym, exch, ccy)
        try:
            await self._ib.qualifyContractsAsync(contract)
        except Exception:  # noqa: BLE001
            pass
        smart = exch == "SMART"   # aggregate depth across US exchanges
        ticker = self._ib.reqMktDepth(contract, numRows=rows, isSmartDepth=smart)
        try:
            await asyncio.sleep(2.0)   # let the book populate
            bids = [(float(d.price), float(d.size)) for d in (ticker.domBids or []) if d.size]
            asks = [(float(d.price), float(d.size)) for d in (ticker.domAsks or []) if d.size]
        finally:
            try:
                self._ib.cancelMktDepth(contract, isSmartDepth=smart)
            except Exception:  # noqa: BLE001
                pass
        if not bids or not asks:
            return None
        bid_vol = sum(s for _, s in bids)
        ask_vol = sum(s for _, s in asks)
        total = bid_vol + ask_vol
        best_bid, best_ask = bids[0][0], asks[0][0]
        return {
            "bid_levels": len(bids), "ask_levels": len(asks),
            "bid_vol": round(bid_vol), "ask_vol": round(ask_vol),
            # % of visible size sitting on the bid: >50 = buy-side pressure
            "imbalance_pct": round(bid_vol / total * 100.0, 1) if total else None,
            "best_bid": best_bid, "best_ask": best_ask,
            "spread_pct": round((best_ask - best_bid) / best_ask * 100.0, 3) if best_ask else None,
        }

    # --- snapshot (fallback; only feeds display-name + 52wk-high/low flavor
    # text -- the technical score/price itself always comes from bars) -----
    def get_snapshot(self, symbols: list[str]) -> "pd.DataFrame | None":
        if Stock is None:
            return None
        with self._lock:
            return self._run(self._snapshot_rows(symbols), timeout=15)

    async def _snapshot_rows(self, symbols: list[str]):
        await self._connect()
        rows = []
        for code in symbols:
            params = _stock_params(code)
            if params is None:
                continue           # unknown market -> just omit it from the snapshot
            sym, exch, ccy = params
            contract = Stock(sym, exch, ccy)
            try:
                await self._ib.qualifyContractsAsync(contract)
            except Exception:  # noqa: BLE001
                pass
            ticker = self._ib.reqMktData(contract, snapshot=True)
            rows.append((code, sym, contract, ticker))
        await asyncio.sleep(2.0)
        out = []
        for code, sym, contract, ticker in rows:
            out.append({
                "code": code,
                "name": sym,  # IBKR snapshot has no long name; the code/symbol is the best available
                "highest52weeks_price": ticker.high52week if ticker.high52week and ticker.high52week > 0 else None,
                "lowest52weeks_price": ticker.low52week if ticker.low52week and ticker.low52week > 0 else None,
            })
            try:
                self._ib.cancelMktData(contract)
            except Exception:  # noqa: BLE001
                pass
        return pd.DataFrame(out) if out else None

    # --- options (fallback when Moomoo has no expirations for this code) ---
    def get_option_expirations(self, code: str) -> "pd.DataFrame | None":
        """Expirations for `code`'s chain, shaped like Moomoo's
        get_option_expirations (option_expiry_date_distance + strike_time
        columns) so analyze_options() needs no broker-specific branch after
        this call succeeds."""
        if Stock is None or Option is None:
            return None
        params = _stock_params(code)
        if params is None:
            return None
        sym, exch, ccy = params
        with self._lock:
            chain = self._run(self._option_chain_meta(sym, exch, ccy), timeout=15)
        if chain is None:
            return None
        today = date.today()
        rows = []
        for exp in sorted(chain.expirations):
            try:
                d = date(int(exp[:4]), int(exp[4:6]), int(exp[6:8]))
            except (ValueError, IndexError):
                continue
            rows.append({
                "option_expiry_date_distance": (d - today).days,
                "strike_time": f"{d.year:04d}-{d.month:02d}-{d.day:02d}",
            })
        return pd.DataFrame(rows) if rows else None

    def get_option_contracts(self, code: str, expiry: str, spot: float) -> "pd.DataFrame | None":
        """Near-the-money contracts for one expiry with live quotes + model
        greeks, shaped to match Moomoo's _option_contracts output exactly
        (code, right, strike, delta, iv, price, bid, ask, last, oi) so the
        broker-agnostic options engine downstream needs no changes."""
        if Stock is None or Option is None:
            return None
        params = _stock_params(code)
        if params is None:
            return None
        sym, exch, ccy = params
        expiry_ib = expiry.replace("-", "")  # 'YYYY-MM-DD' -> IB's 'YYYYMMDD'
        with self._lock:
            return self._run(self._option_quotes(sym, exch, ccy, expiry_ib, spot), timeout=25)

    async def _option_chain_meta(self, sym, exch, ccy):
        """Underlying's option-chain metadata (expirations, strikes, the
        exchange/multiplier/tradingClass to build contracts with). None if
        the account has no options permission or the symbol isn't optionable."""
        await self._connect()
        underlying = Stock(sym, exch, ccy)
        try:
            await self._ib.qualifyContractsAsync(underlying)
        except Exception:  # noqa: BLE001
            pass
        if not underlying.conId:
            return None
        chains = await self._ib.reqSecDefOptParamsAsync(sym, "", "STK", underlying.conId)
        if not chains:
            return None
        return next((c for c in chains if c.exchange == "SMART"), chains[0])

    async def _option_quotes(self, sym, exch, ccy, expiry_ib, spot):
        chain = await self._option_chain_meta(sym, exch, ccy)
        if chain is None or not chain.strikes:
            return None
        lo, hi = spot * 0.75, spot * 1.25
        near = sorted((s for s in chain.strikes if lo <= s <= hi), key=lambda s: abs(s - spot))
        near = near[:20]  # cap contract count (x2 rights = up to 40 live quote requests)
        if not near:
            return None
        contracts = [
            Option(sym, expiry_ib, strike, right, chain.exchange, chain.multiplier or "100", ccy,
                   tradingClass=chain.tradingClass)
            for strike in near for right in ("C", "P")
        ]
        try:
            qualified = [c for c in await self._ib.qualifyContractsAsync(*contracts) if c.conId]
        except Exception:  # noqa: BLE001
            qualified = [c for c in contracts if c.conId]
        if not qualified:
            return None
        tickers = [self._ib.reqMktData(c) for c in qualified]
        try:
            await asyncio.sleep(2.5)  # let quotes + model greeks populate
            rows = []
            for c, t in zip(qualified, tickers):
                bid = t.bid if t.bid and t.bid > 0 else None
                ask = t.ask if t.ask and t.ask > 0 else None
                last = t.last if t.last and t.last > 0 else None
                mid = (bid + ask) / 2.0 if (bid is not None and ask is not None and bid <= ask) else None
                greeks = t.modelGreeks
                rows.append({
                    "code": c.localSymbol or str(c.conId),
                    "right": "CALL" if str(c.right).upper().startswith("C") else "PUT",
                    "strike": float(c.strike),
                    "delta": greeks.delta if greeks and greeks.delta is not None else None,
                    # IB's impliedVol is a fraction (0.25); the app's options engine
                    # displays/compares IV as a percentage (25.3) -- rescale here.
                    "iv": greeks.impliedVol * 100.0 if greeks and greeks.impliedVol else None,
                    "price": mid if mid is not None else last,
                    "bid": bid, "ask": ask, "last": last,
                    "oi": t.openInterest if t.openInterest and t.openInterest > 0 else None,
                })
        finally:
            for c in qualified:
                try:
                    self._ib.cancelMktData(c)
                except Exception:  # noqa: BLE001
                    pass
        return pd.DataFrame(rows) if rows else None

    async def _history(self, sym, exch, ccy, duration, bar_size):
        await self._connect()
        contract = Stock(sym, exch, ccy)
        try:
            await self._ib.qualifyContractsAsync(contract)
        except Exception:  # noqa: BLE001 - use the unqualified contract as-is
            pass
        bars = await self._ib.reqHistoricalDataAsync(
            contract, endDateTime="", durationStr=duration,
            barSizeSetting=bar_size, whatToShow="TRADES", useRTH=True, formatDate=1,
        )
        if not bars:
            return None
        df = util.df(bars)
        if df is None or df.empty:
            return None
        df = df.rename(columns={"date": "time_key"})
        df["time_key"] = df["time_key"].astype(str)
        return df[["time_key", "open", "high", "low", "close", "volume"]]


def _num(value, default=None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _account_from_summary(rows: list[tuple[str, str, str]]) -> Account:
    vals: dict[str, float | None] = {}
    ccy = "USD"
    for tag, value, currency in rows:
        vals[tag] = _num(value)
        if currency and currency != "BASE":
            ccy = currency
    netliq = vals.get("NetLiquidation") or 0.0
    cash = vals.get("TotalCashValue") or 0.0
    return Account(
        currency=ccy,
        total_assets=netliq,
        cash=cash,
        market_value=vals.get("GrossPositionValue") or 0.0,
        available_funds=vals.get("AvailableFunds"),
        buying_power=vals.get("BuyingPower"),
        unrealized_pl=vals.get("UnrealizedPnL"),
        realized_pl=vals.get("RealizedPnL"),
        by_currency={ccy: {"cash": cash, "assets": netliq}},
    )


def _position_from_item(item) -> Position | None:
    c = item.contract
    if getattr(c, "secType", "") not in ("STK", "ETF"):
        return None  # only equities/ETFs are analyzable via the Moomoo data path
    currency = (c.currency or "USD").upper()
    code = normalize.ib_to_moomoo_code(c.symbol, currency, getattr(c, "primaryExchange", ""))
    qty = _num(item.position) or 0.0
    avg_cost = _num(item.averageCost)
    last = _num(item.marketPrice)
    mkt_val = _num(item.marketValue) or 0.0
    upnl = _num(item.unrealizedPNL)
    cost_basis = (avg_cost or 0.0) * qty
    pl_ratio = ((upnl / cost_basis) * 100.0) if (upnl is not None and cost_basis) else None
    return Position(
        code=code,
        name=getattr(c, "symbol", code),
        market=normalize.market_of(code),
        currency=currency,
        broker="ibkr",
        side="LONG" if qty >= 0 else "SHORT",
        qty=qty,
        cost_price=avg_cost,
        last_price=last,
        market_value=mkt_val,
        pl_ratio_pct=pl_ratio,
        pl_value=upnl,
    )


def _smoke() -> None:
    cli = IBKRClient()
    print(f"[ibkr smoke] {cli.host}:{cli.port} clientId={cli.client_id}")
    cli.connect()
    acc = cli.get_account()
    print("account:", acc.model_dump())
    for p in cli.get_positions():
        print("position:", p.code, p.broker, p.qty, p.last_price, p.pl_value)
    cli.close()


if __name__ == "__main__":
    _smoke()
