"""Read-only Tiger Brokers (tigeropen) client.

SAFETY: like the Moomoo and IBKR clients, this module NEVER places or modifies
orders. It only reads account assets/positions and market data (quote client
methods -- bars, chain, greeks -- are all read-only lookups, never order
placement). Tiger holdings are mapped to Moomoo 'MARKET.SYMBOL' codes so they
flow through the existing Moomoo-sourced analytics unchanged, mirroring IBKR.

Tiger's QuoteClient (tigeropen.quote.quote_client) additionally serves as a
market-data FALLBACK -- bars, an options chain with greeks already computed
server-side, and a snapshot brief -- used only when Moomoo (and, for options,
IBKR too) can't serve the request. This is what makes a Tiger-only setup (no
Moomoo/IBKR linked) get real technical scores/charts/options instead of just
balances. NOT LIVE-TESTED against a real Tiger account (none available in
dev) -- built strictly against the column names documented in the installed
tigeropen SDK's own docstrings/source; verify against a real account before
trusting the numbers, especially the IV units noted below.

Optional: only constructed when settings.tiger_enabled is true and the SDK +
credentials are present. Any failure degrades to "no Tiger data" rather than
breaking the app — the same defensive posture as ibkr_client.

Auth (Tiger Open API): a tiger_id, an account number, and an RSA private key
file. Get them from the Tiger 'API' page; the private key never leaves this
machine and is read from the path in settings.tiger_private_key_path.
"""
from __future__ import annotations

import threading
from datetime import date, timedelta

from app.config import settings
from app.data import normalize
from app.data.models import Account, Position

try:  # pragma: no cover - import resolution depends on environment
    from tigeropen.tiger_open_config import TigerOpenClientConfig
    from tigeropen.trade.trade_client import TradeClient
    from tigeropen.quote.quote_client import QuoteClient
    from tigeropen.common.util.signature_utils import read_private_key
    _TIGER_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TIGER_AVAILABLE = False

import pandas as pd

# app tf -> Tiger's BarPeriod string (tigeropen.common.consts.BarPeriod values).
_TIGER_PERIOD = {
    "day": "day", "week": "week", "month": "month",
    "60m": "60min", "30m": "30min", "15m": "15min", "5m": "5min",
}


class TigerError(RuntimeError):
    """Raised when the Tiger API is unreachable or misconfigured."""


def _build_config() -> "TigerOpenClientConfig":
    if not _TIGER_AVAILABLE:
        raise TigerError("tigeropen is not installed (pip install tigeropen).")
    if not (settings.tiger_id and settings.tiger_account and settings.tiger_private_key_path):
        raise TigerError("Tiger not configured: need tiger_id, tiger_account, tiger_private_key_path.")
    cfg = TigerOpenClientConfig()
    cfg.tiger_id = settings.tiger_id
    cfg.account = settings.tiger_account
    cfg.private_key = read_private_key(settings.tiger_private_key_path)
    return cfg


def _moomoo_code_to_tiger_symbol(code: str) -> str:
    """Reverse of _tiger_to_moomoo_code below: strip the market prefix and
    Moomoo's HK zero-padding to get the bare symbol Tiger's own API expects."""
    _, _, sym = code.partition(".")
    if code.upper().startswith("HK.") and sym.isdigit():
        sym = str(int(sym))
    return sym


class TigerClient:
    """Persistent, read-only Tiger client: trade client for positions/assets,
    quote client for market-data fallback (bars/snapshot/options)."""

    def __init__(self) -> None:
        self._client: "TradeClient | None" = None
        self._quote: "QuoteClient | None" = None
        self._account = settings.tiger_account
        self._lock = threading.Lock()

    def _get_client(self) -> "TradeClient":
        if self._client is None:
            self._client = TradeClient(_build_config())
        return self._client

    def _get_quote(self) -> "QuoteClient":
        if self._quote is None:
            self._quote = QuoteClient(_build_config())
        return self._quote

    # --- public read API (thread-safe, read-only) ----------------------
    def get_positions(self) -> list[Position]:
        with self._lock:
            client = self._get_client()
            raw = client.get_positions(account=self._account)
        out: list[Position] = []
        for item in (raw or []):
            p = _position_from_tiger(item)
            if p is not None:
                out.append(p)
        return out

    def get_account(self) -> Account:
        with self._lock:
            client = self._get_client()
            assets = client.get_prime_assets(account=self._account)
        return _account_from_tiger(assets)

    # --- market data (fallback when Moomoo/IBKR can't serve) -----------
    def get_history_kline(self, code: str, ktype: str = "day",
                          lookback_days: int = 430) -> "pd.DataFrame | None":
        if not _TIGER_AVAILABLE:
            return None
        sym = _moomoo_code_to_tiger_symbol(code)
        period = _TIGER_PERIOD.get(ktype, "day")
        end = date.today()
        begin = end - timedelta(days=lookback_days)
        with self._lock:
            raw = self._get_quote().get_bars(
                sym, period=period, begin_time=begin.isoformat(),
                end_time=end.isoformat(), limit=1000,
            )
        if raw is None or raw.empty:
            return None
        out = raw.rename(columns={"time": "time_key"}).copy()
        out["time_key"] = pd.to_datetime(out["time_key"], unit="ms")
        return out[["time_key", "open", "high", "low", "close", "volume"]]

    def get_snapshot(self, symbols: list[str]) -> "pd.DataFrame | None":
        """Only feeds display-name resolution downstream -- price/as_of always
        come from bars, so a missing/partial snapshot is a cosmetic gap."""
        if not _TIGER_AVAILABLE:
            return None
        syms = [_moomoo_code_to_tiger_symbol(c) for c in symbols]
        by_sym = dict(zip(syms, symbols))
        with self._lock:
            briefs = self._get_quote().get_briefs(syms)
        if not briefs:
            return None
        rows = [
            {"code": by_sym.get(b.symbol, b.symbol), "name": getattr(b, "name", None) or b.symbol}
            for b in briefs
        ]
        return pd.DataFrame(rows)

    # --- options (fallback when Moomoo AND IBKR both have nothing) -----
    def get_option_expirations(self, code: str) -> "pd.DataFrame | None":
        if not _TIGER_AVAILABLE:
            return None
        sym = _moomoo_code_to_tiger_symbol(code)
        with self._lock:
            raw = self._get_quote().get_option_expirations(sym)
        if raw is None or raw.empty:
            return None
        today = date.today()
        out = raw.copy()
        exp_date = pd.to_datetime(out["date"]).dt.date
        out["option_expiry_date_distance"] = exp_date.apply(lambda d: (d - today).days)
        out["strike_time"] = out["date"]  # already 'YYYY-MM-DD'
        return out[["option_expiry_date_distance", "strike_time"]]

    def get_option_contracts(self, code: str, expiry: str, spot: float) -> "pd.DataFrame | None":
        """Near-the-money contracts with live quotes + greeks -- Tiger's chain
        call returns all of this in one shot (no separate snapshot step needed
        like Moomoo/IBKR require)."""
        if not _TIGER_AVAILABLE:
            return None
        sym = _moomoo_code_to_tiger_symbol(code)
        with self._lock:
            chain = self._get_quote().get_option_chain(sym, expiry, return_greek_value=True)
        if chain is None or chain.empty:
            return None
        chain = chain.copy()
        chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
        lo, hi = spot * 0.75, spot * 1.25
        near = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)].copy()
        if near.empty:
            return None
        rows = []
        for _, r in near.iterrows():
            bid = normalize._f(r.get("bid_price"))
            ask = normalize._f(r.get("ask_price"))
            last = normalize._f(r.get("latest_price"))
            mid = (bid + ask) / 2.0 if (bid is not None and ask is not None and 0 < bid <= ask) else None
            iv = normalize._f(r.get("implied_vol"))
            rows.append({
                "code": str(r.get("identifier") or f"{sym}{expiry}{r['put_call']}{r['strike']}"),
                "right": str(r["put_call"]).upper(),
                "strike": float(r["strike"]),
                "delta": normalize._f(r.get("delta")),
                # ponytail: tigeropen's documented example values for implied_vol
                # don't clearly confirm fraction-vs-percentage units (no real Tiger
                # account to verify against) -- assumed already a percentage,
                # matching Moomoo's convention. If a real account shows ATM IVs
                # off by ~100x, multiply/divide by 100 here.
                "iv": iv,
                "price": mid if mid is not None else last,
                "bid": bid, "ask": ask, "last": last,
                "oi": normalize._f(r.get("open_interest")),
            })
        return pd.DataFrame(rows) if rows else None


def _g(obj, *names, default=None):
    """First present attribute among `names` (tigeropen models vary by version)."""
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return default


def _num(value, default=None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# Tiger market code -> Moomoo market prefix. Tiger uses the same short codes.
_TIGER_MARKET = {"US": "US", "HK": "HK", "SG": "SG", "JP": "JP", "CN": "CN", "A": "CN"}


def _tiger_to_moomoo_code(symbol: str, market: str, currency: str) -> str:
    """Map a Tiger holding to a Moomoo 'MARKET.SYMBOL' code so it flows through
    the existing Moomoo-sourced analytics unchanged."""
    mkt = _TIGER_MARKET.get((market or "").upper())
    if not mkt:  # fall back to currency, same convention as the IBKR mapper
        mkt = normalize.IB_CCY_MARKET.get((currency or "USD").upper(), "US")
    sym = (symbol or "").upper().replace(" ", ".")
    if mkt == "HK" and sym.isdigit():
        sym = sym.zfill(5)            # Moomoo HK codes are zero-padded
    return f"{mkt}.{sym}"


def _position_from_tiger(item) -> Position | None:
    contract = _g(item, "contract")
    if contract is None:
        return None
    sec_type = str(_g(contract, "sec_type", "secType", default="STK")).upper()
    if sec_type not in ("STK", "ETF", ""):
        return None  # only equities/ETFs are analyzable via the Moomoo data path
    symbol = _g(contract, "symbol", default="")
    market = _g(contract, "market", default="")
    currency = str(_g(contract, "currency", default="USD")).upper()
    code = _tiger_to_moomoo_code(str(symbol), str(market), currency)

    qty = _num(_g(item, "quantity", "position"), 0.0) or 0.0
    avg_cost = _num(_g(item, "average_cost", "average_price", "averageCost"))
    last = _num(_g(item, "market_price", "latest_price", "marketPrice"))
    mkt_val = _num(_g(item, "market_value", "marketValue"), 0.0) or 0.0
    upnl = _num(_g(item, "unrealized_pnl", "unrealizedPnl"))
    cost_basis = (avg_cost or 0.0) * qty
    pl_ratio = ((upnl / cost_basis) * 100.0) if (upnl is not None and cost_basis) else None
    return Position(
        code=code,
        name=str(symbol) or code,
        market=normalize.market_of(code),
        currency=currency,
        broker="tiger",
        side="LONG" if qty >= 0 else "SHORT",
        qty=qty,
        cost_price=avg_cost,
        last_price=last,
        market_value=mkt_val,
        pl_ratio_pct=pl_ratio,
        pl_value=upnl,
    )


def _account_from_tiger(assets) -> Account:
    """Normalize Tiger's prime-assets response into our Account model. The SDK
    returns a list of PortfolioAccount objects; we read the summary segment
    defensively across field-name variants."""
    acc = assets[0] if isinstance(assets, (list, tuple)) and assets else assets
    summary = _g(acc, "summary", default=acc)
    ccy = str(_g(summary, "currency", default="USD")).upper()
    netliq = _num(_g(summary, "net_liquidation", "net_liquidation_value", "netLiquidation"), 0.0) or 0.0
    cash = _num(_g(summary, "cash", "cash_balance", "total_cash"), 0.0) or 0.0
    gross = _num(_g(summary, "gross_position_value", "grossPositionValue"), 0.0) or 0.0
    return Account(
        currency=ccy,
        total_assets=netliq,
        cash=cash,
        market_value=gross,
        available_funds=_num(_g(summary, "available_funds", "buying_power")),
        buying_power=_num(_g(summary, "buying_power", "buyingPower")),
        unrealized_pl=_num(_g(summary, "unrealized_pnl", "unrealizedPnl")),
        realized_pl=_num(_g(summary, "realized_pnl", "realizedPnl")),
        by_currency={ccy: {"cash": cash, "assets": netliq}},
    )


def _smoke() -> None:
    """Connectivity + market-data check. Run from backend/:  python -m app.brokers.tiger_client

    Exercises BOTH the account/positions path AND the newer market-data
    fallback methods (bars/snapshot/options), so the first real Tiger user can
    confirm they actually return usable data -- and eyeball the one open
    question: whether option IV comes back as a percentage (~30) or a fraction
    (~0.30). If it's a fraction, fix the `iv` mapping in get_option_contracts."""
    cli = TigerClient()

    print("=== account ===")
    try:
        print(cli.get_account().model_dump())
    except Exception as exc:  # noqa: BLE001
        print("  unavailable:", exc)

    print("\n=== positions ===")
    try:
        for p in cli.get_positions():
            print(" ", p.code, p.broker, p.qty, p.last_price, p.pl_value)
    except Exception as exc:  # noqa: BLE001
        print("  unavailable:", exc)

    probe = "US.AAPL"
    print(f"\n=== bars ({probe}) ===")
    try:
        bars = cli.get_history_kline(probe)
        print(bars.tail(3).to_string() if bars is not None else "None")
    except Exception as exc:  # noqa: BLE001
        print("  unavailable:", exc)

    print(f"\n=== snapshot ({probe}) ===")
    try:
        print(cli.get_snapshot([probe]))
    except Exception as exc:  # noqa: BLE001
        print("  unavailable:", exc)

    print(f"\n=== option expirations + near-money chain ({probe}) ===")
    try:
        exp = cli.get_option_expirations(probe)
        print(exp.head(5).to_string() if exp is not None else "None")
        if exp is not None and not exp.empty:
            row = exp[exp["option_expiry_date_distance"] >= 20]
            expiry = str((row.iloc[0] if not row.empty else exp.iloc[-1])["strike_time"])
            print(f"  chain for {expiry} (CHECK the `iv` column: percent ~30 vs fraction ~0.30):")
            chain = cli.get_option_contracts(probe, expiry, spot=230.0)
            print(chain.head(6).to_string() if chain is not None else "None")
    except Exception as exc:  # noqa: BLE001
        print("  unavailable:", exc)


if __name__ == "__main__":
    _smoke()
