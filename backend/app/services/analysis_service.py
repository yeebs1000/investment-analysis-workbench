"""AnalysisService: the read path. Pulls account/positions/market data from the
broker (read-only), runs the deterministic analytics, and returns typed models.

A single broker connection is reused behind a lock (OpenD calls are blocking and
not concurrency-safe); FastAPI runs sync endpoints in a threadpool, so the lock
serializes broker access while keeping the API responsive.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

from app.analytics import fundamental_quality
from app.analytics import indicators as ind
from app.analytics import options as options_engine
from app.analytics import performance
from app.analytics import risk, scoring, short_interest as short_interest_engine, technical
try:  # Moomoo SDK (moomoo-api/futu) is imported at moomoo_client top level, so
    # an IBKR/Tiger-only user who never installed it can still run the app.
    from app.brokers.moomoo_client import MoomooClient, MoomooError
except Exception:  # noqa: BLE001 - missing SDK, or import-time failure
    MoomooClient = None  # type: ignore

    class MoomooError(RuntimeError):  # fallback so `except MoomooError` still resolves
        """Placeholder used when the Moomoo SDK isn't installed."""
from app.config import settings
from app.data import finnhub_client, fred_client, local_watchlists, normalize
from app.data.normalize import fx_to_usd
from app.data.models import (
    Account,
    ChartSeries,
    OptimizerPlan,
    OptionsAnalysis,
    Position,
    PortfolioAnalysis,
    TechnicalAnalysis,
    WatchlistAnalysis,
    WatchlistGroup,
)
from app.services.cache import TTLCache

BARS_TTL = 600        # history klines: 10 min
SNAPSHOT_TTL = 30
ACCOUNT_TTL = 20
FUNDAMENTALS_TTL = 3600  # fundamentals move slowly; cache for an hour
WATCHLIST_MAX = 30    # cap symbols analyzed per watchlist to bound runtime
DEFAULT_LOCAL_GROUP = "My Watchlist"   # target when an app-local add names no group
ANALYZE_WORKERS = 6   # thread-pool width for bulk per-symbol analysis (see _analyze_rows)

# Supported chart/analysis timeframes -> the SDK ktype + a calendar lookback
# window sized to yield a few hundred bars (intraday history is permission- and
# volume-limited, so windows shrink as the bar size shrinks).
TIMEFRAMES: dict[str, dict] = {
    "day":   {"label": "Daily",   "lookback_days": 430,  "ppy": 252},
    "week":  {"label": "Weekly",  "lookback_days": 1825, "ppy": 52},
    "month": {"label": "Monthly", "lookback_days": 3650, "ppy": 12},
    "60m":   {"label": "1 hour",  "lookback_days": 90,   "ppy": 1638},
    "30m":   {"label": "30 min",  "lookback_days": 45,   "ppy": 3276},
    "15m":   {"label": "15 min",  "lookback_days": 25,   "ppy": 6552},
    "5m":    {"label": "5 min",   "lookback_days": 12,   "ppy": 19656},
}
DEFAULT_TF = "day"
BENCHMARK_CODE = "US.SPY"           # relative-strength / beta / alpha benchmark

# The next-higher timeframe used to confirm each analysis timeframe.
HIGHER_TF: dict[str, str] = {
    "5m": "30m", "15m": "60m", "30m": "day", "60m": "day",
    "day": "week", "week": "month",
}


def _tf(tf: str | None) -> str:
    return tf if tf in TIMEFRAMES else DEFAULT_TF


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _merge_positions(positions: list[Position]) -> list[Position]:
    """Combine same-symbol holdings across brokers into one row so portfolio
    concentration/exposure reflects the whole book (e.g. MSFT held in both)."""
    by_code: dict[str, Position] = {}
    order: list[str] = []
    for p in positions:
        if p.code not in by_code:
            by_code[p.code] = p.model_copy(deep=True)
            order.append(p.code)
            continue
        a = by_code[p.code]
        qty = a.qty + p.qty
        cost_basis = (a.cost_price or 0.0) * a.qty + (p.cost_price or 0.0) * p.qty
        pl = (a.pl_value or 0.0) + (p.pl_value or 0.0)
        a.market_value += p.market_value
        a.cost_price = (cost_basis / qty) if qty else a.cost_price
        a.pl_value = pl
        if a.today_pl_value is not None or p.today_pl_value is not None:
            a.today_pl_value = (a.today_pl_value or 0.0) + (p.today_pl_value or 0.0)
        a.last_price = a.last_price or p.last_price
        a.qty = qty
        cb = (a.cost_price or 0.0) * qty
        a.pl_ratio_pct = (pl / cb * 100.0) if cb else a.pl_ratio_pct
        a.broker = "+".join(sorted(set(a.broker.split("+")) | set(p.broker.split("+"))))
    return [by_code[c] for c in order]


def _merge_account(moo: Account, ib: Account) -> Account:
    """Fold IBKR balances into the Moomoo account (FX-approx into its currency)."""
    rate = fx_to_usd(ib.currency) / (fx_to_usd(moo.currency) or 1.0)
    moo.total_assets += (ib.total_assets or 0.0) * rate
    moo.cash += (ib.cash or 0.0) * rate
    moo.market_value += (ib.market_value or 0.0) * rate
    for ccy, v in ib.by_currency.items():
        if ccy in moo.by_currency:
            moo.by_currency[ccy]["cash"] += v.get("cash", 0.0)
            moo.by_currency[ccy]["assets"] += v.get("assets", 0.0)
        else:
            moo.by_currency[ccy] = dict(v)
    return moo


class AnalysisService:
    def __init__(self) -> None:
        # Moomoo is optional: None when its SDK isn't installed. Every read path
        # already tries Moomoo first inside try/except and falls back to
        # IBKR/Tiger, so a None client just means those fallbacks always run.
        self._client = MoomooClient() if MoomooClient is not None else None
        self._lock = threading.Lock()
        self._cache = TTLCache()
        self._ibkr = None
        if settings.ibkr_enabled:
            try:
                from app.brokers.ibkr_client import IBKRClient
                self._ibkr = IBKRClient()
            except Exception:  # noqa: BLE001 - app must run even if ib_async is missing
                self._ibkr = None
        self._tiger = None
        if settings.tiger_enabled:
            try:
                from app.brokers.tiger_client import TigerClient
                self._tiger = TigerClient()
            except Exception:  # noqa: BLE001 - app must run even if tigeropen is missing
                self._tiger = None

    # --- health / connectivity ----------------------------------------
    def configured_brokers(self) -> list[str]:
        """Brokers actually wired up on this instance (SDK present + enabled),
        derived from real client objects rather than raw .env flags."""
        out = []
        if self._client is not None:
            out.append("moomoo")
        if self._ibkr is not None:
            out.append("ibkr")
        if self._tiger is not None:
            out.append("tiger")
        return out

    def broker_status(self) -> dict[str, str]:
        """Live-ish connectivity per broker so /api/health reports what's
        actually reachable, not just what's configured. Cached briefly and
        kept non-blocking: Moomoo gets a fast-fail probe (OpenD being down is
        the common false-'ok' case), IBKR a non-blocking socket check, Tiger
        just 'configured' (no cheap probe without a billable request)."""
        def probe() -> dict[str, str]:
            status: dict[str, str] = {}
            if self._client is not None:
                try:
                    with self._lock:
                        self._client.get_watchlist_groups()  # cheap read; fails fast if OpenD down
                    status["moomoo"] = "connected"
                except Exception:  # noqa: BLE001
                    status["moomoo"] = "unreachable"
            if self._ibkr is not None:
                # Non-blocking: reflects a live socket if one's been opened by a
                # prior data call; "configured" before first use (connects lazily).
                status["ibkr"] = "connected" if self._ibkr.is_connected() else "configured"
            if self._tiger is not None:
                status["tiger"] = "configured"
            return status

        return self._cache.get_or_set("broker_status", 20, probe)

    # --- IBKR (read-only, optional) -----------------------------------
    def _ibkr_positions(self) -> list[Position]:
        if self._ibkr is None:
            return []
        try:
            return self._ibkr.get_positions()
        except Exception:  # noqa: BLE001 - degrade to Moomoo-only on any IBKR error
            return []

    def _ibkr_account(self) -> Account | None:
        if self._ibkr is None:
            return None
        try:
            return self._ibkr.get_account()
        except Exception:  # noqa: BLE001
            return None

    # --- Tiger (read-only, optional) ----------------------------------
    def _tiger_positions(self) -> list[Position]:
        if self._tiger is None:
            return []
        try:
            return self._tiger.get_positions()
        except Exception:  # noqa: BLE001 - degrade gracefully on any Tiger error
            return []

    def _tiger_account(self) -> Account | None:
        if self._tiger is None:
            return None
        try:
            return self._tiger.get_account()
        except Exception:  # noqa: BLE001
            return None

    # --- low-level broker access (locked) ------------------------------
    def _bars(self, code: str, tf: str = DEFAULT_TF) -> pd.DataFrame:
        tf = _tf(tf)
        lookback = TIMEFRAMES[tf]["lookback_days"]

        def fetch() -> pd.DataFrame:
            # Moomoo is the primary, fast source for every market it has quote
            # permissions for (this self-heals as you buy more Moomoo data subs).
            # IBKR is only a fallback, and only for markets known to need it.
            try:
                with self._lock:
                    raw = self._client.get_history_kline(code, ktype=tf, lookback_days=lookback)
                df = normalize.bars_from_kline(raw)
                if not df.empty:
                    return df
            except Exception:  # noqa: BLE001 - fall through to IBKR below
                pass
            # IBKR fallback: always for markets Moomoo can't serve, and for ANY
            # market when Moomoo itself is down (OpenD not running / no Moomoo
            # account) — the app must work with whichever broker is linked.
            # Cost: when BOTH sources fail, this adds IBKR's fast-fail timeout.
            if self._ibkr is not None:
                try:
                    df = self._ibkr.get_history_kline(code, ktype=tf, lookback_days=lookback)
                    if df is not None and not df.empty:
                        return normalize.bars_from_kline(df)
                except Exception:  # noqa: BLE001
                    pass
            # Tiger fallback: same idea, for a Tiger-only setup with no Moomoo
            # or IBKR linked (Tiger has no positions-market-data link otherwise).
            if self._tiger is not None:
                try:
                    df = self._tiger.get_history_kline(code, ktype=tf, lookback_days=lookback)
                    if df is not None and not df.empty:
                        return normalize.bars_from_kline(df)
                except Exception:  # noqa: BLE001
                    pass
            return pd.DataFrame()

        return self._cache.get_or_set(f"bars:{code}:{tf}", BARS_TTL, fetch)

    def _snapshots(self, codes: list[str]) -> dict[str, dict]:
        if not codes:
            return {}
        key = "snap:" + ",".join(sorted(codes))

        def fetch() -> dict[str, dict]:
            # Snapshot only feeds display-name resolution + an optional 52wk
            # flavor-text bullet (price/as_of always come from bars) -- so a
            # missing snapshot degrades a cosmetic, not the actual analysis.
            try:
                with self._lock:
                    df = self._client.get_snapshot(codes)
                if df is not None and not df.empty:
                    return {str(r["code"]): r.to_dict() for _, r in df.iterrows()}
            except Exception:  # noqa: BLE001 - fall through to IBKR/Tiger below
                pass
            if self._ibkr is not None:
                try:
                    df = self._ibkr.get_snapshot(codes)
                    if df is not None and not df.empty:
                        return {str(r["code"]): r.to_dict() for _, r in df.iterrows()}
                except Exception:  # noqa: BLE001
                    pass
            if self._tiger is not None:
                try:
                    df = self._tiger.get_snapshot(codes)
                    if df is not None and not df.empty:
                        return {str(r["code"]): r.to_dict() for _, r in df.iterrows()}
                except Exception:  # noqa: BLE001
                    pass
            return {}

        return self._cache.get_or_set(key, SNAPSHOT_TTL, fetch)

    # --- public read API ----------------------------------------------
    def get_account(self) -> Account:
        def fetch() -> Account:
            # Moomoo is one source, not a prerequisite: if OpenD isn't running
            # (or the user has no Moomoo account) the book is built from
            # whichever brokers ARE linked (IBKR/Tiger) instead of erroring.
            try:
                with self._lock:
                    df = self._client.get_account_info()
                acc = normalize.normalize_account(df)
            except Exception:  # noqa: BLE001
                acc = Account(currency="USD")
            ib_acc = self._ibkr_account()
            if ib_acc is not None:
                acc = _merge_account(acc, ib_acc)
            tiger_acc = self._tiger_account()
            if tiger_acc is not None:
                acc = _merge_account(acc, tiger_acc)
            rate = fx_to_usd(acc.currency)
            acc.total_assets_usd = round(acc.total_assets * rate, 2)
            acc.cash_usd = round(acc.cash * rate, 2)
            return acc

        return self._cache.get_or_set("account", ACCOUNT_TTL, fetch)

    @staticmethod
    def _invested_usd(acc) -> float | None:
        """USD market value of positions -- total minus cash. Cash is excluded
        from the performance-vs-SPY series so idle cash doesn't drag the account
        return below a fully-invested benchmark (see performance.py)."""
        if acc is None or acc.total_assets_usd is None:
            return None
        return round(acc.total_assets_usd - (acc.cash_usd or 0.0), 2)

    def get_positions(self) -> list[Position]:
        def fetch() -> list[Position]:
            positions: list[Position] = []
            try:
                with self._lock:
                    df = self._client.get_positions()
                positions = normalize.normalize_positions(df)
            except Exception:  # noqa: BLE001 - no Moomoo: use the other linked brokers
                pass
            positions += self._ibkr_positions()
            positions += self._tiger_positions()
            return _merge_positions(positions)

        return self._cache.get_or_set("positions", ACCOUNT_TTL, fetch)

    def _benchmark(self, tf: str):
        """Cached benchmark (SPY) bars for relative-strength / beta / alpha."""
        def fetch():
            try:
                return self._bars(BENCHMARK_CODE, tf)
            except Exception:  # noqa: BLE001
                return None
        return self._cache.get_or_set(f"bench:{tf}", BARS_TTL, fetch)

    def analyze_symbol(
        self, code: str, name: str | None = None, snapshot: dict | None = None,
        tf: str = DEFAULT_TF, with_context: bool = True, deep_context: bool = True,
    ) -> TechnicalAnalysis:
        tf = _tf(tf)
        try:
            bars = self._bars(code, tf)
        except (MoomooError, Exception) as exc:  # noqa: BLE001
            return TechnicalAnalysis(
                code=code, name=name or code, score=50.0,
                decision=technical.score_to_decision(50.0), confidence=0.0,
                confidence_label="Low", error=f"Data unavailable: {exc}",
            )
        if snapshot is None:
            try:
                snapshot = self._snapshots([code]).get(code)
            except Exception:  # noqa: BLE001
                snapshot = None
        resolved_name = name or (snapshot or {}).get("name") or code
        # Higher-timeframe confirmation: cheap trend read on the next TF up.
        htf = None
        htf_tf = HIGHER_TF.get(tf)
        if htf_tf:
            try:
                hbars = self._bars(code, htf_tf)
                hscore, hsummary = technical.trend_score(hbars)
                htf = {"label": TIMEFRAMES[htf_tf]["label"], "score": hscore, "summary": hsummary}
            except Exception:  # noqa: BLE001 - HTF is a bonus, never fatal
                htf = None
        ppy = TIMEFRAMES[tf]["ppy"]

        # Institutional + benchmark context (skipped in bulk runs for speed/quota).
        analyst = bench = None
        bseries = None
        if with_context:
            try:
                analyst = finnhub_client.recommendation(code)
            except Exception:  # noqa: BLE001
                analyst = None
            if code != BENCHMARK_CODE:
                bseries = self._benchmark(tf)
                if bseries is not None and not bseries.empty and len(bars) > 20:
                    beta, alpha, rel = ind.beta_alpha(bars["close"], bseries["close"], ppy)
                    bench = {"beta": beta, "alpha_pct": alpha, "rel_strength_pct": rel}

        # Optional trained ML forecast (see app/ml/) -- omitted entirely if no
        # model has been trained/activated, or ml deps aren't installed, or
        # anything about the computation fails. Never raises into this path.
        ml_signal = None
        if with_context:
            try:
                from app.ml import inference as ml_inference
                ml_signal = ml_inference.score_symbol(code, bars, bseries, ppy)
            except Exception:  # noqa: BLE001
                ml_signal = None

        ta = technical.analyze(code, str(resolved_name), bars, snapshot,
                               htf=htf, ppy=ppy, analyst=analyst, bench=bench,
                               ml_signal=ml_signal)
        # Short-interest / borrow read: free -- reuses the snapshot dict already
        # fetched above (Moomoo's raw columns, previously never read past
        # name/52wk). No extra network call, so this runs on every analysis,
        # including bulk portfolio/watchlist runs.
        ta.short_interest = short_interest_engine.read(snapshot)
        # Event/positioning context (Finnhub, US-skewed): EPS-surprise/PEAD, net
        # insider buying, and fundamental quality. Attached for display + the LLM
        # report; deliberately NOT fed into the deterministic score (no lookahead-
        # safe backtest, same honesty bar as elsewhere). Gated behind deep_context
        # (off for bulk portfolio/watchlist runs) -- each adds a Finnhub call, and
        # fanning 3 across 20+ holdings would blow the free-tier rate limit.
        if with_context and deep_context:
            try:
                ta.earnings_surprise = finnhub_client.earnings_surprises(code)
            except Exception:  # noqa: BLE001
                ta.earnings_surprise = None
            try:
                ta.insider = finnhub_client.insider_sentiment(code)
            except Exception:  # noqa: BLE001
                ta.insider = None
            try:
                # Discretionary open-market Form-4 detail merged onto the same
                # aggregate-MSPR dict -- a sharper read than the aggregate alone
                # (see finnhub_client.insider_transactions docstring).
                txns = finnhub_client.insider_transactions(code)
                if txns:
                    ta.insider = {**(ta.insider or {}), **txns}
            except Exception:  # noqa: BLE001
                pass
            try:
                raw_fund = self._fundamentals_with_moomoo_fallback(code)
                ta.fundamental_quality = fundamental_quality.score_quality(raw_fund)
                # Size/growth-stage conviction tilt (current fundamentals + price
                # relative strength as the "hot" proxy) -- underweight mega-caps,
                # overweight small-cap growers. Live-decision lens only; like
                # fundamental_quality it never feeds the historical ML.
                ta.growth_tilt = fundamental_quality.size_growth_tilt(raw_fund, ta.rel_strength_pct)
            except Exception:  # noqa: BLE001
                ta.fundamental_quality = None
                ta.growth_tilt = None
            self._refine_symbol_context(ta)
        mkt = normalize.market_of(code)
        ta.market = mkt
        ta.currency = normalize.MARKET_CURRENCY.get(mkt, "USD")
        ta.timeframe = tf
        return ta

    # How close an earnings date has to be before it becomes a risk alert.
    EARNINGS_ALERT_DAYS = 14
    # A positive EPS surprise whose fiscal period ended within this window can
    # plausibly explain a gap-led ramp (reports land weeks after period end).
    PEAD_ATTRIBUTION_DAYS = 120

    def _refine_symbol_context(self, ta: TechnicalAnalysis) -> None:
        """Deep-context refinement over already-fetched data: earnings
        attribution + position-aware advice for the entry-risk flag, the
        two-axis quality×timing verdict, the next earnings date, and one
        consolidated risk-alerts list. Never raises into the read path."""
        import datetime as _dt

        today = _dt.date.today()
        try:
            ta.next_earnings = finnhub_client.next_earnings(ta.code)
        except Exception:  # noqa: BLE001
            ta.next_earnings = None

        holds = False
        try:
            holds = any(p.code == ta.code for p in self.get_positions())
        except Exception:  # noqa: BLE001
            holds = False

        # Level-2 depth (needs the market's L2 quote permission + open market).
        # Pure context — shown as a chip and fed to the LLM, never scored. Tries
        # Moomoo first (the user's US stock/ETF L2 lives there), then IBKR's deep
        # book as a fallback. Same dict shape from either source.
        def _depth():
            if self._client is not None:
                try:
                    # ponytail: holds the broker lock through get_market_depth's
                    # ~1.2s subscribe wait, blocking other Moomoo calls that long.
                    # Fine today (single-symbol, 30s cache); move depth off the
                    # shared lock if concurrent depth requests get common.
                    with self._lock:
                        d = self._client.get_market_depth(ta.code)
                    if d is not None:
                        return d
                except Exception:  # noqa: BLE001 - no L2 permission / closed market
                    pass
            if self._ibkr is not None:
                try:
                    return self._ibkr.get_market_depth(ta.code)
                except Exception:  # noqa: BLE001
                    return None
            return None
        try:
            ta.order_book = self._cache.get_or_set(f"depth:{ta.code}", 30, _depth)
        except Exception:  # noqa: BLE001
            ta.order_book = None

        er = ta.entry_risk
        if er:
            # Earnings attribution: a gap-led ramp right after a genuine EPS
            # beat is a re-rating with PEAD drift behind it, not a narrative
            # chase — the flag softens to "stage in" instead of "avoid".
            if er.get("direction") == "up" and er.get("event_gap") and ta.earnings_surprise:
                last_pct = ta.earnings_surprise.get("last_surprise_pct")
                period = str(ta.earnings_surprise.get("last_period") or "")
                recent = False
                try:
                    recent = (today - _dt.date.fromisoformat(period)).days <= self.PEAD_ATTRIBUTION_DAYS
                except ValueError:
                    recent = False
                if last_pct is not None and last_pct > 0 and recent:
                    er["attribution"] = "earnings"
                    er["advice"] = (
                        f"The ramp is gap-led off a {last_pct:+.1f}% EPS beat — an earnings "
                        f"re-rating, and post-earnings drift tends to continue. Stage in "
                        f"rather than avoid, but don't pay any price: partial size here, "
                        f"the rest on a pullback."
                    )
            # Position-aware advice: for a name the user actually holds, an
            # extension is trim/collar territory (not entry advice), and a
            # flush is precisely when NOT to hit the market sell button.
            if holds and er.get("attribution") != "earnings":
                if er.get("direction") == "up":
                    er["advice"] = (
                        "You already hold this — the extension is not a reason to add. "
                        "This is where trimming into strength or collaring the position "
                        "(see the options view) beats chasing."
                    )
                else:
                    stop_txt = f"{ta.stop:,.2f}" if ta.stop else "the suggested stop"
                    er["advice"] = (
                        f"You hold this and it's in a flush — selling into it locks in the "
                        f"worst prints. Let the stop at {stop_txt} decide the exit, or check "
                        f"the collar in the options view to buy time through the storm."
                    )

        ta.verdict = scoring.two_axis_verdict(
            ta.fundamental_quality, ta.score, ta.entry_risk,
            stop=ta.stop, ema20=ta.indicators.get("ema20"),
        )

        # One consolidated list of every active risk flag — the signals already
        # exist, scattered; a decision wants them in one place.
        alerts: list[str] = []
        if er:
            alerts.append(f"{er['label']}: {'; '.join(er['reasons'])}")
        if ta.next_earnings and ta.next_earnings.get("date"):
            try:
                dte = (_dt.date.fromisoformat(ta.next_earnings["date"]) - today).days
                if 0 <= dte <= self.EARNINGS_ALERT_DAYS:
                    alerts.append(
                        f"Earnings {ta.next_earnings['date']} (in {dte}d) — expect volatility; "
                        f"prefer defined-risk structures into the print."
                    )
            except ValueError:
                pass
        surp = ta.earnings_surprise or {}
        if (surp.get("last_surprise_pct") or 0) < 0:
            alerts.append(
                f"Last earnings missed estimates ({surp['last_surprise_pct']:+.1f}%) — "
                f"post-earnings drift is a headwind."
            )
        ins = ta.insider or {}
        if ins.get("direction") == "net selling":
            alerts.append(
                f"Insiders net selling over the last {ins.get('months', '?')} months "
                f"(MSPR {ins.get('net_mspr')})."
            )
        # Size/growth conviction tilt -> a position-sizing steer + its drivers.
        gt = ta.growth_tilt
        if gt and gt.get("label") not in (None, "Neutral"):
            mult = gt.get("sizing_multiplier")
            steer = f" (size ~{mult:g}x the technical suggestion)" if mult else ""
            alerts.append(f"{gt['label']} on size/growth{steer}: {'; '.join(gt.get('reasons', []))}")
            # Tilt the suggested position size so 'overweight/underweight' is
            # concrete, not just prose. Kept transparent -- the multiplier and
            # its reasons are shown above; the raw technical read is unchanged.
            if ta.kelly_sizing_pct is not None and mult:
                ta.kelly_sizing_pct = round(ta.kelly_sizing_pct * mult, 2)
        try:
            macro = self._cache.get_or_set(
                "macro_regime", FUNDAMENTALS_TTL, fred_client.macro_regime,
            )
            if macro and macro.get("regime") in ("risk-off", "cautious"):
                alerts.append(f"Macro {macro['regime']}: {macro.get('implication', '')}")
        except Exception:  # noqa: BLE001
            pass
        ta.risk_alerts = alerts

    def _analyze_rows(
        self, rows: list[tuple[str, str | None]], tf: str, snaps: dict,
        *, with_context: bool, deep_context: bool,
    ) -> list[TechnicalAnalysis]:
        """Run analyze_symbol across many symbols concurrently, preserving order.

        The broker lock (self._lock) still serializes OpenD access inside each
        call; what actually overlaps is the per-symbol Finnhub network calls and
        the indicator CPU work — which is where a cold-cache bulk run spends its
        time. analyze_symbol swallows its own errors (returns a TechnicalAnalysis
        with .error set), so no worker raises and order-preserving map is safe."""
        def one(row: tuple[str, str | None]) -> TechnicalAnalysis:
            code, name = row
            return self.analyze_symbol(
                code, name, snaps.get(code), tf=tf,
                with_context=with_context, deep_context=deep_context,
            )
        if len(rows) <= 1:
            return [one(r) for r in rows]
        with ThreadPoolExecutor(max_workers=min(ANALYZE_WORKERS, len(rows))) as ex:
            return list(ex.map(one, rows))

    def analyze_portfolio(self, tf: str = DEFAULT_TF) -> PortfolioAnalysis:
        tf = _tf(tf)
        account = self.get_account()
        positions = self.get_positions()
        codes = [p.code for p in positions]
        snaps = {}
        try:
            snaps = self._snapshots(codes)
        except Exception:  # noqa: BLE001
            snaps = {}
        # deep_context off: portfolio fans across all holdings, and per-name
        # earnings/insider/fundamentals Finnhub calls would rate-limit + stall.
        rows = [(p.code, p.name) for p in positions]
        analyzed = self._analyze_rows(rows, tf, snaps, with_context=True, deep_context=False)
        analyses: dict[str, TechnicalAnalysis] = {p.code: ta for p, ta in zip(positions, analyzed)}
        result = risk.analyze_portfolio(account, positions, analyses)
        result.timeframe = tf
        # Fund framing: benchmark trailing return + average book score.
        try:
            bseries = self._benchmark(tf)
            if bseries is not None and not bseries.empty:
                ret = ind.roc(bseries["close"], min(252, len(bseries) - 1))
                result.risk.benchmark = {
                    "code": BENCHMARK_CODE,
                    "return_pct": round(ret, 1) if ret is not None else None,
                    "label": f"{BENCHMARK_CODE} ({TIMEFRAMES[tf]['label']})",
                }
                # Log today's (equity, SPY) point for the performance-vs-SPY
                # tracker. Daily-deduped; failure here never disturbs the read.
                try:
                    performance.record_snapshot(
                        self._invested_usd(account), float(bseries["close"].iloc[-1]),
                        total_usd=account.total_assets_usd,
                    )
                except Exception:  # noqa: BLE001
                    pass
        except Exception:  # noqa: BLE001
            pass
        scored = [a.score for a in analyses.values() if not a.error]
        if scored:
            result.risk.avg_score = round(sum(scored) / len(scored), 1)
        # Macro regime context (FRED; None if no key/unavailable). Cached so it
        # doesn't add per-request latency once warm.
        try:
            result.macro_regime = self._cache.get_or_set(
                "macro_regime", FUNDAMENTALS_TTL, fred_client.macro_regime,
            )
        except Exception:  # noqa: BLE001
            result.macro_regime = None
        result.generated_at = _now_iso()
        return result

    def get_chart_series(self, code: str, lookback: int = 180, tf: str = DEFAULT_TF) -> ChartSeries:
        """Price OHLCV + EMA20/50/200, Bollinger bands and RSI14 as time-aligned
        arrays for charting. All values are pre-computed deterministically; the
        last `lookback` bars are returned (indicators use the full history)."""
        tf = _tf(tf)
        intraday = tf.endswith("m")
        name = code
        try:
            bars = self._bars(code, tf)
        except Exception as exc:  # noqa: BLE001
            return ChartSeries(code=code, name=code, error=f"Data unavailable: {exc}")
        if bars.empty:
            return ChartSeries(code=code, name=code, error="No price history.")
        try:
            snap = self._snapshots([code]).get(code)
            if snap and snap.get("name"):
                name = str(snap["name"])
        except Exception:  # noqa: BLE001
            pass

        close = bars["close"]
        ema20 = ind.ema(close, 20)
        ema50 = ind.ema(close, 50)
        ema200 = ind.ema(close, 200)
        bb_mid, bb_up, bb_lo, _pctb, _bw = ind.bollinger(close, 20, 2.0)
        rsi14 = ind.rsi(close, 14)

        def tail(s) -> list:
            v = s.tail(lookback)
            return [None if pd.isna(x) else round(float(x), 4) for x in v]

        n = min(lookback, len(bars))
        idx = bars.index[-n:]
        # intraday bars need the time-of-day; daily/weekly/monthly just the date.
        times = [str(t)[:16] if intraday else str(t)[:10] for t in idx]
        return ChartSeries(
            code=code, name=name, timeframe=tf, time=times,
            open=tail(bars["open"]), high=tail(bars["high"]),
            low=tail(bars["low"]), close=tail(close), volume=tail(bars["volume"]),
            ema20=tail(ema20), ema50=tail(ema50), ema200=tail(ema200),
            bb_upper=tail(bb_up), bb_mid=tail(bb_mid), bb_lower=tail(bb_lo),
            rsi14=tail(rsi14),
        )

    def optimize(
        self,
        tf: str = DEFAULT_TF,
        method: str = "heuristic",
        cap_pct: float = risk.CONCENTRATION_LIMIT_PCT,
        cash_target_pct: float = risk.DEFAULT_CASH_TARGET_PCT,
    ) -> "OptimizerPlan":
        tf = _tf(tf)
        pa = self.analyze_portfolio(tf=tf)
        cash = pa.account.cash_usd or 0.0

        bars_by_code: dict[str, pd.DataFrame] | None = None
        if method == "risk_aware":
            bars_by_code = {}
            for h in pa.holdings:
                code = h.position.code
                try:
                    bars_by_code[code] = self._bars(code, tf)
                except Exception:  # noqa: BLE001 - risk.py treats a missing code as thin-history
                    continue

        plan = risk.optimize_portfolio(
            pa, cash_usd=cash, cap_pct=cap_pct, cash_target_pct=cash_target_pct,
            now_iso=_now_iso(), method=method, bars_by_code=bars_by_code,
        )
        plan.timeframe = tf
        return plan

    # --- search & watchlist write (non-trade) -------------------------
    def search(self, query: str) -> list:
        from app.data.models import SearchResult
        q = (query or "").strip()
        if not q:
            return []
        # If it already looks like a Moomoo code, surface it directly first.
        results: list[SearchResult] = []
        if "." in q and q.split(".", 1)[0].upper() in normalize.MARKET_CURRENCY:
            results.append(SearchResult(code=q.upper(), name=q.upper()))
        for r in finnhub_client.search(q):
            if any(x.code == r["code"] for x in results):
                continue
            results.append(SearchResult(code=r["code"], name=r["name"],
                                        finnhub_symbol=r["finnhub_symbol"], type=r["type"]))
        return results[:15]

    def _fundamentals_with_moomoo_fallback(self, code: str) -> dict | None:
        """finnhub_client.fundamentals() output, with pe/pb/market-cap/dividend
        backfilled from the Moomoo snapshot when Finnhub has nothing for them --
        Finnhub's free tier skews US-listed, so HK/SG/JP/CN names often come
        back with these fields in `missing_fields` (or no data at all); Moomoo
        carries them for every market it has quote permission for.

        Market cap needs FX conversion: Moomoo returns it in the LISTING's
        local currency, verified live (HK.00700's total_market_val is ~4.34e12,
        matching Tencent's real HKD market cap, not a USD figure) -- pe/pb/
        dividend-yield are ratios/percentages and are currency-invariant, no
        conversion needed for those."""
        code_u = code.strip().upper()
        try:
            raw = finnhub_client.fundamentals(code_u)
        except Exception:  # noqa: BLE001
            raw = None
        try:
            snap = self._snapshots([code_u]).get(code_u) or {}
        except Exception:  # noqa: BLE001
            snap = {}
        if not snap:
            return raw

        ccy = normalize.MARKET_CURRENCY.get(normalize.market_of(code_u), "USD")
        # snap is a raw pandas-row dict -- a missing numeric cell is NaN, not
        # None, so every read here goes through normalize._f (NaN -> None).
        mcap_local = normalize._f(snap.get("total_market_val"))
        backfill = {
            "pe_ttm": normalize._f(snap.get("pe_ttm_ratio")),
            "pb": normalize._f(snap.get("pb_ratio")),
            "market_cap_musd": (mcap_local * fx_to_usd(ccy) / 1_000_000.0) if mcap_local is not None else None,
            "dividend_yield_pct": normalize._f(snap.get("dividend_ratio_ttm")),
        }
        raw = dict(raw) if raw else {"available_fields": [], "missing_fields": []}
        for key, val in backfill.items():
            if raw.get(key) is not None or val is None:
                continue
            raw[key] = round(val, 2)
            missing = [k for k in raw.get("missing_fields") or [] if k != key]
            raw["missing_fields"] = missing
            avail = raw.get("available_fields") or []
            raw["available_fields"] = avail if key in avail else [*avail, key]
        return raw

    def get_fundamentals(self, code: str):
        from app.data.models import FundamentalMetrics

        def fetch() -> FundamentalMetrics:
            code_u = code.strip().upper()
            try:
                data = self._fundamentals_with_moomoo_fallback(code_u)
            except Exception as exc:  # noqa: BLE001
                return FundamentalMetrics(code=code_u, error=f"Fundamentals unavailable: {exc}")
            if data is None:
                return FundamentalMetrics(
                    code=code_u, error="No fundamentals data available for this symbol.",
                )
            return FundamentalMetrics(code=code_u, **data)

        return self._cache.get_or_set(f"fund:{code.strip().upper()}", FUNDAMENTALS_TTL, fetch)

    def get_performance(self) -> dict:
        """Portfolio performance vs SPY from the persisted daily log. Ensures at
        least today's point is recorded first (so a fresh call isn't empty when
        the portfolio view hasn't run yet)."""
        try:
            acc = self.get_account()
            bseries = self._benchmark(DEFAULT_TF)
            if bseries is not None and not bseries.empty:
                performance.record_snapshot(
                    self._invested_usd(acc), float(bseries["close"].iloc[-1]),
                    total_usd=acc.total_assets_usd,
                )
        except Exception:  # noqa: BLE001
            pass
        return performance.compute_performance()

    def add_to_watchlist(self, code: str, group: str | None = None,
                         source: str | None = None) -> dict:
        """Add a symbol to a watchlist group. This is a benign list edit — NOT a
        trade and NOT an order; no money or positions are affected.

        `source="local"` targets the app-owned JSON store (works with any broker,
        auto-creates the group). Otherwise it targets a Moomoo group as before."""
        code = code.strip().upper()
        if source == "local" or self._client is None:
            # No Moomoo linked -> app-local list is the only place to add to.
            grp = group or DEFAULT_LOCAL_GROUP
            local_watchlists.add(grp, code)
            return {"status": "added", "code": code, "group": grp, "source": "local"}
        with self._lock:
            grp = group or self._client.default_watchlist_group()
            self._client.add_to_watchlist(grp, code)
        self._cache.clear()
        return {"status": "added", "code": code, "group": grp, "source": "moomoo"}

    def remove_from_watchlist(self, code: str, group: str) -> dict:
        """Remove a symbol from a LOCAL watchlist group. (Moomoo groups are
        read/add-only through this app — its API here has no remove.)"""
        code = code.strip().upper()
        local_watchlists.remove(group, code)
        return {"status": "removed", "code": code, "group": group, "source": "local"}

    def delete_watchlist(self, group: str) -> dict:
        """Delete a LOCAL watchlist group entirely."""
        local_watchlists.delete(group)
        return {"status": "deleted", "group": group, "source": "local"}

    def analyze_options(self, code: str, target_dte: int = 35) -> OptionsAnalysis:
        ta = self.analyze_symbol(code)
        if ta.error or ta.price is None:
            return OptionsAnalysis(code=code, name=ta.name, error=ta.error or "No price.")
        spot = ta.price
        bars = self._bars(code)

        # 1. choose an expiry near the target DTE (avoid 0-DTE). Tries Moomoo
        # first (its native chain + Greeks-via-snapshot path), then IBKR, then
        # Tiger -- each broker's get_option_expirations/get_option_contracts
        # methods return the same column shape, so nothing below this needs
        # to know which one actually served the request.
        picked = self._pick_option_expiry(code, target_dte)
        if picked is None:
            return OptionsAnalysis(
                code=code, name=ta.name, spot=spot,
                technical_decision=ta.decision, technical_score=ta.score,
                error=f"No options available for {code} (checked every linked broker).",
            )
        source, expiry, dte, usable = picked

        # 2-3. chain + Greeks for the near-money strikes of that expiry.
        contracts = self._option_contracts_from(source, code, expiry, spot)
        if contracts is None or contracts.empty:
            return OptionsAnalysis(
                code=code, name=ta.name, spot=spot,
                technical_decision=ta.decision, technical_score=ta.score,
                expiry_used=expiry, dte=dte, error="No near-the-money strikes found.",
            )

        # 3b. term structure: a SECOND, longer expiry (~2x DTE out) sampled just
        # for its ATM IV, so the strategist can see if this tenor's IV is an
        # event bump or the whole curve. Best-effort -- never blocks the result.
        # Reuses whichever broker served step 1, so the IV comparison is
        # apples-to-apples (not one broker's near tenor vs another's far one).
        next_expiry, next_atm_iv = None, None
        try:
            longer = usable[usable["dist"] >= dte + 20]
            if not longer.empty:
                np_pick = longer.iloc[(longer["dist"] - (dte * 2)).abs().argsort()].iloc[0]
                next_expiry = str(np_pick["strike_time"])
                next_contracts = self._option_contracts_from(source, code, next_expiry, spot)
                if next_contracts is not None and not next_contracts.empty:
                    next_atm_iv = self._atm_iv(next_contracts, spot)
        except Exception:  # noqa: BLE001 - term structure is a bonus, not required
            next_expiry, next_atm_iv = None, None

        # 4. holdings + account + earnings context.
        holds, shares = False, 0.0
        for p in self.get_positions():
            if p.code == code:
                holds, shares = True, p.qty
                break
        book_value_usd = None
        try:
            acc = self.get_account()
            book_value_usd = acc.total_assets_usd or acc.total_assets
        except Exception:  # noqa: BLE001
            book_value_usd = None
        earnings = finnhub_client.next_earnings(code)  # None => unknown, engine states so

        # market regime (benchmark vs 200dma) -- same deterministic signal the
        # options backtest validates. None (e.g. benchmark fetch failed) simply
        # disables the counter-regime gate rather than blocking the analysis.
        market_regime = None
        try:
            bench = self._benchmark(DEFAULT_TF)
            if bench is not None and not bench.empty:
                market_regime = options_engine.benchmark_regime(bench["close"])
        except Exception:  # noqa: BLE001
            market_regime = None

        # 5. strategist (analyst consensus, technical target/stop, plus the new
        # event/liquidity/sizing/term-structure context).
        result = options_engine.build_analysis(
            code=code, name=ta.name, as_of=ta.as_of, spot=spot,
            decision=ta.decision, score=ta.score, bars=bars,
            contracts=contracts, expiry=expiry, dte=dte, holds=holds, shares=shares,
            analyst=ta.analyst_consensus, stock_target=ta.target, stock_stop=ta.stop,
            confidence=ta.confidence, earnings=earnings, book_value_usd=book_value_usd,
            next_atm_iv_pct=next_atm_iv, next_expiry=next_expiry,
            market_regime=market_regime,
        )
        return result

    @staticmethod
    def _is_third_friday_impl(strike_time: str) -> bool:
        try:
            d = datetime.fromisoformat(str(strike_time)[:10]).date()
        except (ValueError, TypeError):
            return False
        return d.weekday() == 4 and 15 <= d.day <= 21   # Friday, 3rd week

    def _pick_option_expiry(self, code: str, target_dte: int):
        """Try Moomoo, then IBKR, then Tiger for expirations on `code`. Returns
        (source, expiry_str, dte, usable_expirations_df) for the first broker
        that has any, or None if none of the linked brokers do."""
        import pandas as pd

        for source, get_exp in (
            ("moomoo", self._moomoo_option_expirations),
            ("ibkr", self._ibkr_option_expirations),
            ("tiger", self._tiger_option_expirations),
        ):
            try:
                exp = get_exp(code)
            except Exception:  # noqa: BLE001
                continue
            if exp is None or exp.empty:
                continue
            exp = exp.copy()
            exp["dist"] = pd.to_numeric(exp["option_expiry_date_distance"], errors="coerce")
            usable = exp[exp["dist"] >= 10]
            if usable.empty and exp.empty:
                continue
            if not usable.empty:
                # Prefer the 3rd-Friday MONTHLY nearest the target: single-name
                # open interest concentrates there, while weeklies are near-empty
                # (verified live: COIN/ARM weekly OI ~1-5 fails the liquidity
                # gate). Only fall back to nearest-any if no monthly is within a
                # reasonable window of the target tenor.
                monthly = usable[usable["strike_time"].apply(self._is_third_friday_impl)]
                cand = monthly if not monthly.empty and \
                    (monthly["dist"] - target_dte).abs().min() <= 25 else usable
                pick = cand.iloc[(cand["dist"] - target_dte).abs().argsort()].iloc[0]
            else:
                pick = exp.iloc[exp["dist"].astype(float).argmax()]
            return source, str(pick["strike_time"]), int(pick["dist"]), usable
        return None

    def _moomoo_option_expirations(self, code: str):
        with self._lock:
            return self._client.get_option_expirations(code)

    def _ibkr_option_expirations(self, code: str):
        return self._ibkr.get_option_expirations(code) if self._ibkr is not None else None

    def _tiger_option_expirations(self, code: str):
        return self._tiger.get_option_expirations(code) if self._tiger is not None else None

    def _option_contracts_from(self, source: str, code: str, expiry: str, spot: float):
        if source == "moomoo":
            return self._option_contracts(code, expiry, spot)
        if source == "ibkr":
            return self._ibkr.get_option_contracts(code, expiry, spot) if self._ibkr else None
        if source == "tiger":
            return self._tiger.get_option_contracts(code, expiry, spot) if self._tiger else None
        return None

    def _option_contracts(self, code: str, expiry: str, spot: float):
        """Fetch one expiry's near-money chain + a live Greeks/quote snapshot,
        returned as the contracts DataFrame the options engine expects (or None
        if nothing near the money is available)."""
        import pandas as pd

        with self._lock:
            chain = self._client.get_option_chain(code, expiry, expiry)
        chain = chain.copy()
        chain["strike"] = pd.to_numeric(chain["strike_price"], errors="coerce")
        lo, hi = spot * 0.75, spot * 1.25
        near = chain[(chain["strike"] >= lo) & (chain["strike"] <= hi)].copy()
        near["dist"] = (near["strike"] - spot).abs()
        near = near.sort_values("dist").head(60)  # cap snapshot size
        if near.empty:
            return None

        codes = near["code"].tolist()
        with self._lock:
            snap = self._client.get_snapshot(codes)
        snap = snap.set_index("code")
        rows = []
        for _, r in near.iterrows():
            c = r["code"]
            s = snap.loc[c] if c in snap.index else {}
            bid = normalize._f(s.get("bid_price")) if hasattr(s, "get") else None
            ask = normalize._f(s.get("ask_price")) if hasattr(s, "get") else None
            last = normalize._f(s.get("last_price")) if hasattr(s, "get") else None
            # Prefer the bid/ask midpoint (a real, current two-sided quote) over the
            # last trade, which can be stale/off-market on a thin contract. Only
            # trust the quote when it's a sane two-sided market.
            mid = (bid + ask) / 2.0 if (bid is not None and ask is not None and 0 < bid <= ask) else None
            rows.append({
                "code": c,
                "right": str(r["option_type"]),
                "strike": float(r["strike"]),
                "delta": normalize._f(s.get("option_delta")) if hasattr(s, "get") else None,
                "iv": normalize._f(s.get("option_implied_volatility")) if hasattr(s, "get") else None,
                "price": mid if mid is not None else last,
                "bid": bid, "ask": ask, "last": last,
                "oi": normalize._f(s.get("option_open_interest")) if hasattr(s, "get") else None,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _atm_iv(contracts, spot: float) -> float | None:
        """Mean of the nearest-to-spot call and put IV -- the ATM IV read used
        for term-structure comparison."""
        if contracts is None or contracts.empty:
            return None
        ivs = []
        for right in ("CALL", "PUT"):
            side = contracts[contracts["right"].str.upper() == right].copy()
            side = side.dropna(subset=["iv"])
            side = side[side["iv"] > 0]
            if side.empty:
                continue
            row = side.iloc[(side["strike"] - spot).abs().argsort()].iloc[0]
            ivs.append(float(row["iv"]))
        return sum(ivs) / len(ivs) if ivs else None

    def list_watchlists(self) -> list[WatchlistGroup]:
        """Broker-side Moomoo groups plus app-local groups, merged. Moomoo is
        best-effort — an IBKR-only user (no OpenD) still gets their local lists."""
        out: list[WatchlistGroup] = []
        try:
            with self._lock:
                df = self._client.get_watchlist_groups()
            out = [WatchlistGroup(name=str(r["group_name"]), source="moomoo")
                   for _, r in df.iterrows()]
        except Exception:  # noqa: BLE001 - Moomoo may not be linked; local lists still work
            out = []
        seen = {g.name for g in out}
        for name in local_watchlists.groups():
            if name in seen:
                continue  # a Moomoo group of the same name wins; local dup hidden
            out.append(WatchlistGroup(
                name=name, source="local", count=len(local_watchlists.codes(name)),
            ))
        return out

    def analyze_watchlist(self, group: str, limit: int = WATCHLIST_MAX,
                          tf: str = DEFAULT_TF, source: str | None = None) -> WatchlistAnalysis:
        tf = _tf(tf)
        # Local when explicitly asked, when the name only exists locally, or when
        # there's no Moomoo client at all (nowhere else a broker group could live).
        if (source == "local" or self._client is None
                or (source is None and local_watchlists.has_group(group))):
            rows: list[tuple[str, str | None]] = [(c, None) for c in local_watchlists.codes(group)]
        else:
            with self._lock:
                wl = self._client.get_watchlist(group)
            rows = [(str(r["code"]), str(r.get("name") or r["code"])) for _, r in wl.iterrows()]
        rows = rows[:limit]
        codes = [c for c, _ in rows]
        snaps = {}
        try:
            snaps = self._snapshots(codes)
        except Exception:  # noqa: BLE001
            snaps = {}
        # bulk run -> skip per-name analyst/benchmark fetches to stay fast.
        items = self._analyze_rows(rows, tf, snaps, with_context=False, deep_context=True)
        errors = [f"{ta.code}: {ta.error}" for ta in items if ta.error]
        items = risk.rank_watchlist(items)
        return WatchlistAnalysis(
            group=group, items=items, generated_at=_now_iso(), errors=errors[:10]
        )


# Module-level singleton reused across requests.
service = AnalysisService()
