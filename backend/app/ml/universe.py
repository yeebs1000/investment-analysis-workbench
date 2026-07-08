"""Resolve the ML training universe.

Two sources:
- "holdings" (default): the user's own live state -- current positions
  (Moomoo + IBKR merged) plus every symbol in every Moomoo watchlist group.
  Dynamic, but survivorship-biased: training only on names the user already
  chose to hold/watch tells you "was my existing judgment historically
  supported," not "what should I buy next," and starves the cross-section.
- a FILE path (e.g. app/ml/universe_sp500.txt): a broad, fixed list of
  MARKET.SYMBOL codes, one per line (# comments allowed). This is the fix for
  the survivorship/effective-sample problem -- hundreds of liquid names give
  the walk-forward real cross-sectional breadth. Residual caveat: a current
  index snapshot still omits dropped/delisted names, so bias is reduced, not
  eliminated (point-in-time membership needs a paid data source).

Either way, state the universe's provenance plainly in the training report.
"""
from __future__ import annotations

from pathlib import Path

from app.data.normalize import is_option_code
from app.services.analysis_service import BENCHMARK_CODE, service as default_service

HOLDINGS_MAX = 60    # cap for the account-derived universe
FILE_MAX = 600       # cap for a file-based universe (enough for the full S&P 500)

# Bundled broad-universe file shipped with the repo.
SP500_FILE = Path(__file__).with_name("universe_sp500.txt")

# Yahoo-style exchange codes (financedatabase inherits these) for the major
# tradable US venues -- excludes OTC/pink-sheet (PNK) and foreign cross-listings.
_US_EXCHANGES = ("NMS", "NYQ", "ASE", "NGM", "NCM")


def smallcap_universe(max_symbols: int | None = None, per_sector: int = 55) -> list[str]:
    """Sector-stratified sample of small/micro-cap US equities, from the
    `financedatabase` package's bundled local snapshot -- no API key, no rate
    limit, since it's a static dataset shipped in the pip package (its
    `market_cap` bucket is therefore a snapshot label, not live).

    The bundled sp500 universe is large/mega-cap by construction (that's what
    makes it the S&P 500), so it can't test whether the small-cap/hot-sector
    growth tilt (fundamental_quality.size_growth_tilt) actually holds up
    out-of-sample -- this gives the ML training universe real small-cap
    breadth, capped per sector so no single sector (Financials is ~40% of the
    raw pool) dominates the sample.
    """
    import financedatabase as fd  # ponytail: lazy import, ML-only optional dep

    df = fd.Equities().select(country="United States")
    df = df[
        df["exchange"].isin(_US_EXCHANGES)
        & (df["delisted"] == False)  # noqa: E712
        & df["market_cap"].isin(("Small Cap", "Micro Cap"))
    ]
    codes: list[str] = []
    seen: set[str] = set()
    for _, grp in df.groupby("sector"):
        for ticker in grp.index[:per_sector]:
            if ticker.isalnum():  # skip preferred/unit tickers like "AAIC^C"
                _dedup_add(codes, seen, f"US.{ticker}")
    return codes[: max_symbols or FILE_MAX]


def _dedup_add(codes: list[str], seen: set[str], code: str) -> None:
    # option contracts (e.g. US.IREN260702C44000) are excluded: the training
    # pipeline models equity/ETF OHLCV forward returns; an option's price path
    # is a different process entirely.
    c = (code or "").strip().upper()
    if c and c not in seen and not is_option_code(c):
        seen.add(c)
        codes.append(c)


def _load_universe_file(path: Path) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()  # strip comments + whitespace
        if line:
            _dedup_add(codes, seen, line)
    return codes


def resolve_universe(svc=None, source: str = "holdings", max_symbols: int | None = None) -> list[str]:
    """Return the training universe + the benchmark.

    source="holdings" -> account positions + watchlists (default).
    source="sp500" -> the bundled broad-list file.
    source="smallcap" -> sector-stratified small/micro-cap sample (financedatabase).
    source=<path> -> read codes from that file."""
    if source == "smallcap":
        codes = smallcap_universe(max_symbols=max_symbols)
        cap = max_symbols or FILE_MAX
    elif source and source != "holdings":
        path = SP500_FILE if source == "sp500" else Path(source)
        if not path.exists():
            raise FileNotFoundError(f"universe file not found: {path}")
        codes = _load_universe_file(path)
        cap = max_symbols or FILE_MAX
    else:
        svc = svc or default_service
        codes, seen = [], set()
        for p in svc.get_positions():
            _dedup_add(codes, seen, p.code)
        for grp in svc.list_watchlists():
            try:
                with svc._lock:  # same lock every other broker call goes through
                    wl = svc._client.get_watchlist(grp.name)
            except Exception:  # noqa: BLE001 - one bad group shouldn't kill the whole universe
                continue
            for _, row in wl.iterrows():
                _dedup_add(codes, seen, str(row.get("code", "")))
        cap = max_symbols or HOLDINGS_MAX

    universe = codes[:cap]
    # check the CAPPED list: the benchmark could have been truncated out, and
    # features (beta/relative-strength) are useless without it.
    if BENCHMARK_CODE not in universe:
        universe.append(BENCHMARK_CODE)
    return universe


# Back-compat for any caller importing the old constant name.
UNIVERSE_MAX = HOLDINGS_MAX


def demo() -> None:
    """Manual check, not run by CI: `financedatabase` fetches its dataset over
    the network on first use (nothing meaningful is bundled in the pip
    package), unlike every other test in this suite. Run directly:
    `python -m app.ml.universe`."""
    codes = smallcap_universe()
    assert 0 < len(codes) <= FILE_MAX, len(codes)
    assert all(c.startswith("US.") for c in codes)
    print(f"smallcap_universe: {len(codes)} symbols, e.g. {codes[:5]}")


if __name__ == "__main__":
    demo()
