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
    source=<path> or "sp500" -> read codes from that file (broad list)."""
    if source and source != "holdings":
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
