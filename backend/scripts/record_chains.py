"""Daily option-chain snapshot recorder.

No historical option chains exist for this account -- the backtest prices
synthetic ones. This recorder builds the real archive going forward: once a
day (during US regular hours) it snapshots the near-money chain for every
symbol in the watch file and appends a dated parquet under
data_store/chains/<YYYY-MM-DD>/<code>.parquet with bid/ask/iv/delta/OI --
exactly the columns the strategist consumes, plus spot/expiry/dte/source.

In 6-12 months this archive supports what the synthetic backtest cannot:
real-quote entry premiums, fill-location tests, and liquidity filters.

Usage:  python -m scripts.record_chains          (from backend/, venv python)
Idempotent per day: symbols already recorded today are skipped, so a re-run
after a partial failure only fetches what's missing.

Schedule (Windows Task Scheduler, 23:00 local = late US morning):
  schtasks /Create /TN "TechnicalOptimiser\\RecordChains" /SC DAILY /ST 23:00 ^
    /TR "cmd /c cd /d <backend dir> && .venv\\Scripts\\python.exe -m scripts.record_chains"
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

# 49 CALENDAR days ~= the validated 35-TRADING-day tenor (the original 35 here
# was calendar -- a shorter tenor than the backtest validated). The DTE x delta
# plateau grid also favors the longer cell (+23.4% at DTE45-trading vs +21.7%).
TARGET_DTE = 49
WATCH_FILE = BACKEND / "data_store" / "chain_watch.txt"
CHAINS_DIR = BACKEND / "data_store" / "chains"

# Starter universe (seeds chain_watch.txt on first run; edit that file to
# change yours). Layers: index ETFs + liquid mega-caps, sector-stratified
# S&P (5/GICS sector), mid-caps (2/sector), liquid small-caps, high-IV
# names, NASDAQ depth, S&P breadth, and liquid Russell names -- all
# verified optionable at add time. ~218 names; recorder paces ~2s/symbol
# (~35-40 min/day).
DEFAULT_WATCH = [
    "US.SPY", "US.QQQ", "US.IWM", "US.AAPL", "US.MSFT", "US.NVDA",
    "US.AMZN", "US.GOOGL", "US.META", "US.TSLA", "US.AMD", "US.AVGO",
    "US.JPM", "US.XOM", "US.UNH", "US.LLY", "US.V", "US.COST",
    "US.NFLX", "US.CRM", "US.MSTR", "US.FUTU", "US.NVO", "US.GRAB",
    "US.TEM", "US.IREN", "US.ADM", "US.AIG", "US.ALB", "US.AOS",
    "US.APA", "US.BG", "US.BSX", "US.BXP", "US.C", "US.CHRW",
    "US.COIN", "US.CPT", "US.D", "US.DECK", "US.DIS", "US.DLR",
    "US.EA", "US.ECHO", "US.EOG", "US.EQIX", "US.ERIE", "US.EVRG",
    "US.EXC", "US.FCX", "US.FTV", "US.GEV", "US.GIS", "US.GM",
    "US.HONA", "US.HRL", "US.IFF", "US.JBHT", "US.JBL", "US.LYV",
    "US.MCD", "US.MGM", "US.MPWR", "US.MSCI", "US.NEM", "US.NKE",
    "US.NUE", "US.PRU", "US.PSKY", "US.PSX", "US.REG", "US.SJM",
    "US.SO", "US.SYK", "US.T", "US.TDY", "US.UHS", "US.VLO",
    "US.VTRS", "US.XEL", "US.ZBH", "US.ACI", "US.CARG", "US.DK",
    "US.EVCM", "US.FA", "US.FROG", "US.FRT", "US.HWKN", "US.IPAR",
    "US.KNTK", "US.LUMN", "US.MTRN", "US.NSA", "US.RNST", "US.SR",
    "US.STOK", "US.UA", "US.UGI", "US.ADV", "US.AEVA", "US.BRCC",
    "US.BYND", "US.CURI", "US.CVGI", "US.CWH", "US.IMKTA", "US.IZEA",
    "US.MGNI", "US.MARA", "US.RIOT", "US.CLSK", "US.HUT", "US.PLTR",
    "US.SMCI", "US.AFRM", "US.SOFI", "US.RBLX", "US.DKNG", "US.CVNA",
    "US.UPST", "US.HOOD", "US.ROKU", "US.SNAP", "US.NIO", "US.RIVN",
    "US.LCID", "US.PLUG", "US.IONQ", "US.RGTI", "US.SOUN", "US.ARM",
    "US.MRNA", "US.ENPH", "US.CELH", "US.GME", "US.AMC", "US.APP",
    "US.ACHR", "US.JOBY", "US.RKLB", "US.LUNR", "US.ASTS", "US.BBAI",
    "US.OKLO", "US.SMR", "US.ADBE", "US.INTC", "US.CSCO", "US.QCOM",
    "US.TXN", "US.AMAT", "US.ISRG", "US.INTU", "US.PYPL", "US.BKNG",
    "US.PANW", "US.SNPS", "US.CDNS", "US.MRVL", "US.MU", "US.LRCX",
    "US.KLAC", "US.ABNB", "US.DASH", "US.MELI", "US.WDAY", "US.DDOG",
    "US.CRWD", "US.ZS", "US.NET", "US.MDB", "US.TTD", "US.GILD",
    "US.REGN", "US.VRTX", "US.SBUX", "US.PDD", "US.BAC", "US.WFC",
    "US.GS", "US.MS", "US.SCHW", "US.CAT", "US.DE", "US.BA",
    "US.GE", "US.LMT", "US.RTX", "US.HON", "US.UPS", "US.FDX",
    "US.CVX", "US.COP", "US.SLB", "US.OXY", "US.DAL", "US.UAL",
    "US.CCL", "US.RCL", "US.WMT", "US.TGT", "US.HD", "US.LOW",
    "US.PFE", "US.JNJ", "US.MRK", "US.ABBV", "US.BMY", "US.OSCR",
    "US.ETSY", "US.CROX", "US.W", "US.RUM", "US.CHPT", "US.PTON",
    "US.MQ", "US.FUBO",
]


def load_watchlist() -> list[str]:
    if not WATCH_FILE.exists():
        WATCH_FILE.write_text(
            "# one code per line; recorded daily by scripts/record_chains.py\n"
            + "\n".join(DEFAULT_WATCH) + "\n"
        )
    return [ln.strip().upper() for ln in WATCH_FILE.read_text().splitlines()
            if ln.strip() and not ln.startswith("#")]


PACING_S = 3.5        # Moomoo hard-caps get_option_chain at 10 per 30s (a 3.0s
                      # floor); 3.5 leaves ~8.6/30s of margin. Measured 2026-07-16:
                      # at 2.0 the run hit 13.4 calls/30s, tripped the cap ~50
                      # symbols in, and could not recover -- rejections return in
                      # 0ms, so a failing loop paces FASTER than a working one.
                      # ponytail: fixed pacing, not a token bucket. Revisit only if
                      # the per-symbol call count changes.
RATE_BACKOFF_S = 31.0  # on a "high frequency" rejection, wait out the 30s window
RETRY_PASSES = 2      # transient timeouts usually clear on a later pass


def _record_one(service, code: str, out_dir: Path, today: str) -> None:
    """Snapshot one symbol's near-money chain to parquet. Raises on any miss."""
    # spot from a live snapshot (cheap; no full technical analysis)
    with service._lock:
        snap = service._client.get_snapshot([code])
    spot = float(snap.iloc[0]["last_price"]) if snap is not None and not snap.empty else None
    if not spot or spot <= 0:
        raise ValueError("no spot")
    picked = service._pick_option_expiry(code, TARGET_DTE)
    if picked is None:
        raise ValueError("no expiry")
    source, expiry, dte, _ = picked
    contracts = service._option_contracts_from(source, code, expiry, spot)
    if contracts is None or contracts.empty:
        raise ValueError("empty chain")
    contracts = contracts.copy()
    contracts["snap_date"] = today
    contracts["snap_ts"] = dt.datetime.now().isoformat(timespec="seconds")
    contracts["underlying"] = code
    contracts["spot"] = spot
    contracts["expiry"] = expiry
    contracts["dte"] = dte
    contracts["source"] = source
    # atomic: a kill mid-write must not leave a corrupt .parquet that the
    # resume-set glob counts as "done" (silently losing the symbol's day)
    tmp = out_dir / f"{code}.tmp"
    contracts.to_parquet(tmp, index=False)
    os.replace(tmp, out_dir / f"{code}.parquet")
    two_sided = int(((contracts["bid"] > 0) & (contracts["ask"] > 0)).sum())
    print(f"  {code}: {len(contracts)} contracts ({two_sided} two-sided) "
          f"exp {expiry} via {source}", flush=True)


def main() -> None:
    import time
    from app.services.analysis_service import service
    from scripts._session import session_date

    today = session_date()   # ET session date, so a post-midnight retry appends to tonight's dir
    out_dir = CHAINS_DIR / today
    out_dir.mkdir(parents=True, exist_ok=True)
    codes = load_watchlist()

    ok = 0
    for attempt in range(1, RETRY_PASSES + 2):
        done = {p.stem for p in out_dir.glob("*.parquet")}
        todo = [c for c in codes if c not in done]
        if not todo:
            break
        print(f"{today} pass {attempt}: {len(todo)} to record ({len(done)} done)", flush=True)
        for code in todo:
            try:
                _record_one(service, code, out_dir, today)
                ok += 1
            except Exception as e:  # noqa: BLE001 - one bad symbol never kills the run
                print(f"  {code}: SKIP ({e})", flush=True)
                # A rate rejection returns in ~0ms, so a tripped loop paces faster
                # than a healthy one and never recovers. Wait out the window.
                if "high frequency" in str(e).lower():
                    print(f"  rate-limited -- backing off {RATE_BACKOFF_S:.0f}s", flush=True)
                    time.sleep(RATE_BACKOFF_S)
            time.sleep(PACING_S)   # don't trip Moomoo's packet pacing
    missing = [c for c in codes if c not in {p.stem for p in out_dir.glob('*.parquet')}]
    print(f"done: {ok} recorded this run, {len(missing)} missing "
          f"({', '.join(missing) if missing else 'none'}) -> {out_dir}", flush=True)
    # honest exit code: >50% of the watchlist missing is a FAILED night, not a
    # green one (a 0-symbol night used to report success into the void)
    return 1 if len(missing) > len(codes) // 2 else 0


if __name__ == "__main__":
    from scripts._lock import single_instance
    rc = 0
    with single_instance("record_chains") as got:
        if got:
            rc = main() or 0
        else:
            print("another record_chains instance holds the lock -- exiting (work is being done)")
    # Moomoo SDK leaves non-daemon threads; exit hard AFTER the lock released.
    sys.stdout.flush()
    os._exit(rc)
