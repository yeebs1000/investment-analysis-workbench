"""Deep-history backfill for pool names missing >=260 cached daily bars.

Moomoo caps distinct-symbol kline history (free tier); IBKR has no such quota
but paces ~60 hist requests / 10 min and drops long-lived connections. This
script is RESUMABLE (skips names already >=260 bars) and reconnects every
CHUNK symbols, so re-running it repeatedly converges. Run:

    PYTHONPATH=. .venv/Scripts/python.exe scripts/backfill_bars.py [client_id]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

from app.data import normalize          # noqa: E402
from app.ml.data_store import BarStore   # noqa: E402
from app.brokers.ibkr_client import IBKRClient  # noqa: E402

PACING_S = 11.0     # under IBKR's 60/10min historical-data limit
CHUNK = 12          # reconnect every N symbols (connection goes stale)
MIN_BARS = 260


def main() -> None:
    base_cid = int(sys.argv[1]) if len(sys.argv) > 1 else 2211
    store = BarStore()
    pool = [l.strip() for l in (BACKEND / "data_store" / "chain_watch.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    todo = [c for c in pool if len(store.load(c)) < MIN_BARS]
    print(f"backfill: {len(todo)} names remaining", flush=True)

    ok = 0
    too_new = []
    ib = None
    for i, code in enumerate(todo):
        if ib is None or i % CHUNK == 0:
            if ib is not None:
                try: ib.close()
                except Exception: pass  # noqa: BLE001
            ib = IBKRClient(client_id=base_cid + (i // CHUNK)).connect()
        try:
            raw = ib.get_history_kline(code, ktype="day", lookback_days=1100, duration="3 Y")
            bars = normalize.bars_from_kline(raw) if raw is not None else None
            n = 0 if bars is None or bars.empty else len(bars)
            if n < 200:
                too_new.append(f"{code[3:]}({n})")
                print(f"  {code}: only {n} bars (new listing)", flush=True)
            else:
                store.save(code, bars)
                ok += 1
                print(f"  {code}: {n} bars saved  [{ok}]", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {code}: ERR {str(e)[:60]}", flush=True)
        time.sleep(PACING_S)
    if ib is not None:
        try: ib.close()
        except Exception: pass  # noqa: BLE001

    remaining = [c for c in pool if len(store.load(c)) < MIN_BARS]
    print(f"DONE this pass: +{ok} saved. {len(remaining)} still short "
          f"(genuinely new listings: {', '.join(too_new)})", flush=True)
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
