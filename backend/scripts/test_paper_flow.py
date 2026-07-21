"""Self-checks for the entry/lifecycle changes of 2026-07-18: write-ahead
records, strategy cap, EV floor, same-session retry, PENDING adoption, and the
post-close sweep. All offline -- a FakeBroker stands in for OpenD.

Run: PYTHONPATH=. .venv/Scripts/python.exe scripts/test_paper_flow.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402

import scripts.paper_trade as pt  # noqa: E402


class FakeBroker:
    def __init__(self, reject_codes=(), reject_once_codes=(), positions=None):
        self.reject = set(reject_codes)
        self.reject_once = set(reject_once_codes)
        self.placed, self.cancelled = [], []
        self._pos = positions if positions is not None else pd.DataFrame(columns=["code", "qty"])

    def place_limit(self, code, qty, side, price, note=""):
        self.placed.append((code, side, qty, price, note))
        if code in self.reject or code in self.reject_once:
            self.reject_once.discard(code)
            return {"status": "REJECTED", "order_id": None}
        return {"status": "SUBMITTED", "order_id": f"o{len(self.placed)}"}

    def cancel_order(self, oid):
        self.cancelled.append(oid)
        return True

    def positions(self):
        return self._pos

    def orders(self, status_filter_list=None):
        return pd.DataFrame(columns=["code", "order_id", "create_time"])


def _sig_rows(code, strat, n_legs, ev, max_loss=2.0, date="2026-07-20"):
    rows = []
    for i in range(n_legs):
        rows.append({
            "date": date, "code": code, "spot": 100.0, "expiry": "2026-08-21", "dte": 32,
            "decision": "BUY", "confidence": 0.5, "strategy": strat, "direction": "Bullish",
            "pop_pct": 40.0, "ev_per_share": ev, "net_debit_credit": -1.0, "max_loss": max_loss,
            "leg_action": "Buy" if i == 0 else "Sell", "leg_right": "Call",
            "leg_strike": 100.0 + 5 * i, "leg_code": f"US.{code[3:]}C{100 + 5 * i}",
            "leg_iv": 40.0, "leg_delta": 0.4, "bid": 1.0, "ask": 1.1,
            "mid_used": 1.05, "theo_bsm": 1.02, "oi": 500, "n_tradeable": 20, "n_chain": 60,
        })
    return rows


def _write_sig(tmp, rows):
    p = Path(tmp) / "2026-07-20.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def demo() -> None:
    today = "2026-07-20"
    tmp = tempfile.mkdtemp()

    # ---- strategy cap + EV floor + retry --------------------------------
    rows = []
    for i in range(6):                                   # 6 CDS candidates
        rows += _sig_rows(f"US.CD{i}", "Call Debit Spread", 2, ev=0.5 - i * 0.05)
    rows += _sig_rows("US.PX", "Put Debit Spread", 2, ev=0.30)
    rows += _sig_rows("US.PENNY", "Put Debit Spread", 2, ev=0.02)   # below floor
    rows += _sig_rows("US.RT", "Long Straddle", 2, ev=0.28)
    sig = _write_sig(tmp, rows)

    b = FakeBroker(reject_once_codes={"US.RTC100"})      # RT's first leg fails once
    structs, journal = [], []
    pt.STRUCT_LEDGER = Path(tmp) / "structures.jsonl"    # sandbox the ledger
    n = pt.do_entries(b, structs, journal, sig, budget=300000.0, today=today)

    entered = [s for s in structs if s["status"] == "OPEN"]
    by_strat = {}
    for s in entered:
        by_strat[s["strategy"]] = by_strat.get(s["strategy"], 0) + 1
    assert by_strat.get("Call Debit Spread", 0) <= pt.MAX_PER_STRATEGY_PER_DAY, by_strat
    assert all(s["underlying"] != "US.PENNY" for s in entered), "EV floor failed"
    assert any("friction floor" in j for j in journal), "EV floor not journaled"
    # retry: RT aborted once, then entered on the -r1 retry
    rt = [s for s in structs if s["underlying"] == "US.RT"]
    assert any(s["status"] == "ABORTED_ENTRY" for s in rt), "no abort recorded"
    assert any(s["status"] == "OPEN" and s["id"].endswith("-r1") for s in rt), "retry did not enter"
    assert n == len(entered)
    # every OPEN record went through PENDING first => legs recorded, ledger file exists
    assert pt.STRUCT_LEDGER.exists()

    # ---- stale-signal refusal ------------------------------------------
    stale = _write_sig(tmp, _sig_rows("US.ZZ", "Call Debit Spread", 2, 0.5, date="2026-07-17"))
    j2 = []
    n2 = pt.do_entries(FakeBroker(), [], j2, stale, budget=None, today=today)
    assert n2 == 0 and any("STALE" in j for j in j2), "stale signals were traded"

    # ---- PENDING adoption on restart -----------------------------------
    pend = {"id": "x", "underlying": "US.AA", "strategy": "Call Debit Spread",
            "status": "PENDING_ENTRY", "entry_date": "2026-07-19", "expiry": "2026-08-21",
            "legs": [{"code": "US.AAC100", "side": "BUY", "qty": 1}]}
    held = pd.DataFrame([{"code": "US.AAC100", "qty": 1.0}])
    lines = pt.sync_lifecycle(FakeBroker(positions=held), [dict(pend)], today)
    assert any("PENDING_ENTRY -> OPEN" in l for l in lines), lines
    lines = pt.sync_lifecycle(FakeBroker(), [dict(pend)], today)
    assert any("PENDING_ENTRY -> ABORTED_ENTRY" in l for l in lines), lines

    # ---- post-close sweep: lone filled leg flattened same session -------
    s = {"id": "y", "underlying": "US.BB", "strategy": "Call Debit Spread",
         "status": "OPEN", "entry_date": today, "expiry": "2026-08-21",
         "legs": [{"code": "US.BBC100", "side": "BUY", "qty": 1, "entry_mid": 1.0},
                  {"code": "US.BBC105", "side": "SELL", "qty": 1, "entry_mid": 0.5}]}
    held = pd.DataFrame([{"code": "US.BBC100", "qty": 1.0}])
    b3 = FakeBroker(positions=held)
    j3 = []
    pt.post_close_sweep(b3, [s], today, j3)
    assert s["status"] == "CLOSING", s["status"]
    assert any(c[0] == "US.BBC100" and c[1] == "SELL" for c in b3.placed), "no flatten order"
    # and an entirely-unfilled structure is closed out as CANCELLED_UNFILLED
    s2 = {**s, "id": "z", "status": "OPEN",
          "legs": [{"code": "US.CCC100", "side": "BUY", "qty": 1, "entry_mid": 1.0}]}
    pt.post_close_sweep(FakeBroker(), [s2], today, [])
    assert s2["status"] == "CANCELLED_UNFILLED"

    # ---- manual-flag exits: close_if_profit closes only when green ------
    pt._save_flags = lambda f: None                 # no disk writes in the test
    csp_leg = {"code": "US.AMAT260918P480000", "side": "SELL", "qty": 1, "entry_mid": 3.0}
    base_csp = {"id": "F1", "underlying": "US.AMAT", "strategy": "Cash-Secured Put",
                "status": "OPEN", "entry_date": today, "expiry": "2026-09-18",
                "max_profit": 3.0, "net_debit_credit": 3.0, "legs": [csp_leg]}

    def run_flag(mark, flag):
        pt._load_flags = lambda: {"F1": flag}
        pos = pd.DataFrame([{"code": csp_leg["code"], "qty": -1.0, "nominal_price": mark}])
        b = FakeBroker(positions=pos)
        s = dict(base_csp); s["legs"] = [dict(csp_leg)]
        pt.check_exits(b, [s], today)
        return s["status"], b.placed

    # short at 3.0, now 2.0 -> +1.0/sh profit -> close_if_profit CLOSES it
    st, placed = run_flag(2.0, "close_if_profit")
    assert st == "CLOSING" and placed, f"profit flag should close: {st}"
    # short at 3.0, now 4.0 -> -1.0/sh loss -> close_if_profit HOLDS
    st, placed = run_flag(4.0, "close_if_profit")
    assert st == "OPEN" and not placed, f"loss flag should hold: {st}"
    # unconditional close fires regardless of P&L
    st, _ = run_flag(4.0, "close")
    assert st == "CLOSING", f"unconditional close should fire: {st}"

    print("paper_flow: strategy cap, EV floor, retry, write-ahead, adoption, sweep, flags -- all pass")


if __name__ == "__main__":
    import os
    demo()
    sys.stdout.flush()
    os._exit(0)     # sync_lifecycle imports the moomoo SDK, whose non-daemon threads hang exit
