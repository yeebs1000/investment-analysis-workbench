"""Pre-market portfolio scan: what needs attention before the US open.

Read-only against broker truth. Flags, in priority order:
  NAKED    a SHORT leg is held but its protecting LONG leg is not -> undefined
           risk. Auto-flagged for immediate close at the next trade run.
  PARTIAL  some legs held, some not (a broken structure) -> auto-flagged close.
  PHANTOM  ledger says OPEN but the broker holds none of its legs.
  STOP     marked loss has breached the position's stop (25% of capital).
  DTE      short-leg structure within the DTE exit window.
Alerts (Telegram/ntfy) if anything actionable is found; otherwise a quiet
line. Placing the actual closes is the trade run's job, not this scan's --
this just makes them visible and pre-flags the broken ones.

Scheduled ~21:00 SGT (30 min before the 21:30 open). Run standalone anytime.
"""
import datetime as dt
import json
import re
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

PAPER = BACKEND / "data_store" / "paper"
OPT = re.compile(r"\d{6}[CP]\d+$")


def main() -> None:
    from scripts.paper_trade import (_load_structures, _load_flags, _save_flags,
                                     EXIT_DTE, EXIT_DTE_ALL_LONG, CONTRACT_SIZE,
                                     UNDEFINED_RISK_STOP, UNDEFINED_STOP_PCT)
    from app.brokers.paper_broker import PaperBroker

    b = PaperBroker().connect()
    pos = b.positions()
    qty = {str(r["code"]): float(r["qty"]) for _, r in pos.iterrows() if abs(float(r["qty"])) > 0}
    mark = {str(r["code"]): float(r["nominal_price"]) for _, r in pos.iterrows()
            if "nominal_price" in pos.columns and r.get("nominal_price")}
    b.close()

    structs = _load_structures()
    open_s = [s for s in structs if s.get("status") in ("OPEN", "CLOSING")
              and s.get("kind") != "HEDGE"]
    today = dt.date.today()
    flags = _load_flags()

    issues, rows, new_flags = [], [], False
    for s in open_s:
        held = [l for l in s["legs"] if abs(qty.get(l["code"], 0.0)) > 0]
        shorts_held = [l for l in held if l["side"] == "SELL"]
        longs_missing = [l for l in s["legs"] if l["side"] == "BUY" and abs(qty.get(l["code"], 0.0)) == 0]
        tag = ""
        if shorts_held and longs_missing:
            tag = "NAKED"          # short with no protecting long -> undefined risk
        elif held and len(held) < len(s["legs"]):
            tag = "PARTIAL"
        elif not held and s["status"] == "OPEN":
            tag = "PHANTOM"

        # dollar P&L from broker marks
        pnl = None
        if all(l["code"] in mark for l in s["legs"]):
            sh = 0.0
            for l in s["legs"]:
                sign = 1.0 if l["side"] == "BUY" else -1.0
                sh += sign * (mark[l["code"]] - (l.get("fill_price") or l["entry_mid"]))
            pnl = round(sh * CONTRACT_SIZE * s.get("contracts", 1), 0)

        cap = s.get("capital") or 0
        # hard STOP only where the engine actually stops (undefined-risk); a deep
        # loss on a DEFINED-risk spread is shown as info, not an action item
        if pnl is not None and cap and tag == "":
            if s["strategy"] in UNDEFINED_RISK_STOP and pnl <= -UNDEFINED_STOP_PCT * cap:
                tag = "STOP"
            elif pnl <= -0.6 * cap:
                tag = "deep-loss"
        try:
            dte = (dt.date.fromisoformat(s["expiry"]) - today).days
            has_short = any(l["side"] == "SELL" for l in s["legs"])
            if not tag and dte <= (EXIT_DTE if has_short else EXIT_DTE_ALL_LONG):
                tag = "DTE"
        except Exception:  # noqa: BLE001
            dte = "?"

        rows.append((s["underlying"], s["strategy"], dte, pnl, tag))
        if tag in ("NAKED", "PARTIAL"):
            issues.append(f"{tag} {s['underlying']} {s['strategy']}")
            if flags.get(s["id"]) != "close":     # pre-flag broken structures to close
                flags[s["id"]] = "close"; new_flags = True
        elif tag == "STOP":
            issues.append(f"STOP {s['underlying']} {s['strategy']} ({pnl:+,.0f})")

    if new_flags:
        _save_flags(flags)

    print(f"{'SYMBOL':9}{'STRUCTURE':22}{'DTE':>5}{'P&L':>9}   FLAG")
    print("-" * 56)
    for u, st, dte, pnl, tag in sorted(rows, key=lambda r: (r[4] == "", r[3] if r[3] is not None else 0)):
        ps = f"{pnl:+,.0f}" if pnl is not None else "--"
        print(f"{u:9}{st:22}{str(dte):>5}{ps:>9}   {tag}")
    print("-" * 56)
    print(f"open (non-hedge): {len(open_s)} | actionable: {len(issues)}")

    if issues:
        try:
            from scripts._alert import alert
            alert("Pre-market: positions need action",
                  f"{len(issues)} flagged: {'; '.join(issues[:8])}. "
                  f"Broken structures auto-flagged to close at the open.", priority="high")
        except Exception:  # noqa: BLE001
            pass
    print(json.dumps({"verdict": "SCAN", "issues": issues,
                      "open_non_hedge": len(open_s)}))


if __name__ == "__main__":
    import os
    main()
    sys.stdout.flush()
    os._exit(0)
