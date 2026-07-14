"""Daily PAPER trading engine (Moomoo SIMULATE account only -- see the
guardrails in app/brokers/paper_broker.py).

Flow (scheduled weekdays 23:45, after record_chains 23:00 + log_signals 23:30):
 1. EXITS   -- mark open structures from paper positions; close any that hit
               the SAME management rules the backtest validated
               (options_backtest.MANAGEMENT), or DTE <= 7.
 2. HYGIENE -- cancel yesterday's unfilled entry orders (each one is a
               failed-fill datapoint, logged, not silently retried).
 3. ENTRIES -- read today's signal log, apply liquidity criteria, place
               1-contract structures as per-leg limit orders at the logged mid.
 4. JOURNAL -- write an Obsidian-ready markdown daily note + dashboard and
               a machine-readable ledger (data_store/paper/structures.jsonl).

Sizing is deliberately 1 contract per structure: the object of paper trading
is FILL AND TRACKING DATA, not P&L optimization. ponytail: portfolio caps are
constants below; revisit only after the first month of fills.
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))

import pandas as pd

# --- criteria / caps ---------------------------------------------------------
MAX_LEG_SPREAD_PCT = 10.0     # per leg, bid/ask over mid
MIN_LEG_OI = 25       # lockstep with log_signals prefilter
MAX_NEW_PER_DAY = 8
MAX_OPEN_STRUCTURES = 30

# --- budget sizing (pass --budget N; without it every structure is 1 lot) ----
RISK_FRAC = 0.01              # max loss per structure as fraction of budget
MAX_CONTRACTS = 10            # per structure, regardless of budget
CAP_FRAC = 0.15               # max capital tied to ONE structure (diversification)

# --- trailing high-water exit (user directive: close at the highs, don't ride
# a winner back down). Arms once P&L reaches TRAIL_ARM x base (base = max
# profit for defined-risk, entry debit for straddles); closes when P&L gives
# back TRAIL_GIVEBACK of the PEAK. Peak is tracked per structure in the ledger
# at each beat -- daily granularity, so intraday spikes between beats are
# invisible (known limit of a once-a-day loop).
TRAIL_ARM = 0.50
TRAIL_GIVEBACK = 0.35
EXIT_DTE = 7                  # structures with SHORT legs: close at <= this DTE (pin/assignment risk)
EXIT_DTE_ALL_LONG = 2         # all-long structures (straddles) have no assignment risk;
                              # the exit grid showed time-stops cost them ~4pts -- ride to near expiry
CONTRACT_SIZE = 100

SIGNALS_DIR = BACKEND / "data_store" / "signals"
PAPER_DIR = BACKEND / "data_store" / "paper"
JOURNAL_DIR = BACKEND / "data_store" / "journal"
STRUCT_LEDGER = PAPER_DIR / "structures.jsonl"


def _load_structures() -> list[dict]:
    if not STRUCT_LEDGER.exists():
        return []
    return [json.loads(l) for l in STRUCT_LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def _save_structures(structs: list[dict]) -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    STRUCT_LEDGER.write_text("\n".join(json.dumps(s) for s in structs) + "\n", encoding="utf-8")


def _mid(row) -> float | None:
    b, a = row.get("bid"), row.get("ask")
    if b and a and b > 0 and a >= b:
        return (b + a) / 2.0
    return None


def capital_required(strategy: str, legs: list[dict], net: float | None,
                     max_loss: float | None, n: int) -> float:
    """Paper capital a structure ties up: CSP posts the strike; debits pay the
    debit; defined-risk credit spreads post the max loss as margin."""
    if strategy == "Cash-Secured Put" and legs:
        return legs[0]["strike"] * CONTRACT_SIZE * n
    if net is not None and net < 0:
        return abs(net) * CONTRACT_SIZE * n
    return (max_loss or 0.0) * CONTRACT_SIZE * n


def size_structure(budget: float | None, strategy: str, legs: list[dict],
                   net: float | None, max_loss: float | None,
                   capital_in_use: float) -> tuple[int, float, str]:
    """(contracts, capital, reason-if-zero). Without a budget: 1 lot."""
    if budget is None:
        return 1, capital_required(strategy, legs, net, max_loss, 1), ""
    if not max_loss or max_loss <= 0:
        return 0, 0.0, "no defined max loss"
    n = int((budget * RISK_FRAC) // (max_loss * CONTRACT_SIZE))
    # a single lot is allowed slightly above the risk sliver (expensive
    # straddles are the P&L engine; 1 lot of a $47 straddle beats 0 lots)
    if n == 0 and max_loss * CONTRACT_SIZE <= budget * RISK_FRAC * 2.5:
        n = 1
    n = min(n, MAX_CONTRACTS)
    while n > 0:
        cap = capital_required(strategy, legs, net, max_loss, n)
        if cap <= budget * CAP_FRAC and capital_in_use + cap <= budget:
            return n, cap, ""
        n -= 1
    return 0, 0.0, "risk/capital budget exceeded"


def structure_groups(sig: pd.DataFrame):
    """Yield (underlying, strategy, legs-DataFrame) for structures whose every
    leg passes the liquidity criteria."""
    for (code, strat), g in sig.groupby(["code", "strategy"]):
        ok = True
        for _, leg in g.iterrows():
            m = _mid(leg)
            if m is None or m <= 0:
                ok = False; break
            if (leg["ask"] - leg["bid"]) / m * 100.0 > MAX_LEG_SPREAD_PCT:
                ok = False; break
            if pd.isna(leg["oi"]) or leg["oi"] < MIN_LEG_OI:
                ok = False; break
        if ok:
            yield code, strat, g


def check_exits(broker, structs: list[dict], today: str) -> list[str]:
    """Apply backtest management rules to open structures using paper position
    marks. Returns journal lines."""
    from app.analytics.options_backtest import MANAGEMENT, DEFAULT_RULE
    lines = []
    try:
        pos = broker.positions()
    except Exception as e:  # noqa: BLE001
        return [f"- exits skipped: positions query failed ({e})"]
    by_code = {}
    if not pos.empty and "code" in pos.columns:
        for _, p in pos.iterrows():
            by_code[str(p["code"])] = p

    for s in structs:
        if s["status"] != "OPEN":
            continue
        dte_left = (dt.date.fromisoformat(s["expiry"]) - dt.date.fromisoformat(today)).days
        exit_dte = EXIT_DTE_ALL_LONG if all(l["side"] == "BUY" for l in s["legs"]) else EXIT_DTE
        # mark: sum of leg (market - entry) * sign
        pnl = 0.0
        marks_ok = True
        for leg in s["legs"]:
            p = by_code.get(leg["code"])
            mark = float(p["nominal_price"]) if p is not None and pd.notna(p.get("nominal_price")) else None
            if mark is None:
                marks_ok = False; break
            sign = 1.0 if leg["side"] == "BUY" else -1.0
            pnl += sign * (mark - leg["entry_mid"])
        rule = MANAGEMENT.get(s["strategy"], DEFAULT_RULE)
        base = s.get("max_profit")
        # straddles have no defined max profit -- trail off the entry debit
        trail_base = base if (base and base > 0) else abs(s.get("net_debit_credit") or 0) or None
        if marks_ok and trail_base:
            if pnl > (s.get("peak_pnl") or -1e9):
                s["peak_pnl"] = round(pnl, 3)
        peak = s.get("peak_pnl")
        reason = None
        if dte_left <= exit_dte:
            reason = f"DTE {dte_left} <= {exit_dte}"
        elif marks_ok and rule.profit_target is not None and base and base > 0 \
                and pnl >= rule.profit_target * base:
            reason = f"profit target ({pnl:+.2f} >= {rule.profit_target:.0%} of {base:.2f})"
        elif marks_ok and trail_base and peak is not None \
                and peak >= TRAIL_ARM * trail_base \
                and pnl <= peak * (1 - TRAIL_GIVEBACK):
            reason = (f"trailing high-water stop (peak {peak:+.2f}, now {pnl:+.2f}, "
                      f"gave back >{TRAIL_GIVEBACK:.0%})")
        elif marks_ok and rule.stop_to_be is not None and base and base > 0:
            if pnl >= rule.stop_to_be * base:
                s["be_armed"] = True
            elif s.get("be_armed") and pnl <= 0:
                reason = "breakeven stop (winner faded)"
        if reason is None:
            continue
        # close: opposite side per leg, limit at current mark
        for leg in s["legs"]:
            p = by_code.get(leg["code"])
            mark = float(p["nominal_price"]) if p is not None and pd.notna(p.get("nominal_price")) else leg["entry_mid"]
            side = "SELL" if leg["side"] == "BUY" else "BUY"
            broker.place_limit(leg["code"], leg["qty"], side, mark,
                               note=f"EXIT {s['id']}: {reason}")
        s["status"] = "CLOSING"
        s["exit_reason"] = reason
        s["exit_date"] = today
        lines.append(f"- EXIT {s['underlying']} {s['strategy']}: {reason} (marked P&L {pnl:+.2f}/sh)")
    return lines


def reconcile(broker, structs: list[dict]) -> list[str]:
    """Post-session counter-check against broker truth (user directive after a
    day-1 mis-attribution: order LIMIT price != executed price on SIMULATE).
    Updates each open leg's fill price from the broker's position cost_price,
    recomputes structure net entry, and flags qty mismatches."""
    lines = []
    try:
        pos = broker.positions()
    except Exception as e:  # noqa: BLE001
        return [f"- reconcile skipped: {e}"]
    cost = {str(p["code"]): float(p["cost_price"]) for _, p in pos.iterrows()}
    bqty = {str(p["code"]): float(p["qty"]) for _, p in pos.iterrows() if abs(p["qty"]) > 0}
    lqty: dict[str, float] = {}
    for s in structs:
        if s["status"] not in ("OPEN", "CLOSING"):
            continue
        net = 0.0
        for leg in s["legs"]:
            sign = 1 if leg["side"] == "BUY" else -1
            lqty[leg["code"]] = lqty.get(leg["code"], 0) + sign * leg["qty"]
            if leg["code"] in cost and cost[leg["code"]] > 0:
                actual = cost[leg["code"]]
                if abs(actual - (leg.get("fill_price") or leg["entry_mid"])) >= 0.01:
                    lines.append(f"- fill corrected {leg['code']}: recorded "
                                 f"{leg.get('fill_price') or leg['entry_mid']} -> broker {actual}")
                leg["fill_price"] = actual
            net += sign * (leg.get("fill_price") or leg["entry_mid"])
        s["net_entry_actual"] = round(net, 2)
    for c in set(lqty) | set(bqty):
        if abs(lqty.get(c, 0) - bqty.get(c, 0)) >= 0.5:
            lines.append(f"- QTY MISMATCH {c}: ledger {lqty.get(c, 0):+g} vs broker {bqty.get(c, 0):+g}")
    return lines or ["- reconciled clean: fills and quantities match broker"]


def main() -> None:
    from app.brokers.paper_broker import PaperBroker

    budget = None
    if "--budget" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--budget") + 1])
    today = dt.date.today().isoformat()
    sig_path = SIGNALS_DIR / f"{today}.csv"
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    structs = _load_structures()
    journal: list[str] = []

    broker = PaperBroker().connect()
    acc = broker.account()
    equity = float(acc.iloc[0]["total_assets"]) if "total_assets" in acc.columns else None

    # 1. exits
    journal.append("## Exits")
    journal += check_exits(broker, structs, today) or ["- none"]

    # 2. cancel stale unfilled entry orders (failed-fill datapoints)
    journal.append("\n## Order hygiene")
    try:
        from moomoo import OrderStatus
        open_orders = broker.orders(status_filter_list=[
            OrderStatus.SUBMITTED, OrderStatus.WAITING_SUBMIT, OrderStatus.SUBMITTING])
        stale = open_orders[open_orders.get("create_time", "").astype(str).str[:10] < today] \
            if not open_orders.empty else open_orders
        n_cancel = 0
        if not stale.empty:
            for _, o in stale.iterrows():
                if broker.cancel_order(str(o["order_id"])):
                    n_cancel += 1
        journal.append(f"- cancelled {n_cancel} stale unfilled order(s) -- logged as failed fills")
    except Exception as e:  # noqa: BLE001
        journal.append(f"- hygiene skipped ({e})")

    # 3. entries
    journal.append("\n## Entries")
    n_open = sum(1 for s in structs if s["status"] in ("OPEN", "CLOSING"))
    open_unders = {s["underlying"] for s in structs if s["status"] in ("OPEN", "CLOSING")}
    placed = 0
    if not sig_path.exists():
        journal.append(f"- no signal file for {today} (market holiday or recorder gap)")
    else:
        sig = pd.read_csv(sig_path)
        capital_in_use = sum(s.get("capital", 0.0) for s in structs
                             if s["status"] in ("OPEN", "CLOSING"))
        for code, strat, legs in structure_groups(sig):
            if placed >= MAX_NEW_PER_DAY or n_open + placed >= MAX_OPEN_STRUCTURES:
                journal.append("- caps reached; remaining signals skipped")
                break
            if code in open_unders:
                continue                    # one structure per underlying
            sid = f"{today}-{code}-{strat}".replace(" ", "_")
            row0 = legs.iloc[0]
            net = float(row0["net_debit_credit"]) if pd.notna(row0["net_debit_credit"]) else None
            max_loss = float(row0["max_loss"]) if pd.notna(row0["max_loss"]) else None
            leg_meta = [{"strike": float(l["leg_strike"])} for _, l in legs.iterrows()]
            n_contracts, capital, why = size_structure(budget, strat, leg_meta, net,
                                                       max_loss, capital_in_use)
            if n_contracts < 1:
                journal.append(f"- skip {code} {strat}: {why}")
                continue
            leg_recs = []
            for _, leg in legs.iterrows():
                m = _mid(leg)
                res = broker.place_limit(leg["leg_code"], n_contracts,
                                         leg["leg_action"].upper(), m,
                                         note=f"ENTRY {sid}")
                leg_recs.append({
                    "code": leg["leg_code"], "side": leg["leg_action"].upper(), "qty": n_contracts,
                    "right": leg["leg_right"], "strike": float(leg["leg_strike"]),
                    "entry_mid": round(m, 2), "theo": round(float(leg["theo_bsm"]), 2) if pd.notna(leg["theo_bsm"]) else None,
                    "bid": float(leg["bid"]), "ask": float(leg["ask"]),
                    "order_status": res.get("status"), "order_id": res.get("order_id"),
                })
            capital_in_use += capital
            # max profit: credit structures = credit; 2-leg verticals = width - debit
            max_profit = None
            if net is not None and net > 0:
                max_profit = net
            elif net is not None and len(leg_recs) == 2:
                width = abs(leg_recs[0]["strike"] - leg_recs[1]["strike"])
                max_profit = width - abs(net)
            structs.append({
                "id": sid, "underlying": code, "strategy": strat, "status": "OPEN",
                "entry_date": today, "expiry": str(row0["expiry"]), "dte": int(row0["dte"]),
                "pop_pct": float(row0["pop_pct"]) if pd.notna(row0["pop_pct"]) else None,
                "ev_per_share": float(row0["ev_per_share"]) if pd.notna(row0["ev_per_share"]) else None,
                "net_debit_credit": net, "max_loss": max_loss, "max_profit": max_profit,
                "contracts": n_contracts, "capital": round(capital, 2),
                "legs": leg_recs,
            })
            open_unders.add(code)
            placed += 1
            journal.append(f"- ENTER {code} {strat}: {n_contracts}x {len(leg_recs)} leg(s) at mid, "
                           f"capital ~${capital:,.0f} (POP {row0['pop_pct']}%, EV {row0['ev_per_share']})")
        if placed == 0 and sig_path.exists():
            journal.append("- no structures passed criteria today")

    # post-session counter-check: broker truth beats our order records
    journal.append("\n## Reconciliation (broker truth)")
    journal += reconcile(broker, structs)

    _save_structures(structs)

    # 4. journal note (Obsidian-ready) + dashboard
    n_open_now = sum(1 for s in structs if s["status"] in ("OPEN", "CLOSING"))
    note = JOURNAL_DIR / f"{today}.md"
    front = (f"---\ndate: {today}\ntype: paper-journal\nentries_placed: {placed}\n"
             f"open_structures: {n_open_now}\npaper_equity: {equity}\n---\n")
    body = (f"# Paper journal — {today}\n\n"
            f"Paper account equity: **{equity:,.0f}** · open structures: **{n_open_now}**\n\n"
            + "\n".join(journal) + "\n\n"
            f"[[Dashboard]]\n")
    note.write_text(front + body, encoding="utf-8")

    closed = [s for s in structs if s["status"] == "CLOSING"]
    dash = JOURNAL_DIR / "Dashboard.md"
    dash.write_text(
        "---\ntype: paper-dashboard\n---\n# Paper Trading Dashboard\n\n"
        f"Updated: {today}\n\n"
        f"- Paper equity: **{equity:,.0f}**\n"
        f"- Open structures: **{n_open_now}**\n"
        f"- Structures ever entered: **{len(structs)}**\n"
        f"- Closed/closing: **{len(closed)}**\n\n"
        "## Recent daily notes\n"
        + "\n".join(f"- [[{p.stem}]]" for p in sorted(JOURNAL_DIR.glob('2*.md'), reverse=True)[:15])
        + "\n\nLedger: `data_store/paper/structures.jsonl` · orders: `data_store/paper/order_ledger.jsonl`\n",
        encoding="utf-8")

    # equity snapshot + dashboard refresh (best-effort; never blocks the run)
    try:
        from scripts.build_dashboard import build, snapshot_equity
        cash = float(acc.iloc[0]["cash"]) if "cash" in acc.columns else None
        if equity is not None:
            snapshot_equity(equity, cash)
        build()
    except Exception as e:  # noqa: BLE001
        print(f"dashboard refresh failed: {e}")

    print(f"{today}: {placed} entered, {len(closed)} closing, {n_open_now} open -> {note}")
    broker.close()
    sys.stdout.flush()
    import os
    os._exit(0)


if __name__ == "__main__":
    main()
