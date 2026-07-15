"""Daily PAPER trading engine (Moomoo SIMULATE account only -- see the
guardrails in app/brokers/paper_broker.py).

Flow (scheduled weekdays 23:40 local, after record_chains 22:30 + log_signals 23:20):
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
    """Atomic write (tmp + replace): a crash or OneDrive sync lock mid-write
    must never truncate the ledger."""
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STRUCT_LEDGER.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(json.dumps(s) for s in structs) + "\n", encoding="utf-8")
    import os
    os.replace(tmp, STRUCT_LEDGER)


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


def _broker_qty(broker):
    """{code: signed qty} of actual broker holdings (zero rows dropped)."""
    pos = broker.positions()
    return {str(p["code"]): float(p["qty"]) for _, p in pos.iterrows()
            if abs(float(p["qty"])) > 0}, pos


def _close_legs(broker, s: dict, legs: list[dict], reason: str, bqty: dict,
                snap=None) -> None:
    """Place CROSSING exit orders (sell at bid / buy at ask) for the legs we
    actually hold -- certainty of fill over price finesse, and every fill is a
    real cost datapoint. Falls back to the ledger mid if no live quote."""
    for leg in legs:
        held = abs(bqty.get(leg["code"], 0.0))
        qty = min(leg["qty"], held)
        if qty < 1:
            continue
        side = "SELL" if leg["side"] == "BUY" else "BUY"
        px = None
        if snap is not None and leg["code"] in snap.index:
            row = snap.loc[leg["code"]]
            px = float(row["bid_price"]) if side == "SELL" else float(row["ask_price"])
            if not px or px <= 0:
                px = None
        if px is None:
            px = leg.get("fill_price") or leg["entry_mid"]
        broker.place_limit(leg["code"], qty, side, round(px, 2),
                           note=f"EXIT {s['id']}: {reason}")


def _live_snapshot(codes: list[str]):
    """Best-effort live quotes for exit pricing; None if unavailable."""
    try:
        from app.services.analysis_service import service
        with service._lock:
            snap = service._client.get_snapshot(codes)
        return snap.set_index("code")
    except Exception:  # noqa: BLE001
        return None


def sync_lifecycle(broker, structs: list[dict], today: str) -> list[str]:
    """State-machine sync against broker truth, run FIRST each session.
    OPEN (entered before today) with no legs held  -> CANCELLED_UNFILLED
    OPEN with only some legs held                  -> BROKEN -> close held legs
    CLOSING with no legs held                      -> CLOSED (capital released)
    CLOSING with legs still held                   -> re-place crossing exits
    Without this, CLOSING was a dead-end and phantom OPEN structures would
    eventually fire DTE exits for positions that never existed."""
    lines = []
    try:
        bqty, _ = _broker_qty(broker)
    except Exception as e:  # noqa: BLE001
        return [f"- lifecycle sync skipped: {e}"]
    retry_codes = [l["code"] for s in structs if s["status"] == "CLOSING" for l in s["legs"]]
    snap = _live_snapshot(retry_codes) if retry_codes else None
    for s in structs:
        if s["status"] == "OPEN" and s["entry_date"] < today:
            held = [l for l in s["legs"] if abs(bqty.get(l["code"], 0.0)) > 0]
            if not held:
                s["status"] = "CANCELLED_UNFILLED"
                s["exit_reason"] = "entry orders never filled"
                s["exit_date"] = today
                lines.append(f"- {s['underlying']} {s['strategy']}: entry never filled -> CANCELLED_UNFILLED")
            elif len(held) < len(s["legs"]):
                s["status"] = "CLOSING"
                s["exit_reason"] = "broken partial entry -- flattening held legs"
                s["exit_date"] = today
                leg_snap = _live_snapshot([l["code"] for l in held])
                _close_legs(broker, s, held, s["exit_reason"], bqty, leg_snap)
                lines.append(f"- {s['underlying']} {s['strategy']}: PARTIAL entry -> flattening {len(held)} held leg(s)")
        elif s["status"] == "CLOSING":
            held = [l for l in s["legs"] if abs(bqty.get(l["code"], 0.0)) > 0]
            if not held:
                s["status"] = "CLOSED"
                s["closed_date"] = today
                lines.append(f"- {s['underlying']} {s['strategy']}: exits filled -> CLOSED")
            else:
                _close_legs(broker, s, held, f"retry: {s.get('exit_reason', 'exit')}", bqty, snap)
                lines.append(f"- {s['underlying']} {s['strategy']}: retrying exit on {len(held)} leg(s) at crossing prices")
    return lines


def check_exits(broker, structs: list[dict], today: str) -> list[str]:
    """Apply backtest management rules to open structures using paper position
    marks (fill_price basis where reconciled). Returns journal lines."""
    from app.analytics.options_backtest import MANAGEMENT, DEFAULT_RULE
    lines = []
    try:
        bqty, pos = _broker_qty(broker)
    except Exception as e:  # noqa: BLE001
        return [f"- exits skipped: positions query failed ({e})"]
    by_code = {}
    if not pos.empty and "code" in pos.columns:
        for _, p in pos.iterrows():
            by_code[str(p["code"])] = p

    for s in structs:
        if s["status"] != "OPEN":
            continue
        # only manage structures whose every leg is actually held (fresh
        # entries may still be filling; phantom ones were handled by lifecycle)
        if not all(abs(bqty.get(l["code"], 0.0)) > 0 for l in s["legs"]):
            continue
        dte_left = (dt.date.fromisoformat(s["expiry"]) - dt.date.fromisoformat(today)).days
        exit_dte = EXIT_DTE_ALL_LONG if all(l["side"] == "BUY" for l in s["legs"]) else EXIT_DTE
        # mark-to-market vs BROKER-TRUE entry basis (fill_price from reconcile)
        pnl = 0.0
        marks_ok = True
        for leg in s["legs"]:
            p = by_code.get(leg["code"])
            mark = float(p["nominal_price"]) if p is not None and pd.notna(p.get("nominal_price")) else None
            if mark is None or mark <= 0:
                marks_ok = False; break
            sign = 1.0 if leg["side"] == "BUY" else -1.0
            basis = leg.get("fill_price") or leg["entry_mid"]
            pnl += sign * (mark - basis)
        rule = MANAGEMENT.get(s["strategy"], DEFAULT_RULE)
        base = s.get("max_profit")
        # straddles have no defined max profit -- trail off the entry debit
        trail_base = base if (base and base > 0) else abs(s.get("net_debit_credit") or 0) or None
        if marks_ok and trail_base:
            prev = s.get("peak_pnl")
            if prev is None or pnl > prev:
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
        snap = _live_snapshot([l["code"] for l in s["legs"]])
        _close_legs(broker, s, s["legs"], reason, bqty, snap)
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
    today = dt.date.today().isoformat()
    cost = {str(p["code"]): float(p["cost_price"]) for _, p in pos.iterrows()}
    bqty = {str(p["code"]): float(p["qty"]) for _, p in pos.iterrows() if abs(p["qty"]) > 0}
    lqty: dict[str, float] = {}
    for s in structs:
        if s["status"] not in ("OPEN", "CLOSING"):
            continue
        net = 0.0
        for leg in s["legs"]:
            sign = 1 if leg["side"] == "BUY" else -1
            # qty check only for SETTLED structures: today's entries are still
            # filling and CLOSING has exits in flight -- flagging them is noise
            if s["status"] == "OPEN" and s["entry_date"] < today:
                lqty[leg["code"]] = lqty.get(leg["code"], 0) + sign * leg["qty"]
            if leg["code"] in cost and cost[leg["code"]] > 0:
                actual = cost[leg["code"]]
                if abs(actual - (leg.get("fill_price") or leg["entry_mid"])) >= 0.01:
                    lines.append(f"- fill corrected {leg['code']}: recorded "
                                 f"{leg.get('fill_price') or leg['entry_mid']} -> broker {actual}")
                leg["fill_price"] = actual
            net += sign * (leg.get("fill_price") or leg["entry_mid"])
        # store in the strategist's sign convention (credit positive)
        s["net_entry_actual"] = round(-net, 2)
    for c in lqty:
        if abs(lqty.get(c, 0) - bqty.get(c, 0)) >= 0.5:
            lines.append(f"- QTY MISMATCH {c}: ledger {lqty.get(c, 0):+g} vs broker {bqty.get(c, 0):+g}")
    return lines or ["- reconciled clean: fills and quantities match broker"]


# expected leg counts per strategy: a partial signals row-set must never
# become a live incomplete structure (e.g. a 3-legged "condor")
EXPECTED_LEGS = {
    "Iron Condor": 4, "Call Debit Spread": 2, "Put Debit Spread": 2,
    "Bull Put Spread (credit)": 2, "Bear Call Spread (credit)": 2,
    "Long Straddle": 2, "Collar": 2, "Cash-Secured Put": 1, "Covered Call": 1,
}


def chase_entry_fills(broker, structs, today, walk_frac=0.5) -> list[str]:
    """MODIFY today's still-resting entry legs toward the market (walk_frac of
    the half-spread) instead of cancel+replace -- preserves queue priority and
    avoids the naked window. Runs each session; over successive beats a leg
    converges to a fill. Live finding: resting at signal-time mid gets
    adversely selected within ~30 min."""
    lines = []
    try:
        from moomoo import OrderStatus
        working = broker.orders(status_filter_list=[OrderStatus.SUBMITTED])
    except Exception as e:  # noqa: BLE001
        return [f"- entry-fill chase skipped ({e})"]
    if working.empty:
        return ["- no resting entry orders"]
    # map working orders to today's OPEN structures' legs
    today_legs = {l["code"]: (s, l) for s in structs if s["status"] == "OPEN"
                  and s["entry_date"] == today for l in s["legs"]}
    codes = [str(o["code"]) for _, o in working.iterrows() if str(o["code"]) in today_legs]
    if not codes:
        return ["- no resting entry legs to chase"]
    snap = _live_snapshot(codes)
    n = 0
    for _, o in working.iterrows():
        c = str(o["code"])
        if c not in today_legs:
            continue
        _, leg = today_legs[c]
        oid, qty, old = str(o["order_id"]), float(o["qty"]), float(o["price"])
        if snap is None or c not in snap.index:
            continue
        bid = float(snap.loc[c, "bid_price"]); ask = float(snap.loc[c, "ask_price"])
        if bid <= 0 or ask <= 0:
            continue
        mid = (bid + ask) / 2
        newpx = (round(mid - walk_frac * (mid - bid), 2) if leg["side"] == "SELL"
                 else round(mid + walk_frac * (ask - mid), 2))
        if abs(newpx - old) < 0.01:
            continue
        res = broker.modify_price(oid, newpx, qty=qty, note="chase entry fill")
        if res.get("status") == "MODIFIED":
            n += 1
            lines.append(f"- modify {c}: {old} -> {newpx} (mkt {bid}/{ask})")
    return lines or ["- resting entries already at market"]


def do_entries(broker, structs, journal, sig_path, budget, today, max_new=None) -> int:
    # --max-new (explicitly passed) marks a DISTINCT session: reset the
    # allowance to 0. Needed because the ~12h local/ET offset makes one local
    # date span two US sessions, and the default per-day ledger count would
    # otherwise carry the prior session's entries into this one.
    session_reset = max_new is not None
    cap = max_new if max_new is not None else MAX_NEW_PER_DAY
    n_open = sum(1 for s in structs if s["status"] in ("OPEN", "CLOSING"))
    open_unders = {s["underlying"] for s in structs if s["status"] in ("OPEN", "CLOSING")}
    existing_ids = {s["id"] for s in structs}
    placed = 0 if session_reset else sum(
        1 for s in structs if s.get("entry_date") == today
        and s["status"] not in ("CANCELLED_UNFILLED", "ABORTED_ENTRY"))
    if not sig_path.exists():
        journal.append(f"- no signal file for {today} (market holiday or recorder gap)")
        return 0
    try:
        sig = pd.read_csv(sig_path)
        if sig.empty or "code" not in sig.columns:
            raise ValueError("empty")
    except Exception:  # noqa: BLE001 - empty/corrupt CSV must not kill the run
        journal.append("- signals file empty or unreadable -- no entries today")
        return 0
    capital_in_use = sum(s.get("capital", 0.0) for s in structs
                         if s["status"] in ("OPEN", "CLOSING"))
    # rank candidates by model EV per unit risk, best first (was alphabetical)
    groups = sorted(structure_groups(sig),
                    key=lambda t: -((t[2]["ev_per_share"].iloc[0] or 0)
                                    / max(t[2]["max_loss"].iloc[0] or 1e9, 0.01)))
    entered = 0
    for code, strat, legs in groups:
        if placed >= cap or n_open + entered >= MAX_OPEN_STRUCTURES:
            journal.append("- caps reached; remaining signals skipped")
            break
        if code in open_unders:
            continue                    # one structure per underlying
        if EXPECTED_LEGS.get(strat) is not None and len(legs) != EXPECTED_LEGS[strat]:
            journal.append(f"- skip {code} {strat}: {len(legs)} legs in signals, "
                           f"expected {EXPECTED_LEGS[strat]} (partial row-set)")
            continue
        sid = f"{today}-{code}-{strat}".replace(" ", "_")
        if sid in existing_ids:
            continue                    # same-day re-entry after a close: skip
        row0 = legs.iloc[0]
        net = float(row0["net_debit_credit"]) if pd.notna(row0["net_debit_credit"]) else None
        max_loss = float(row0["max_loss"]) if pd.notna(row0["max_loss"]) else None
        leg_meta = [{"strike": float(l["leg_strike"])} for _, l in legs.iterrows()]
        n_contracts, capital, why = size_structure(budget, strat, leg_meta, net,
                                                   max_loss, capital_in_use)
        if n_contracts < 1:
            journal.append(f"- skip {code} {strat}: {why}")
            continue
        leg_recs, rejected = [], False
        for _, leg in legs.iterrows():
            m = _mid(leg)
            res = broker.place_limit(leg["leg_code"], n_contracts,
                                     leg["leg_action"].upper(), m,
                                     note=f"ENTRY {sid}")
            if res.get("status") == "REJECTED":
                rejected = True
            leg_recs.append({
                "code": leg["leg_code"], "side": leg["leg_action"].upper(), "qty": n_contracts,
                "right": leg["leg_right"], "strike": float(leg["leg_strike"]),
                "entry_mid": round(m, 2), "theo": round(float(leg["theo_bsm"]), 2) if pd.notna(leg["theo_bsm"]) else None,
                "bid": float(leg["bid"]), "ask": float(leg["ask"]),
                "order_status": res.get("status"), "order_id": res.get("order_id"),
            })
        # max profit: credit = the credit; VERTICALS (one buy + one sell) =
        # width - debit. All-long 2-leg structures (straddle) have none.
        max_profit = None
        if net is not None and net > 0:
            max_profit = net
        elif net is not None and len(leg_recs) == 2 \
                and {l["side"] for l in leg_recs} == {"BUY", "SELL"}:
            width = abs(leg_recs[0]["strike"] - leg_recs[1]["strike"])
            max_profit = width - abs(net) if width - abs(net) > 0 else None
        rec = {
            "id": sid, "underlying": code, "strategy": strat, "status": "OPEN",
            "entry_date": today, "expiry": str(row0["expiry"]), "dte": int(row0["dte"]),
            "pop_pct": float(row0["pop_pct"]) if pd.notna(row0["pop_pct"]) else None,
            "ev_per_share": float(row0["ev_per_share"]) if pd.notna(row0["ev_per_share"]) else None,
            "net_debit_credit": net, "max_loss": max_loss, "max_profit": max_profit,
            "contracts": n_contracts, "capital": round(capital, 2),
            "legs": leg_recs,
        }
        if rejected:
            # abort: cancel the sibling orders that did submit; hold no capital
            for lr in leg_recs:
                if lr.get("order_id") and lr.get("order_status") == "SUBMITTED":
                    broker.cancel_order(lr["order_id"])
            rec["status"] = "ABORTED_ENTRY"
            rec["capital"] = 0.0
            structs.append(rec)
            journal.append(f"- ABORT {code} {strat}: a leg was rejected; siblings cancelled")
            continue
        structs.append(rec)
        existing_ids.add(sid)
        capital_in_use += capital
        open_unders.add(code)
        placed += 1
        entered += 1
        journal.append(f"- ENTER {code} {strat}: {n_contracts}x {len(leg_recs)} leg(s) at mid, "
                       f"capital ~${capital:,.0f} (POP {row0['pop_pct']}%, EV {row0['ev_per_share']})")
    if entered == 0:
        journal.append("- no new structures entered")
    return entered


def main() -> None:
    from app.brokers.paper_broker import PaperBroker

    budget = None
    if "--budget" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--budget") + 1])
    max_new = int(sys.argv[sys.argv.index("--max-new") + 1]) if "--max-new" in sys.argv else None
    today = dt.date.today().isoformat()
    sig_path = SIGNALS_DIR / f"{today}.csv"
    JOURNAL_DIR.mkdir(parents=True, exist_ok=True)
    structs = _load_structures()
    journal: list[str] = []
    exit_code = 0
    broker = None
    placed = 0
    equity = None

    try:
        broker = PaperBroker().connect()
        acc = broker.account()
        equity = float(acc.iloc[0]["total_assets"]) if "total_assets" in acc.columns else None

        # 1. hygiene: cancel stale unfilled orders (failed-fill datapoints).
        # Runs FIRST so lifecycle/exits can re-place fresh orders after it.
        journal.append("## Order hygiene")
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
            journal.append(f"- cancelled {n_cancel} stale unfilled order(s)")
        except Exception as e:  # noqa: BLE001
            journal.append(f"- hygiene skipped ({e})")

        # 2. lifecycle sync: CANCELLED_UNFILLED / BROKEN / CLOSED transitions
        # and crossing-price retries for CLOSING structures
        journal.append("\n## Lifecycle sync")
        journal += sync_lifecycle(broker, structs, today) or ["- all states consistent"]
        _save_structures(structs)

        # 3. rule-based exits
        journal.append("\n## Exits")
        journal += check_exits(broker, structs, today) or ["- none"]
        _save_structures(structs)

        # 4. chase yesterday-and-today's resting entry legs toward fill (modify)
        journal.append("\n## Entry-fill chase")
        journal += chase_entry_fills(broker, structs, today)

        # 5. entries (EV-ranked, per-day capped, leg-validated)
        journal.append("\n## Entries")
        placed = do_entries(broker, structs, journal, sig_path, budget, today, max_new)
        _save_structures(structs)

        # 5. reconciliation: broker truth beats our order records
        journal.append("\n## Reconciliation (broker truth)")
        journal += reconcile(broker, structs)
        _save_structures(structs)
    except Exception as e:  # noqa: BLE001
        import traceback
        journal.append(f"\n## RUN ERROR\n- {e}\n```\n{traceback.format_exc()}\n```")
        _save_structures(structs)
        exit_code = 1

    # 6. journal note (Obsidian-ready) + dashboard -- always written
    n_open_now = sum(1 for s in structs if s["status"] in ("OPEN", "CLOSING"))
    closing = [s for s in structs if s["status"] == "CLOSING"]
    eq_s = f"{equity:,.0f}" if equity is not None else "n/a"
    note = JOURNAL_DIR / f"{today}.md"
    front = (f"---\ndate: {today}\ntype: paper-journal\nentries_placed: {placed}\n"
             f"open_structures: {n_open_now}\npaper_equity: {equity if equity is not None else 'null'}\n---\n")
    body = (f"# Paper journal — {today}\n\n"
            f"Paper account equity: **{eq_s}** · open structures: **{n_open_now}**\n\n"
            + "\n".join(journal) + "\n\n[[Dashboard]]\n")
    note.write_text(front + body, encoding="utf-8")

    dash = JOURNAL_DIR / "Dashboard.md"
    closed_all = [s for s in structs if s["status"] in ("CLOSED", "CLOSING", "CANCELLED_UNFILLED", "ABORTED_ENTRY")]
    dash.write_text(
        "---\ntype: paper-dashboard\n---\n# Paper Trading Dashboard\n\n"
        f"Updated: {today}\n\n- Paper equity: **{eq_s}**\n"
        f"- Open structures: **{n_open_now}**\n"
        f"- Structures ever entered: **{len(structs)}**\n"
        f"- Closed / closing / cancelled: **{len(closed_all)}**\n\n"
        "## Recent daily notes\n"
        + "\n".join(f"- [[{p.stem}]]" for p in sorted(JOURNAL_DIR.glob('2*.md'), reverse=True)[:15])
        + "\n\nInteractive charts: `dashboard.html` · ledgers in `data_store/paper/`\n",
        encoding="utf-8")

    try:
        from scripts.build_dashboard import build, snapshot_equity
        if equity is not None:
            cash = float(acc.iloc[0]["cash"]) if "cash" in acc.columns else None
            snapshot_equity(equity, cash)
        build()
    except Exception as e:  # noqa: BLE001
        print(f"dashboard refresh failed: {e}")

    print(f"{today}: {placed} entered, {len(closing)} closing, {n_open_now} open -> {note}")
    if broker is not None:
        broker.close()
    sys.stdout.flush()
    import os
    os._exit(exit_code)   # ALWAYS hard-exit: moomoo threads must never hang the task


if __name__ == "__main__":
    main()
