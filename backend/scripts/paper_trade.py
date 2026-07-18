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
# The per-leg execution gate lives in log_signals (leg_tradeable) so the entry
# check and the signal prefilter cannot drift apart -- they were duplicated
# constants with a "keep them in lockstep" comment, which is how drift starts.
from scripts.log_signals import MIN_OI as MIN_LEG_OI, leg_tradeable  # noqa: E402
from scripts._session import session_date  # noqa: E402
MAX_NEW_PER_DAY = 8
MAX_OPEN_STRUCTURES = 30
# Diversity cap (weekly review 2026-07-18): regime gate + drift prior + EV/risk
# ranking made the book 22/25 long-delta -- one selloff hit everything at once.
# Capping same-strategy entries per night admits the best OTHER structures
# without touching any validated per-structure formula. Forward-validate.
MAX_PER_STRATEGY_PER_DAY = 3
# EV floor: EV/max_loss ranking preferentially tops exactly the small-denominator
# structures where round-trip friction (~$2.60 on a 2-leg spread) eats 7-50% of
# the modeled edge. No commission term exists in EV; this floor stands in for it.
MIN_EV_PER_SHARE = 0.05

# Kill switch: data_store/HALT (manual, just create the file) or a daily loss
# beyond DAILY_LOSS_HALT_PCT flips the run to degraded mode -- exits/hygiene
# still run, NEW entries are skipped. Fail-open direction is "skip entries",
# never "skip exits".
HALT_FILE = BACKEND / "data_store" / "HALT"
DAILY_LOSS_HALT_PCT = 2.0

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

# --- tail hedge overlay (weekly review 2026-07-18) ---------------------------
# The book is ~all long-delta bullish structures with NO protection on the
# standing positions; the 200-SMA regime gate only reshapes FUTURE entries, and
# it hasn't engaged (SPY ~8% above its 200-SMA). This overlay holds a balanced
# SPY put debit spread as portfolio insurance: long ~5.5% OTM, short ~12% OTM
# (caps payoff, finances carry), ~45 DTE, rolled at 21 DTE, sized to a monthly
# carry budget. Defined risk (long strike above short), so never a naked short.
# manage_hedge() owns it; check_exits skips kind=="HEDGE" so the premium-harvest
# rules can't profit-take the insurance away before the crash it's there for.
HEDGE_ENABLED = True
HEDGE_UNDER = "US.SPY"
HEDGE_LONG_OTM = 0.055        # long put strike ~5.5% below spot
HEDGE_SHORT_OTM = 0.12        # short put strike ~12% below spot
HEDGE_TARGET_DTE = 45
HEDGE_ROLL_DTE = 21           # roll when the active hedge decays to this
HEDGE_MONTHLY_BUDGET = 1200.0 # ~0.4% of a 300k book; bounds the carry per roll
HEDGE_MAX_CONTRACTS = 15


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
        if all(leg_tradeable(l["bid"], l["ask"], l["oi"]) for _, l in g.iterrows()):
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
        signed = bqty.get(leg["code"], 0.0)
        held = abs(signed)
        qty = min(leg["qty"], held)
        if qty < 1:
            continue
        # flatten side from the BROKER's position sign, not the ledger's memory:
        # if any state drift ever leaves the position flipped, closing "the
        # ledger's side" would grow the error instead of erasing it
        side = "SELL" if signed > 0 else "BUY"
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
    # Broker-state failure is RUN-FATAL, not a skip: proceeding without
    # positions would leave CLOSING structures without working exit orders
    # while the night still reads "successful".
    bqty, _ = _broker_qty(broker)
    retry_codes = [l["code"] for s in structs if s["status"] == "CLOSING" for l in s["legs"]]
    snap = _live_snapshot(retry_codes) if retry_codes else None
    # open orders once, for CLOSING-exit refresh and PENDING adoption below.
    # RUN-FATAL on failure (like positions): with an empty order map the
    # CLOSING cancel-half would silently no-op while the re-place half still
    # fires -- duplicate exits that can flip a position and grow it nightly.
    try:
        from moomoo import OrderStatus
        open_orders = broker.orders(status_filter_list=[
            OrderStatus.SUBMITTED, OrderStatus.WAITING_SUBMIT, OrderStatus.SUBMITTING])
    except ImportError:
        open_orders = None          # test environment without the SDK
    open_by_code: dict[str, list[str]] = {}
    if open_orders is not None and not open_orders.empty:
        for _, o in open_orders.iterrows():
            open_by_code.setdefault(str(o["code"]), []).append(str(o["order_id"]))

    for s in structs:
        if s["status"] == "PENDING_ENTRY":
            # write-ahead record from a run killed mid-placement: adopt if the
            # broker shows any evidence of it, else mark aborted
            claimed = {l["code"] for x in structs if x is not s
                       and x["status"] in ("OPEN", "CLOSING") for l in x["legs"]}
            evidence = any((abs(bqty.get(l["code"], 0.0)) > 0 or l["code"] in open_by_code)
                           and l["code"] not in claimed for l in s["legs"])
            s["status"] = "OPEN" if evidence else "ABORTED_ENTRY"
            if not evidence:
                s["capital"] = 0.0
                s["exit_reason"] = "run killed before any leg was placed"
            lines.append(f"- {s['underlying']} {s['strategy']}: PENDING_ENTRY -> {s['status']}"
                         f" (killed mid-entry {s['entry_date']})")
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
                # this structure OWNS its exit refresh: cancel its stale exit
                # orders, then re-place -- back-to-back, so there is no window
                # where the legs sit with no working exit (hygiene no longer
                # touches CLOSING codes at all)
                for l in held:
                    for oid in open_by_code.get(l["code"], []):
                        broker.cancel_order(oid)
                _close_legs(broker, s, held, f"retry: {s.get('exit_reason', 'exit')}", bqty, snap)
                lines.append(f"- {s['underlying']} {s['strategy']}: refreshed exit on {len(held)} leg(s) at crossing prices")
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
        if s.get("kind") == "HEDGE":
            continue    # insurance: manage_hedge rolls it; premium-exit rules must not touch it
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
    today = session_date()
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
        and s.get("kind") != "HEDGE"      # the hedge has its own budget; never a per-day entry slot
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
    # freshness gate: refuse to trade a stale signal file (a leftover copy of
    # yesterday's file once nearly traded on yesterday's mids/IVs/spots)
    if "date" in sig.columns and str(sig["date"].iloc[0]) != today:
        journal.append(f"- signals file is dated {sig['date'].iloc[0]}, session is {today}"
                       f" -- STALE, refusing to enter")
        return 0
    capital_in_use = sum(s.get("capital", 0.0) for s in structs
                         if s["status"] in ("OPEN", "CLOSING"))
    # rank candidates by model EV per unit risk, best first (was alphabetical)
    groups = sorted(structure_groups(sig),
                    key=lambda t: -((t[2]["ev_per_share"].iloc[0] or 0)
                                    / max(t[2]["max_loss"].iloc[0] or 1e9, 0.01)))
    state = {"entered": 0, "placed": placed, "capital_in_use": capital_in_use}
    by_strategy: dict[str, int] = {}
    for s in structs:
        if s.get("entry_date") == today and s.get("kind") != "HEDGE" \
                and s["status"] in ("OPEN", "CLOSING", "PENDING_ENTRY"):
            by_strategy[s["strategy"]] = by_strategy.get(s["strategy"], 0) + 1

    def try_enter(code, strat, legs, sid) -> str:
        """Attempt one structure. Returns 'entered' | 'aborted' | 'skipped'."""
        row0 = legs.iloc[0]
        net = float(row0["net_debit_credit"]) if pd.notna(row0["net_debit_credit"]) else None
        max_loss = float(row0["max_loss"]) if pd.notna(row0["max_loss"]) else None
        leg_meta = [{"strike": float(l["leg_strike"])} for _, l in legs.iterrows()]
        n_contracts, capital, why = size_structure(budget, strat, leg_meta, net,
                                                   max_loss, state["capital_in_use"])
        if n_contracts < 1:
            journal.append(f"- skip {code} {strat}: {why}")
            return "skipped"
        # max profit: credit = the credit; VERTICALS (one buy + one sell) =
        # width - debit. All-long 2-leg structures (straddle) have none.
        strikes = [float(l["leg_strike"]) for _, l in legs.iterrows()]
        sides = {str(l["leg_action"]).upper() for _, l in legs.iterrows()}
        max_profit = None
        if net is not None and net > 0:
            max_profit = net
        elif net is not None and len(strikes) == 2 and sides == {"BUY", "SELL"}:
            width = abs(strikes[0] - strikes[1])
            max_profit = width - abs(net) if width - abs(net) > 0 else None
        # full leg skeletons FIRST (codes/sides/qty from the signal rows) so
        # the write-ahead record is adoptable: lifecycle sync matches broker
        # evidence BY LEG CODE, and an empty legs list can never match anything
        leg_recs = []
        for _, leg in legs.iterrows():
            leg_recs.append({
                "code": leg["leg_code"], "side": leg["leg_action"].upper(), "qty": n_contracts,
                "right": leg["leg_right"], "strike": float(leg["leg_strike"]),
                "entry_mid": round(_mid(leg), 2),
                "theo": round(float(leg["theo_bsm"]), 2) if pd.notna(leg["theo_bsm"]) else None,
                "bid": float(leg["bid"]), "ask": float(leg["ask"]),
                "order_status": "PENDING", "order_id": None,
            })
        rec = {
            "id": sid, "underlying": code, "strategy": strat, "status": "PENDING_ENTRY",
            "entry_date": today, "expiry": str(row0["expiry"]), "dte": int(row0["dte"]),
            "pop_pct": float(row0["pop_pct"]) if pd.notna(row0["pop_pct"]) else None,
            "ev_per_share": float(row0["ev_per_share"]) if pd.notna(row0["ev_per_share"]) else None,
            "net_debit_credit": net, "max_loss": max_loss, "max_profit": max_profit,
            "contracts": n_contracts, "capital": round(capital, 2),
            "legs": leg_recs,
        }
        # WRITE-AHEAD: the record hits disk BEFORE any order leaves. If this
        # process dies mid-placement, the next run's lifecycle sync adopts or
        # aborts the PENDING row instead of discovering orphan fills months on.
        structs.append(rec)
        _save_structures(structs)
        rejected = False
        for lr in leg_recs:
            res = broker.place_limit(lr["code"], n_contracts, lr["side"],
                                     lr["entry_mid"], note=f"ENTRY {sid}")
            if res.get("status") == "REJECTED":
                rejected = True
            lr["order_status"] = res.get("status")
            lr["order_id"] = res.get("order_id")
        if rejected:
            # abort: cancel the sibling orders that did submit; hold no capital
            for lr in leg_recs:
                if lr.get("order_id") and lr.get("order_status") == "SUBMITTED":
                    broker.cancel_order(lr["order_id"])
            rec["status"] = "ABORTED_ENTRY"
            rec["capital"] = 0.0
            _save_structures(structs)
            journal.append(f"- ABORT {code} {strat}: a leg was rejected; siblings cancelled")
            return "aborted"
        rec["status"] = "OPEN"
        _save_structures(structs)
        state["capital_in_use"] += capital
        open_unders.add(code)
        by_strategy[strat] = by_strategy.get(strat, 0) + 1
        state["placed"] += 1
        state["entered"] += 1
        journal.append(f"- ENTER {code} {strat}: {n_contracts}x {len(leg_recs)} leg(s) at mid, "
                       f"capital ~${capital:,.0f} (POP {row0['pop_pct']}%, EV {row0['ev_per_share']})")
        return "entered"

    aborted_tonight: list[tuple] = []
    for code, strat, legs in groups:
        if state["placed"] >= cap or n_open + state["entered"] >= MAX_OPEN_STRUCTURES:
            journal.append("- caps reached; remaining signals skipped")
            break
        if code in open_unders:
            continue                    # one structure per underlying
        if by_strategy.get(strat, 0) >= MAX_PER_STRATEGY_PER_DAY:
            journal.append(f"- skip {code} {strat}: strategy cap "
                           f"({MAX_PER_STRATEGY_PER_DAY}/night) -- diversity")
            continue
        ev = legs.iloc[0]["ev_per_share"]
        if pd.notna(ev) and float(ev) < MIN_EV_PER_SHARE:
            journal.append(f"- skip {code} {strat}: EV {float(ev):.02f}/sh below "
                           f"friction floor {MIN_EV_PER_SHARE}")
            continue
        if EXPECTED_LEGS.get(strat) is not None and len(legs) != EXPECTED_LEGS[strat]:
            journal.append(f"- skip {code} {strat}: {len(legs)} legs in signals, "
                           f"expected {EXPECTED_LEGS[strat]} (partial row-set)")
            continue
        sid = f"{today}-{code}-{strat}".replace(" ", "_")
        if sid in existing_ids:
            continue                    # same-day re-entry after a close: skip
        existing_ids.add(sid)
        if try_enter(code, strat, legs, sid) == "aborted":
            aborted_tonight.append((code, strat, legs, sid))

    # one same-session retry for rate-limit/transient aborts: the throttle has
    # cleared the 30s window by now, and RIVN/IONQ-class candidates used to
    # die twice and never execute
    for code, strat, legs, sid in aborted_tonight:
        if state["placed"] >= cap or n_open + state["entered"] >= MAX_OPEN_STRUCTURES:
            break
        if code in open_unders:
            continue
        if by_strategy.get(strat, 0) >= MAX_PER_STRATEGY_PER_DAY:
            continue                # the diversity cap binds retries too
        rsid = sid + "-r1"
        if rsid in existing_ids:
            continue
        existing_ids.add(rsid)
        journal.append(f"- RETRY {code} {strat} (one shot)")
        try_enter(code, strat, legs, rsid)

    if state["entered"] == 0:
        journal.append("- no new structures entered")
    return state["entered"]


def _pick_hedge_spread(spot: float, puts, budget: float) -> dict | None:
    """Pure selection + sizing for the SPY put debit spread. `puts` is a DataFrame
    with strike/bid/ask (and optional code). Returns the chosen legs, contract
    count, cost and modeled drawdown payoffs, or None if no sane spread exists.
    Kept pure (no I/O) so test_hedge can exercise it."""
    if puts is None or len(puts) == 0:
        return None
    p = puts.copy()
    p = p[(p["bid"] > 0) & (p["ask"] >= p["bid"])]
    if p.empty:
        return None
    p["mid"] = (p["bid"] + p["ask"]) / 2.0
    long_leg = p.iloc[(p["strike"] - spot * (1 - HEDGE_LONG_OTM)).abs().argmin()]
    short_leg = p.iloc[(p["strike"] - spot * (1 - HEDGE_SHORT_OTM)).abs().argmin()]
    if short_leg["strike"] >= long_leg["strike"]:
        return None                       # need a real width: short strictly below long
    net_debit = float(long_leg["mid"] - short_leg["mid"])
    if net_debit <= 0:                    # degenerate/inverted quotes
        return None
    width = float(long_leg["strike"] - short_leg["strike"])
    n = max(1, min(int(budget // (net_debit * CONTRACT_SIZE)) or 1, HEDGE_MAX_CONTRACTS))

    def payoff(drop: float) -> float:
        s2 = spot * (1 - drop)
        intrinsic = min(width, max(0.0, float(long_leg["strike"]) - s2))
        return round((intrinsic - net_debit) * CONTRACT_SIZE * n, 0)

    return {
        "long_strike": float(long_leg["strike"]), "short_strike": float(short_leg["strike"]),
        "long_code": str(long_leg.get("code", "")), "short_code": str(short_leg.get("code", "")),
        "long_mid": round(float(long_leg["mid"]), 2), "short_mid": round(float(short_leg["mid"]), 2),
        "net_debit": round(net_debit, 2), "width": width, "contracts": n,
        "cost": round(net_debit * CONTRACT_SIZE * n, 2),
        "payoff": {"10%": payoff(0.10), "15%": payoff(0.15), "20%": payoff(0.20)},
    }


def _spy_otm_puts(expiry: str, spot: float):
    """Live OTM put quotes for the hedge band. The offline recorder keeps only
    near-money strikes, so this needs OpenD. Returns a DataFrame or None."""
    import pandas as pd
    from app.services.analysis_service import service
    try:
        with service._lock:
            chain = service._client.get_option_chain(HEDGE_UNDER, expiry, expiry)
        chain = chain.copy()
        chain["strike"] = pd.to_numeric(chain["strike_price"], errors="coerce")
        puts = chain[chain["option_type"].astype(str).str.upper().str.contains("PUT")]
        lo, hi = spot * (1 - HEDGE_SHORT_OTM - 0.03), spot * (1 - HEDGE_LONG_OTM + 0.03)
        band = puts[(puts["strike"] >= lo) & (puts["strike"] <= hi)]
        if band.empty:
            return None
        with service._lock:
            snap = service._client.get_snapshot(band["code"].tolist()).set_index("code")
        rows = []
        for _, r in band.iterrows():
            c = r["code"]
            if c not in snap.index:
                continue
            s = snap.loc[c]
            rows.append({"code": c, "strike": float(r["strike"]),
                         "bid": float(s.get("bid_price") or 0), "ask": float(s.get("ask_price") or 0),
                         "delta": float(s.get("option_delta") or 0),
                         "oi": float(s.get("option_open_interest") or 0)})
        return pd.DataFrame(rows) if rows else None
    except Exception as e:  # noqa: BLE001
        print(f"  hedge chain fetch failed: {e}", flush=True)
        return None


def manage_hedge(broker, structs: list[dict], today: str, journal: list[str]) -> None:
    """Hold or roll the SPY tail-hedge put spread. Runs each session before entries."""
    if not HEDGE_ENABLED:
        return
    today_d = dt.date.fromisoformat(today)
    active = [s for s in structs if s.get("kind") == "HEDGE" and s["status"] in ("OPEN", "CLOSING")]

    live = []
    for s in active:
        try:
            dleft = (dt.date.fromisoformat(s["expiry"]) - today_d).days
        except Exception:  # noqa: BLE001
            dleft = None
        if s["status"] == "OPEN" and dleft is not None and dleft <= HEDGE_ROLL_DTE:
            bqty, _ = _broker_qty(broker)
            snap = _live_snapshot([l["code"] for l in s["legs"]])
            _close_legs(broker, s, s["legs"], f"hedge roll (DTE {dleft})", bqty, snap)
            s["status"] = "CLOSING"
            journal.append(f"- HEDGE roll: closing {s['id']} (DTE {dleft} <= {HEDGE_ROLL_DTE})")
        elif s["status"] == "OPEN":
            live.append((s, dleft))

    if live:
        s, dleft = live[0]
        journal.append(f"- HEDGE active: {s['id']} DTE {dleft}, holding")
        return

    try:
        from app.services.analysis_service import service
        with service._lock:
            snap0 = service._client.get_snapshot([HEDGE_UNDER])
        spot = float(snap0.iloc[0]["last_price"])
        picked = service._pick_option_expiry(HEDGE_UNDER, HEDGE_TARGET_DTE)
    except Exception as e:  # noqa: BLE001
        journal.append(f"- HEDGE skipped: SPY spot/expiry fetch failed ({e})")
        return
    if not spot or picked is None:
        journal.append("- HEDGE skipped: no SPY spot/expiry")
        return
    _, expiry, dte, _ = picked
    pick = _pick_hedge_spread(spot, _spy_otm_puts(expiry, spot), HEDGE_MONTHLY_BUDGET)
    if pick is None:
        journal.append("- HEDGE skipped: no tradeable OTM put spread found")
        return

    sid = f"{today}-HEDGE-SPY"
    if any(s["id"] == sid for s in structs):
        journal.append("- HEDGE already placed today; skipping")
        return
    n = pick["contracts"]
    legs_spec = [("BUY", pick["long_strike"], pick["long_code"], pick["long_mid"]),
                 ("SELL", pick["short_strike"], pick["short_code"], pick["short_mid"])]
    leg_recs, rejected = [], False
    for side, strike, code, mid in legs_spec:
        res = broker.place_limit(code, n, side, mid, note=f"HEDGE {sid}")
        if res.get("status") == "REJECTED":
            rejected = True
        leg_recs.append({"code": code, "side": side, "qty": n, "right": "PUT",
                         "strike": strike, "entry_mid": round(mid, 2),
                         "order_status": res.get("status"), "order_id": res.get("order_id")})
    rec = {
        "id": sid, "underlying": HEDGE_UNDER, "strategy": "SPY Tail Hedge", "kind": "HEDGE",
        "status": "OPEN", "entry_date": today, "expiry": str(expiry), "dte": int(dte),
        "net_debit_credit": -pick["net_debit"], "max_loss": pick["net_debit"],
        "max_profit": round(pick["width"] - pick["net_debit"], 2),
        "contracts": n, "capital": pick["cost"], "legs": leg_recs,
    }
    if rejected:
        for lr in leg_recs:
            if lr.get("order_id") and lr.get("order_status") == "SUBMITTED":
                broker.cancel_order(lr["order_id"])
        rec["status"] = "ABORTED_ENTRY"; rec["capital"] = 0.0
        structs.append(rec)
        journal.append("- HEDGE abort: a leg rejected; siblings cancelled")
        return
    structs.append(rec)
    journal.append(
        f"- HEDGE placed: {n}x SPY {pick['long_strike']:.0f}/{pick['short_strike']:.0f}p "
        f"@ ${pick['net_debit']:.2f} (cost ${pick['cost']:.0f}), DTE {dte} | pays "
        f"~${pick['payoff']['10%']:,.0f} at -10%, ~${pick['payoff']['15%']:,.0f} at -15%")


def orphan_check(broker, structs: list[dict], journal: list[str]) -> None:
    """Broker->ledger direction: any held option position that belongs to NO
    open structure is invisible to every other check (reconcile iterates ledger
    codes only). Journal it loudly and flatten it -- an untracked leg has no
    exit rule watching it."""
    try:
        bqty, pos = _broker_qty(broker)
    except Exception as e:  # noqa: BLE001
        journal.append(f"- orphan check skipped: {e}")
        return
    known = {l["code"] for s in structs if s["status"] in ("OPEN", "CLOSING", "PENDING_ENTRY")
             for l in s["legs"]}
    # OPTION legs only: US.XXX<yymmdd>[C|P]<strike>. A substring heuristic would
    # match stock tickers like US.COST and try to flatten a SHARE position.
    import re
    opt_pat = re.compile(r"\d{6}[CP]\d+$")
    orphans = [c for c in bqty if c not in known and opt_pat.search(c)]
    for c in orphans:
        qty = bqty[c]
        side = "SELL" if qty > 0 else "BUY"
        snap = _live_snapshot([c])
        px = None
        if snap is not None and c in snap.index:
            row = snap.loc[c]
            px = float(row["bid_price"]) if side == "SELL" else float(row["ask_price"])
        if px and px > 0:
            broker.place_limit(c, abs(qty), side, round(px, 2), note="ORPHAN flatten")
            journal.append(f"- ORPHAN {c}: qty {qty} in no structure -- flattening at crossing")
        else:
            journal.append(f"- ORPHAN {c}: qty {qty} in no structure -- NO QUOTE, flatten manually")
    if orphans:
        try:
            from scripts._alert import alert
            alert("Paper book ORPHAN legs", f"{len(orphans)} untracked position(s): "
                  f"{', '.join(orphans[:6])}", priority="high")
        except Exception:  # noqa: BLE001
            pass


def post_close_sweep(broker, structs: list[dict], today: str, journal: list[str]) -> None:
    """Runs AFTER the US close (16:05 ET, same session): tonight's day orders
    are dead, so any partially-filled structure is now a fact, not a
    work-in-progress. Cancel leftover entry orders, flatten filled legs in the
    SAME session -- the CVX/SCHW class used to carry a lone leg 24h (72h over
    a weekend) before the next night's lifecycle sync noticed."""
    bqty, _ = _broker_qty(broker)
    # belt-and-braces: cancel any still-listed entry orders for tonight's
    # partial/unfilled structures before judging them (the market IS closed --
    # the guard in main enforces that -- but a listed order costs a cancel)
    try:
        from moomoo import OrderStatus
        oo = broker.orders(status_filter_list=[
            OrderStatus.SUBMITTED, OrderStatus.WAITING_SUBMIT, OrderStatus.SUBMITTING])
    except Exception:  # noqa: BLE001
        oo = None
    order_ids: dict[str, list[str]] = {}
    if oo is not None and len(oo):
        for _, o in oo.iterrows():
            order_ids.setdefault(str(o["code"]), []).append(str(o["order_id"]))
    for s in structs:
        if s.get("entry_date") != today or s["status"] != "OPEN" or s.get("kind") == "HEDGE":
            continue
        held = [l for l in s["legs"] if abs(bqty.get(l["code"], 0.0)) > 0]
        if len(held) == len(s["legs"]):
            continue                              # fully filled: a real position
        for l in s["legs"]:                       # kill lingering entry orders first
            if abs(bqty.get(l["code"], 0.0)) == 0:
                for oid in order_ids.get(l["code"], []):
                    broker.cancel_order(oid)
        if not held:
            s["status"] = "CANCELLED_UNFILLED"
            s["exit_reason"] = "day orders expired unfilled (post-close sweep)"
            s["exit_date"] = today
            journal.append(f"- SWEEP {s['underlying']} {s['strategy']}: nothing filled -> CANCELLED_UNFILLED")
            continue
        s["status"] = "CLOSING"
        s["exit_reason"] = "partial fill at close -- flattening same session"
        s["exit_date"] = today
        snap = _live_snapshot([l["code"] for l in held])
        _close_legs(broker, s, held, s["exit_reason"], bqty, snap)
        journal.append(f"- SWEEP {s['underlying']} {s['strategy']}: {len(held)}/{len(s['legs'])} "
                       f"legs filled -> flattening now (no overnight lone legs)")


def main() -> None:
    from app.brokers.paper_broker import PaperBroker

    budget = None
    if "--budget" in sys.argv:
        budget = float(sys.argv[sys.argv.index("--budget") + 1])
    max_new = int(sys.argv[sys.argv.index("--max-new") + 1]) if "--max-new" in sys.argv else None
    today = session_date()      # ET session date: a post-midnight (SGT) retry stays on tonight's session
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

        if "--sweep" in sys.argv:
            # post-close mode: partial-fill repair + orphan check only -- no
            # entries, no exits (those ran at 23:40). HARD time gate: judging
            # "day orders are dead" while the market is open would cancel and
            # flatten the run's own still-working entries (and 04:05 SGT is
            # BEFORE the close all winter -- EST moves the close to 05:00 SGT).
            import zoneinfo
            now_et = dt.datetime.now(zoneinfo.ZoneInfo("America/New_York"))
            if now_et.strftime("%H:%M") < "16:01" and now_et.date().isoformat() == today:
                print(f"sweep refused: US market not closed yet ({now_et:%H:%M} ET)")
                broker.close()
                return
            journal.append("## Post-close sweep")
            post_close_sweep(broker, structs, today, journal)
            orphan_check(broker, structs, journal)
            _save_structures(structs)
            (PAPER_DIR / f"sweep_{today}.done").write_text(
                dt.datetime.now().isoformat(timespec="seconds"), encoding="utf-8")
            note = JOURNAL_DIR / f"{today}.md"
            if note.exists():
                note.write_text(note.read_text(encoding="utf-8")
                                + "\n\n## Post-close sweep\n" + "\n".join(journal[1:] or ["- clean"])
                                + "\n", encoding="utf-8")
            try:
                from scripts.build_dashboard import build, write_marks
                write_marks(broker)
                build()
            except Exception as e:  # noqa: BLE001
                print(f"sweep dashboard refresh failed: {e}")
            print(f"{today}: post-close sweep done")
            broker.close()
            return

        # 1. hygiene: cancel stale unfilled orders (failed-fill datapoints).
        # Runs FIRST so lifecycle/exits can re-place fresh orders after it.
        journal.append("## Order hygiene")
        try:
            from moomoo import OrderStatus
            open_orders = broker.orders(status_filter_list=[
                OrderStatus.SUBMITTED, OrderStatus.WAITING_SUBMIT, OrderStatus.SUBMITTING])
            stale = open_orders[open_orders.get("create_time", "").astype(str).str[:10] < today] \
                if not open_orders.empty else open_orders
            # never cancel a CLOSING structure's working EXIT order here: that
            # cancel belongs to sync_lifecycle, paired atomically with its
            # re-place (a kill between blanket-cancel and re-place used to
            # leave held legs with no exit order at all)
            closing_codes = {l["code"] for s in structs
                             if s["status"] == "CLOSING" for l in s["legs"]}
            n_cancel = 0
            if not stale.empty:
                for _, o in stale.iterrows():
                    if str(o["code"]) in closing_codes:
                        continue
                    if broker.cancel_order(str(o["order_id"])):
                        n_cancel += 1
            journal.append(f"- cancelled {n_cancel} stale unfilled ENTRY order(s) "
                           f"(exit orders left to lifecycle sync)")
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

        # 3b. tail hedge overlay: hold/roll SPY put spread (check_exits skips it)
        journal.append("\n## Tail hedge")
        manage_hedge(broker, structs, today, journal)
        _save_structures(structs)

        # 4. chase yesterday-and-today's resting entry legs toward fill (modify)
        journal.append("\n## Entry-fill chase")
        journal += chase_entry_fills(broker, structs, today)

        # 5. entries (EV-ranked, per-day capped, leg-validated) -- unless halted.
        # HALT file = manual kill switch; daily-loss breach = automatic one.
        # Both degrade to management-only: exits/hygiene/hedge ran above and
        # always will; only NEW risk is refused. Guard fails OPEN to "skip
        # entries", never to "skip exits".
        journal.append("\n## Entries")
        halt_reason = None
        if HALT_FILE.exists():
            halt_reason = "HALT file present (manual kill switch)"
        else:
            try:
                eq_csv = pd.read_csv(PAPER_DIR / "equity.csv")
                if len(eq_csv) >= 2:
                    prev, last = float(eq_csv["equity"].iloc[-2]), float(eq_csv["equity"].iloc[-1])
                    day_pct = (last - prev) / prev * 100
                    if day_pct <= -DAILY_LOSS_HALT_PCT:
                        halt_reason = f"daily loss {day_pct:.1f}% breaches -{DAILY_LOSS_HALT_PCT}%"
            except Exception:  # noqa: BLE001 - unreadable equity file must not block entries
                pass
        if halt_reason:
            journal.append(f"- ENTRIES SKIPPED: {halt_reason}")
            try:
                from scripts._alert import alert
                alert("Paper pipeline HALTED entries", f"{today}: {halt_reason}", priority="urgent")
            except Exception:  # noqa: BLE001
                pass
        else:
            placed = do_entries(broker, structs, journal, sig_path, budget, today, max_new)
        _save_structures(structs)

        # 5. reconciliation: broker truth beats our order records
        journal.append("\n## Reconciliation (broker truth)")
        journal += reconcile(broker, structs)
        orphan_check(broker, structs, journal)   # broker->ledger direction
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
        from scripts.build_dashboard import build, snapshot_equity, write_marks
        if equity is not None:
            cash = float(acc.iloc[0]["cash"]) if "cash" in acc.columns else None
            snapshot_equity(equity, cash)
        if broker is not None:
            write_marks(broker)
        build()
    except Exception as e:  # noqa: BLE001
        print(f"dashboard refresh failed: {e}")

    print(f"{today}: {placed} entered, {len(closing)} closing, {n_open_now} open -> {note}")
    if broker is not None:
        broker.close()
    return exit_code


if __name__ == "__main__":
    from scripts._lock import single_instance
    rc = 0
    with single_instance("paper_trade") as got:
        if got:
            rc = main() or 0
        else:
            # two concurrent paper_trade processes would DOUBLE-PLACE entries
            # (sid dedup is per-process); the backstop yields to the live run
            print("another paper_trade instance holds the lock -- exiting (work is being done)")
    # hard-exit AFTER the lock is released: moomoo SDK non-daemon threads
    # (quote/trade contexts) otherwise hang the process until the runner's
    # timeout kills it -- and an os._exit inside the lock scope leaks the lock
    sys.stdout.flush()
    import os
    os._exit(rc)
