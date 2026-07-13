"""Build the daily paper-trading dashboard (self-contained HTML, private).

Reads baseline.json, equity.csv, structures.jsonl, order_ledger.jsonl and the
latest VRP reading; writes data_store/journal/dashboard.html. Called at the
end of every paper_trade.py run and runnable standalone:

    PYTHONPATH=. .venv/Scripts/python.exe scripts/build_dashboard.py
"""
from __future__ import annotations

import datetime as dt
import html
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
PAPER = BACKEND / "data_store" / "paper"
JOURNAL = BACKEND / "data_store" / "journal"


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def build() -> Path:
    import pandas as pd

    base = json.loads((PAPER / "baseline.json").read_text()) if (PAPER / "baseline.json").exists() else {}
    start_eq = base.get("account_equity_at_start")
    budget = base.get("working_budget_usd")

    eq_path = PAPER / "equity.csv"
    eq = pd.read_csv(eq_path) if eq_path.exists() else pd.DataFrame(columns=["date", "equity", "cash"])
    cur_eq = float(eq["equity"].iloc[-1]) if len(eq) else start_eq
    pnl = (cur_eq - start_eq) if (cur_eq is not None and start_eq) else None

    structs = load_jsonl(PAPER / "structures.jsonl")
    open_s = [s for s in structs if s.get("status") in ("OPEN", "CLOSING")]
    closed = [s for s in structs if s.get("status") not in ("OPEN", "CLOSING")]
    cap_in_use = sum(s.get("capital", 0) or 0 for s in open_s)

    orders = load_jsonl(PAPER / "order_ledger.jsonl")
    n_sub = sum(1 for o in orders if o.get("status") == "SUBMITTED")
    n_rej = sum(1 for o in orders if o.get("status") == "REJECTED")
    n_can = sum(1 for o in orders if o.get("status") == "CANCELLED")

    # equity sparkline (inline SVG)
    spark = ""
    if len(eq) >= 2:
        v = eq["equity"].astype(float).tolist()
        lo, hi = min(v), max(v)
        rng = (hi - lo) or 1.0
        pts = " ".join(f"{i * (280 / (len(v) - 1)):.1f},{40 - (x - lo) / rng * 36:.1f}"
                       for i, x in enumerate(v))
        color = "#1f7a52" if v[-1] >= v[0] else "#b23b3b"
        spark = (f'<svg width="280" height="44" viewBox="0 0 280 44">'
                 f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{pts}"/></svg>')

    def money(x):
        return f"${x:,.0f}" if x is not None else "—"

    def pct(x, ref):
        return f"{x / ref * 100:+.2f}%" if (x is not None and ref) else ""

    rows_open = "".join(
        f"<tr><td>{html.escape(s['underlying'])}</td><td>{html.escape(s['strategy'])}</td>"
        f"<td>{s.get('contracts', 1)}</td><td>{s['entry_date']}</td><td>{s['expiry']}</td>"
        f"<td>{money(s.get('capital'))}</td><td>{s.get('pop_pct') or '—'}</td></tr>"
        for s in sorted(open_s, key=lambda x: x["entry_date"], reverse=True))
    rows_closed = "".join(
        f"<tr><td>{html.escape(s['underlying'])}</td><td>{html.escape(s['strategy'])}</td>"
        f"<td>{s['entry_date']}</td><td>{s.get('exit_date', '—')}</td>"
        f"<td>{html.escape(str(s.get('exit_reason', '—')))[:60]}</td></tr>"
        for s in sorted(closed, key=lambda x: x.get("exit_date") or x["entry_date"], reverse=True)[:15])

    # latest VRP median if the weekly log exists
    vrp_line = ""
    vlog = BACKEND / "data_store" / "reports" / "vrp_weekly.log"
    if vlog.exists():
        med = [l for l in vlog.read_text().splitlines() if "median" in l]
        if med:
            vrp_line = f"<p class='muted'>Latest VRP check: {html.escape(med[-1].strip())}</p>"

    now = dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Paper Trading Dashboard</title><style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;color:#1a202c;background:#f7f8fa}}
h1{{font-size:1.4rem}} .muted{{color:#68748a;font-size:.85rem}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:.8rem;margin:1rem 0}}
.card{{background:#fff;border:1px solid #e2e7ef;border-radius:10px;padding: .9rem 1rem}}
.card .l{{font-size:.72rem;text-transform:uppercase;letter-spacing:.04em;color:#68748a}}
.card .v{{font-size:1.5rem;font-weight:600;font-variant-numeric:tabular-nums}}
.pos{{color:#1f7a52}} .neg{{color:#b23b3b}}
table{{border-collapse:collapse;width:100%;background:#fff;font-size:.85rem;margin:.5rem 0 1.5rem}}
th,td{{padding:.45rem .6rem;border-bottom:1px solid #edf0f5;text-align:left;font-variant-numeric:tabular-nums}}
th{{font-size:.72rem;text-transform:uppercase;color:#68748a}}
</style></head><body>
<h1>📈 Paper Trading Dashboard</h1>
<p class="muted">Updated {now} · started {base.get('start_date', '—')} with a
{money(budget)} working budget (account equity at start {money(start_eq)}) · SIMULATE account only</p>
<div class="cards">
<div class="card"><div class="l">Account equity</div><div class="v">{money(cur_eq)}</div>{spark}</div>
<div class="card"><div class="l">P&amp;L since start</div><div class="v {'pos' if (pnl or 0) >= 0 else 'neg'}">{money(pnl)} <span style="font-size:.9rem">{pct(pnl, start_eq)}</span></div></div>
<div class="card"><div class="l">Capital deployed</div><div class="v">{money(cap_in_use)}</div><div class="muted">of {money(budget)} budget</div></div>
<div class="card"><div class="l">Open structures</div><div class="v">{len(open_s)}</div></div>
<div class="card"><div class="l">Orders (sub/rej/can)</div><div class="v" style="font-size:1.1rem">{n_sub} / {n_rej} / {n_can}</div></div>
</div>
{vrp_line}
<h2 style="font-size:1.05rem">Open structures</h2>
<table><tr><th>Underlying</th><th>Strategy</th><th>Lots</th><th>Entered</th><th>Expiry</th><th>Capital</th><th>POP%</th></tr>{rows_open or '<tr><td colspan=7 class=muted>none</td></tr>'}</table>
<h2 style="font-size:1.05rem">Recently closed / cancelled</h2>
<table><tr><th>Underlying</th><th>Strategy</th><th>Entered</th><th>Exited</th><th>Reason</th></tr>{rows_closed or '<tr><td colspan=5 class=muted>none</td></tr>'}</table>
<p class="muted">Sources: data_store/paper/*.jsonl · daily notes in data_store/journal/ (open as an Obsidian vault) · private, never committed.</p>
</body></html>"""
    JOURNAL.mkdir(parents=True, exist_ok=True)
    out = JOURNAL / "dashboard.html"
    out.write_text(page, encoding="utf-8")
    return out


def snapshot_equity(equity: float, cash: float) -> None:
    """Append today's equity row (idempotent per day, last write wins)."""
    import pandas as pd
    eq_path = PAPER / "equity.csv"
    today = dt.date.today().isoformat()
    df = pd.read_csv(eq_path) if eq_path.exists() else pd.DataFrame(columns=["date", "equity", "cash"])
    df = df[df["date"] != today]
    df = pd.concat([df, pd.DataFrame([{"date": today, "equity": equity, "cash": cash}])])
    df.to_csv(eq_path, index=False)


if __name__ == "__main__":
    print(f"dashboard -> {build()}")
