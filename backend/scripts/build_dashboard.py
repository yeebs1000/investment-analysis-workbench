"""Build the daily paper-trading dashboard (self-contained dark HTML, private).

Reads baseline.json, equity.csv, structures.jsonl, order_ledger.jsonl, an
optional marks.json (live per-position P&L written by a scan), and the latest
VRP reading; writes data_store/journal/dashboard.html. Called at the end of
every paper_trade.py run and runnable standalone:

    PYTHONPATH=. .venv/Scripts/python.exe scripts/build_dashboard.py
"""
from __future__ import annotations

import datetime as dt
import html
import json
import math
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND))
PAPER = BACKEND / "data_store" / "paper"
JOURNAL = BACKEND / "data_store" / "journal"

# --- palette (dark trading terminal; committed single theme) -----------------
BG, PANEL, PANEL2 = "#0c0e13", "#14171e", "#1b1f28"
LINE = "#252b36"
INK, MUTED, FAINT = "#e7eaf1", "#8b94a5", "#59616f"
POS, NEG, ACCENT = "#43b581", "#e0575d", "#4fc9c2"
POS_DIM, NEG_DIM = "#1c3a30", "#3a2126"


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def area_chart(dates: list[str], ys: list[float], ref: float | None,
               w: int = 940, h: int = 260) -> str:
    """Equity curve: gradient area, gridlines, baseline ref, endpoint pulse."""
    if len(ys) < 2:
        return (f'<div class="chart-empty">Equity curve builds from the second '
                f'daily snapshot onward.</div>')
    pl, pr, pt, pb = 58, 14, 16, 30
    lo = min(min(ys), ref or min(ys)); hi = max(max(ys), ref or max(ys))
    span = (hi - lo) or 1.0
    lo -= span * 0.12; hi += span * 0.12; span = hi - lo
    px = lambda i: pl + i * (w - pl - pr) / (len(ys) - 1)
    py = lambda v: pt + (hi - v) / span * (h - pt - pb)
    pts = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(ys))
    up = ys[-1] >= (ref if ref else ys[0])
    col = POS if up else NEG
    area = f"{px(0):.1f},{h-pb} {pts} {px(len(ys)-1):.1f},{h-pb}"
    grid = lab = ""
    for k in range(5):
        v = lo + span * k / 4; y = py(v)
        grid += f'<line x1="{pl}" y1="{y:.1f}" x2="{w-pr}" y2="{y:.1f}" stroke="{LINE}" stroke-width="1"/>'
        lab += (f'<text x="{pl-10}" y="{y+3.5:.1f}" text-anchor="end" class="axis">'
                f'{v/1000:,.1f}k</text>')
    step = max(1, len(dates) // 7)
    for i in range(0, len(dates), step):
        lab += f'<text x="{px(i):.1f}" y="{h-9}" text-anchor="middle" class="axis">{dates[i][5:]}</text>'
    refline = ""
    if ref is not None:
        refline = (f'<line x1="{pl}" y1="{py(ref):.1f}" x2="{w-pr}" y2="{py(ref):.1f}" '
                   f'stroke="{FAINT}" stroke-width="1" stroke-dasharray="3 4"/>'
                   f'<text x="{w-pr}" y="{py(ref)-6:.1f}" text-anchor="end" class="axis">start</text>')
    ex, ey = px(len(ys)-1), py(ys[-1])
    length = 2000
    return f"""<svg viewBox="0 0 {w} {h}" class="equity" preserveAspectRatio="none" role="img" aria-label="equity curve">
<defs><linearGradient id="eg" x1="0" y1="0" x2="0" y2="1">
<stop offset="0" stop-color="{col}" stop-opacity="0.22"/>
<stop offset="1" stop-color="{col}" stop-opacity="0"/></linearGradient></defs>
{grid}{refline}
<polygon points="{area}" fill="url(#eg)"/>
<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2" stroke-linejoin="round"
 stroke-linecap="round" class="eqline" style="stroke-dasharray:{length};stroke-dashoffset:{length}"/>
<circle cx="{ex:.1f}" cy="{ey:.1f}" r="4.5" fill="{col}" class="pulse"/>
<circle cx="{ex:.1f}" cy="{ey:.1f}" r="3" fill="{BG}"/><circle cx="{ex:.1f}" cy="{ey:.1f}" r="1.8" fill="{col}"/>
{lab}</svg>"""


def alloc_bars(pairs: list[tuple[str, float]]) -> str:
    if not pairs:
        return '<div class="chart-empty">No positions.</div>'
    mx = max(v for _, v in pairs) or 1.0
    rows = ""
    for lab, v in pairs:
        w = v / mx * 100
        rows += (f'<div class="abar"><span class="abar-l">{html.escape(lab)}</span>'
                 f'<span class="abar-t"><span class="abar-f" style="width:{w:.1f}%"></span></span>'
                 f'<span class="abar-v">${v:,.0f}</span></div>')
    return f'<div class="alloc">{rows}</div>'


def gauge(frac: float, sub: str) -> str:
    frac = max(0.0, min(1.0, frac))
    r, cx, cy, w = 62, 84, 78, 168
    a0 = math.pi; a1 = math.pi * (1 - frac)
    x0, y0 = cx + r*math.cos(a0), cy - r*math.sin(a0)
    x1, y1 = cx + r*math.cos(a1), cy - r*math.sin(a1)
    col = ACCENT if frac < 0.8 else NEG
    return (f'<svg viewBox="0 0 {w} 96" class="gauge">'
            f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 0 1 {cx+r} {cy}" stroke="{LINE}" stroke-width="9" fill="none" stroke-linecap="round"/>'
            f'<path d="M {x0:.1f} {y0:.1f} A {r} {r} 0 {1 if frac>0.5 else 0} 1 {x1:.1f} {y1:.1f}" stroke="{col}" stroke-width="9" fill="none" stroke-linecap="round"/>'
            f'<text x="{cx}" y="{cy-6}" text-anchor="middle" class="g-big">{frac*100:.0f}%</text>'
            f'<text x="{cx}" y="{cy+12}" text-anchor="middle" class="g-sub">{html.escape(sub)}</text></svg>')


def pnl_bar(v: float, mx: float) -> str:
    """Centered diverging bar for a table cell: green right, red left."""
    if not mx:
        return ""
    frac = max(-1.0, min(1.0, v / mx))
    if frac >= 0:
        return f'<span class="pb"><span class="pb-pos" style="width:{frac*50:.1f}%;left:50%"></span></span>'
    return f'<span class="pb"><span class="pb-neg" style="width:{-frac*50:.1f}%;right:50%"></span></span>'


def build() -> Path:
    import pandas as pd

    base = json.loads((PAPER / "baseline.json").read_text()) if (PAPER / "baseline.json").exists() else {}
    start_eq = base.get("account_equity_at_start")
    budget = base.get("working_budget_usd")
    marks = json.loads((PAPER / "marks.json").read_text()) if (PAPER / "marks.json").exists() else {}

    eq_path = PAPER / "equity.csv"
    eq = pd.read_csv(eq_path) if eq_path.exists() else pd.DataFrame(columns=["date", "equity", "cash"])
    cur_eq = float(eq["equity"].iloc[-1]) if len(eq) else start_eq
    cash = float(eq["cash"].iloc[-1]) if len(eq) and "cash" in eq else None
    pnl = (cur_eq - start_eq) if (cur_eq is not None and start_eq) else None
    day_pnl = (float(eq["equity"].iloc[-1]) - float(eq["equity"].iloc[-2])) if len(eq) >= 2 else None
    open_mark = marks.get("_total_open_pnl")

    structs = load_jsonl(PAPER / "structures.jsonl")
    open_s = [s for s in structs if s.get("status") in ("OPEN", "CLOSING")]
    closed = [s for s in structs if s.get("status") not in ("OPEN", "CLOSING")]
    cap_in_use = sum(s.get("capital", 0) or 0 for s in open_s)

    orders = load_jsonl(PAPER / "order_ledger.jsonl")
    n_sub = sum(1 for o in orders if o.get("status") == "SUBMITTED")

    equity_chart = area_chart(eq["date"].tolist(), eq["equity"].astype(float).tolist(), ref=start_eq)
    mix = {}
    for s in open_s:
        mix[s["strategy"]] = mix.get(s["strategy"], 0) + (s.get("capital") or 0)
    alloc = alloc_bars(sorted(mix.items(), key=lambda kv: -kv[1]))
    deploy = gauge((cap_in_use / budget) if budget else 0.0, "of budget")

    def money(x, dp=0):
        return f"${x:,.{dp}f}" if x is not None else "—"

    # positions table, sorted by live P&L when marks exist else by entry date
    def spnl(s):
        m = marks.get(s["id"], {})
        return m.get("pnl")
    have_marks = any(spnl(s) is not None for s in open_s)
    ordered = sorted(open_s, key=lambda s: (spnl(s) if spnl(s) is not None else -1e12), reverse=True) \
        if have_marks else sorted(open_s, key=lambda s: s["entry_date"], reverse=True)
    mx_pnl = max((abs(spnl(s)) for s in open_s if spnl(s) is not None), default=0.0)
    today = dt.date.today()
    rows_open = ""
    for s in ordered:
        m = marks.get(s["id"], {})
        p = m.get("pnl")
        try:
            dte = (dt.date.fromisoformat(s["expiry"]) - today).days
        except Exception:  # noqa: BLE001
            dte = "—"
        cls = "pos" if (p or 0) > 0 else "neg" if (p or 0) < 0 else "flat"
        pcell = f'<span class="{cls}">{("+" if (p or 0)>=0 else "")}{p:,.0f}</span>' if p is not None else '<span class="flat">—</span>'
        entry = m.get("entry"); mark = m.get("mark")
        emcell = (f'{entry:.2f}<span class="arw">→</span>{mark:.2f}'
                  if entry is not None and mark is not None else "—")
        rows_open += (
            f'<tr><td class="u">{html.escape(s["underlying"])}</td>'
            f'<td><span class="chip">{html.escape(s["strategy"])}</span></td>'
            f'<td class="n">{s.get("contracts",1)}</td>'
            f'<td class="n dte">{dte}</td>'
            f'<td class="n mono">{emcell}</td>'
            f'<td class="n mono {cls}">{pcell}</td>'
            f'<td class="pbar">{pnl_bar(p, mx_pnl) if p is not None else ""}</td>'
            f'<td class="n mono">{money(s.get("capital"))}</td>'
            f'<td class="n">{s.get("pop_pct") or "—"}</td></tr>')
    if not rows_open:
        rows_open = '<tr><td colspan="9" class="empty">No open structures.</td></tr>'

    rows_closed = ""
    for s in sorted(closed, key=lambda x: x.get("exit_date") or x["entry_date"], reverse=True)[:10]:
        st = s.get("status", "")
        badge = {"CLOSED": "closed", "CANCELLED_UNFILLED": "cancel",
                 "ABORTED_ENTRY": "cancel", "CLOSING": "closing"}.get(st, "cancel")
        rows_closed += (
            f'<tr><td class="u">{html.escape(s["underlying"])}</td>'
            f'<td><span class="chip">{html.escape(s["strategy"])}</span></td>'
            f'<td><span class="badge {badge}">{html.escape(st.replace("_"," ").title())}</span></td>'
            f'<td class="rsn">{html.escape(str(s.get("exit_reason","—")))[:52]}</td></tr>')
    if not rows_closed:
        rows_closed = '<tr><td colspan="4" class="empty">Nothing closed yet.</td></tr>'

    vrp = ""
    vlog = BACKEND / "data_store" / "reports" / "vrp_weekly.log"
    if vlog.exists():
        med = [l for l in vlog.read_text().splitlines() if "median" in l]
        if med:
            vrp = html.escape(med[-1].strip())

    pct = f'{pnl/start_eq*100:+.2f}%' if (pnl is not None and start_eq) else ''
    daycls = "pos" if (day_pnl or 0) >= 0 else "neg"
    totalcls = "pos" if (pnl or 0) >= 0 else "neg"
    omcls = "pos" if (open_mark or 0) >= 0 else "neg"
    now = dt.datetime.now().strftime("%a %d %b · %H:%M")

    head = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Paper Book · {money(cur_eq)}</title><style>{CSS}</style></head><body>"""

    body = f"""
<header class="bar">
  <div class="brand"><span class="dot"></span><b>Paper Book</b>
    <span class="env">SIMULATE</span></div>
  <div class="bar-eq">
    <span class="mono eq">{money(cur_eq)}</span>
    <span class="mono {totalcls} delta">{('+' if (pnl or 0)>=0 else '')}{money(pnl)} · {pct}</span>
  </div>
  <div class="upd">{now}</div>
</header>

<main>
  <section class="kpis">
    <div class="kpi"><span class="k">Day change</span>
      <span class="mono v {daycls}">{('+' if (day_pnl or 0)>=0 else '')}{money(day_pnl)}</span></div>
    <div class="kpi"><span class="k">Open P&amp;L (mark)</span>
      <span class="mono v {omcls}">{('+' if (open_mark or 0)>=0 else '')}{money(open_mark) if open_mark is not None else '—'}</span></div>
    <div class="kpi"><span class="k">Deployed</span>
      <span class="mono v">{money(cap_in_use)}</span><span class="sub">of {money(budget)}</span></div>
    <div class="kpi"><span class="k">Open structures</span>
      <span class="mono v">{len(open_s)}</span><span class="sub">{n_sub} orders</span></div>
    <div class="kpi"><span class="k">Cash</span>
      <span class="mono v">{money(cash)}</span></div>
  </section>

  <section class="panel chart">
    <div class="phead"><h2>Equity</h2><span class="note">since {base.get('start_date','—')} · dashed = start</span></div>
    {equity_chart}
  </section>

  <section class="split">
    <div class="panel">
      <div class="phead"><h2>Open positions</h2><span class="note">{len(open_s)} · ranked by P&amp;L</span></div>
      <div class="tbl-wrap"><table class="pos-tbl">
        <thead><tr><th>Symbol</th><th>Structure</th><th class="n">Lots</th><th class="n">DTE</th>
          <th class="n">Entry→Mark</th><th class="n">P&amp;L</th><th></th><th class="n">Capital</th><th class="n">POP</th></tr></thead>
        <tbody>{rows_open}</tbody>
      </table></div>
    </div>
    <aside class="rail">
      <div class="panel"><div class="phead"><h2>Budget</h2></div>{deploy}</div>
      <div class="panel"><div class="phead"><h2>Allocation</h2><span class="note">capital by structure</span></div>{alloc}</div>
      {f'<div class="panel vrp"><div class="phead"><h2>Volatility premium</h2></div><p class="mono vrpv">{vrp}</p><p class="note">breakeven at 1.25 · assumption 1.05</p></div>' if vrp else ''}
    </aside>
  </section>

  <section class="panel">
    <div class="phead"><h2>Recently closed</h2></div>
    <div class="tbl-wrap"><table class="pos-tbl closed">
      <thead><tr><th>Symbol</th><th>Structure</th><th>Status</th><th>Reason</th></tr></thead>
      <tbody>{rows_closed}</tbody></table></div>
  </section>

  <footer class="foot">Private · SIMULATE account only · ledgers in <code>data_store/paper/</code> · never committed</footer>
</main>
</body></html>"""

    JOURNAL.mkdir(parents=True, exist_ok=True)
    out = JOURNAL / "dashboard.html"
    out.write_text(head + body, encoding="utf-8")
    return out


def write_marks(broker) -> None:
    """Live per-position P&L from a fresh OpenD snapshot -> marks.json, so the
    dashboard shows mark-to-market P&L (build() is otherwise offline). Best
    effort; a failure leaves the dashboard on capital/POP only."""
    try:
        from app.services.analysis_service import service
        structs = load_jsonl(PAPER / "structures.jsonl")
        opn = [s for s in structs if s.get("status") in ("OPEN", "CLOSING")]
        codes = list({l["code"] for s in opn for l in s["legs"]})
        if not codes:
            return
        with service._lock:
            snap = service._client.get_snapshot(codes).set_index("code")
        out, total = {}, 0.0
        for s in opn:
            nm = ne = 0.0; ok = True
            for l in s["legs"]:
                c = l["code"]; sg = 1 if l["side"] == "BUY" else -1
                if c not in snap.index:
                    ok = False; break
                bid = float(snap.loc[c, "bid_price"]); ask = float(snap.loc[c, "ask_price"])
                mid = (bid + ask) / 2 if bid > 0 and ask > 0 else float(snap.loc[c, "last_price"])
                nm += sg * mid; ne += sg * (l.get("fill_price") or l["entry_mid"])
            if not ok:
                continue
            q = s["legs"][0]["qty"]
            pnl = (nm - ne) * 100 * q
            out[s["id"]] = {"pnl": round(pnl, 0), "entry": round(ne, 2), "mark": round(nm, 2)}
            total += pnl
        out["_total_open_pnl"] = round(total, 0)
        (PAPER / "marks.json").write_text(json.dumps(out), encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass


def snapshot_equity(equity: float, cash: float) -> None:
    """Append today's equity row (idempotent per day, last write wins)."""
    import pandas as pd
    eq_path = PAPER / "equity.csv"
    today = dt.date.today().isoformat()
    df = pd.read_csv(eq_path) if eq_path.exists() else pd.DataFrame(columns=["date", "equity", "cash"])
    df = df[df["date"] != today]
    df = pd.concat([df, pd.DataFrame([{"date": today, "equity": equity, "cash": cash}])])
    df.to_csv(eq_path, index=False)


CSS = """
*{box-sizing:border-box;margin:0}
:root{color-scheme:dark}
body{background:%(BG)s;color:%(INK)s;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Inter,system-ui,sans-serif;
  font-size:14px;line-height:1.45;-webkit-font-smoothing:antialiased;padding-bottom:3rem}
.mono{font-family:ui-monospace,"SF Mono","Cascadia Code","Roboto Mono",Consolas,monospace;font-variant-numeric:tabular-nums}
h2{font-size:.8rem;font-weight:600;letter-spacing:.01em;color:%(INK)s}
main{max-width:1180px;margin:0 auto;padding:1.25rem 1.25rem 0}

/* top bar */
.bar{position:sticky;top:0;z-index:20;display:flex;align-items:center;gap:1.25rem;
  padding:.7rem 1.25rem;background:rgba(12,14,19,.82);backdrop-filter:blur(10px);
  border-bottom:1px solid %(LINE)s}
.brand{display:flex;align-items:center;gap:.5rem;font-size:.95rem}
.brand .dot{width:8px;height:8px;border-radius:50%%;background:%(ACCENT)s;box-shadow:0 0 8px %(ACCENT)s}
.env{font-size:.62rem;letter-spacing:.08em;color:%(ACCENT)s;border:1px solid %(LINE)s;
  padding:.12rem .4rem;border-radius:4px;margin-left:.15rem}
.bar-eq{margin-left:auto;display:flex;align-items:baseline;gap:.7rem}
.bar-eq .eq{font-size:1.15rem;font-weight:600}
.bar-eq .delta{font-size:.82rem}
.upd{color:%(FAINT)s;font-size:.72rem;min-width:9ch;text-align:right}

/* kpi strip */
.kpis{display:grid;grid-template-columns:repeat(5,1fr);gap:1px;background:%(LINE)s;
  border:1px solid %(LINE)s;border-radius:12px;overflow:hidden;margin:1.1rem 0}
.kpi{background:%(PANEL)s;padding:.85rem 1rem;display:flex;flex-direction:column;gap:.28rem}
.kpi .k{font-size:.68rem;letter-spacing:.03em;color:%(MUTED)s;text-transform:uppercase}
.kpi .v{font-size:1.35rem;font-weight:600;line-height:1}
.kpi .sub{font-size:.7rem;color:%(FAINT)s}

.pos{color:%(POS)s}.neg{color:%(NEG)s}.flat{color:%(MUTED)s}

/* panels */
.panel{background:%(PANEL)s;border:1px solid %(LINE)s;border-radius:12px;padding:1rem 1.1rem;margin-bottom:1.1rem}
.phead{display:flex;align-items:baseline;gap:.6rem;margin-bottom:.7rem}
.phead .note{font-size:.72rem;color:%(FAINT)s;margin-left:auto}
.chart{padding-bottom:.4rem}
.equity{width:100%%;height:260px;display:block}
.axis{fill:%(FAINT)s;font-size:10px;font-family:ui-monospace,monospace}
.chart-empty,.empty{color:%(FAINT)s;font-size:.82rem;padding:1.5rem 0;text-align:center}
.eqline{animation:draw 1.1s cubic-bezier(.2,.7,.2,1) forwards}
.pulse{animation:pulse 2.4s ease-out infinite}
@keyframes draw{to{stroke-dashoffset:0}}
@keyframes pulse{0%%{opacity:.5}50%%{opacity:.12}100%%{opacity:.5}}

/* layout split */
.split{display:grid;grid-template-columns:1fr 300px;gap:1.1rem;align-items:start}
.rail .panel{margin-bottom:1.1rem}

/* tables */
.tbl-wrap{overflow-x:auto;margin:0 -.3rem}
.pos-tbl{width:100%%;border-collapse:collapse;font-size:.82rem}
.pos-tbl th{font-size:.66rem;letter-spacing:.04em;text-transform:uppercase;color:%(MUTED)s;
  font-weight:600;text-align:left;padding:.4rem .55rem;border-bottom:1px solid %(LINE)s}
.pos-tbl td{padding:.5rem .55rem;border-bottom:1px solid rgba(37,43,54,.5)}
.pos-tbl tbody tr{transition:background .12s ease}
.pos-tbl tbody tr:hover{background:%(PANEL2)s}
.pos-tbl .n{text-align:right}
.pos-tbl .u{font-weight:600;letter-spacing:.01em}
.chip{font-size:.68rem;color:%(MUTED)s;background:%(PANEL2)s;border:1px solid %(LINE)s;
  padding:.12rem .45rem;border-radius:5px;white-space:nowrap}
.dte{color:%(MUTED)s}
.arw{color:%(FAINT)s;padding:0 .3rem}
.pbar{width:74px;padding:0 .3rem!important}
.pb{position:relative;display:block;height:6px;width:70px;background:%(PANEL2)s;border-radius:3px}
.pb-pos{position:absolute;height:6px;background:%(POS)s;border-radius:3px}
.pb-neg{position:absolute;height:6px;background:%(NEG)s;border-radius:3px}
.badge{font-size:.66rem;padding:.12rem .45rem;border-radius:5px;letter-spacing:.02em}
.badge.closed{color:%(POS)s;background:%(POS_DIM)s}
.badge.cancel{color:%(MUTED)s;background:%(PANEL2)s}
.badge.closing{color:%(ACCENT)s;background:rgba(79,201,194,.12)}
.rsn{color:%(MUTED)s;font-size:.76rem}

/* allocation bars */
.alloc{display:flex;flex-direction:column;gap:.55rem}
.abar{display:grid;grid-template-columns:1fr 90px auto;align-items:center;gap:.6rem;font-size:.76rem}
.abar-l{color:%(MUTED)s;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.abar-t{height:7px;background:%(PANEL2)s;border-radius:4px;overflow:hidden}
.abar-f{display:block;height:7px;background:%(ACCENT)s;border-radius:4px}
.abar-v{font-family:ui-monospace,monospace;color:%(INK)s;text-align:right}

/* gauge / vrp */
.gauge{width:168px;height:96px;display:block;margin:0 auto}
.g-big{fill:%(INK)s;font-size:22px;font-weight:600;font-family:ui-monospace,monospace}
.g-sub{fill:%(FAINT)s;font-size:9px}
.vrpv{font-size:.8rem;color:%(INK)s}
.vrp .note{color:%(FAINT)s;font-size:.7rem;margin-top:.3rem}

.foot{color:%(FAINT)s;font-size:.72rem;text-align:center;padding:1.5rem 0 0}
.foot code{color:%(MUTED)s}

@media (max-width:820px){
  .kpis{grid-template-columns:repeat(2,1fr)}
  .split{grid-template-columns:1fr}
  .bar{flex-wrap:wrap;gap:.6rem}.upd{display:none}
}
@media (prefers-reduced-motion:reduce){
  .eqline{animation:none;stroke-dashoffset:0}.pulse{animation:none}
}
""" % {"BG": BG, "PANEL": PANEL, "PANEL2": PANEL2, "LINE": LINE, "INK": INK,
       "MUTED": MUTED, "FAINT": FAINT, "POS": POS, "NEG": NEG, "ACCENT": ACCENT,
       "POS_DIM": POS_DIM, "NEG_DIM": NEG_DIM}


if __name__ == "__main__":
    print(f"dashboard -> {build()}")
