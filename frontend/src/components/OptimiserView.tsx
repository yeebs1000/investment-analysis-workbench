import { Fragment, useEffect, useState } from 'react'
import { api } from '../api'
import type { OptimizerPlan, OptimizerAction } from '../types'
import { Loading, Error as ErrorBox } from './PortfolioView'
import { fmtNum } from '../format'

const ACTION_COLOR: Record<string, string> = {
  BUY: 'var(--up)',
  ADD: 'var(--up)',
  HOLD: 'var(--muted)',
  TRIM: 'var(--warn)',
  SELL: 'var(--down)',
}

const TFS = [
  { value: 'day', label: 'Daily' },
  { value: 'week', label: 'Weekly' },
  { value: 'month', label: 'Monthly' },
  { value: '60m', label: '1 hour' },
  { value: '15m', label: '15 min' },
]

const METHODS: { value: 'heuristic' | 'risk_aware'; label: string }[] = [
  { value: 'heuristic', label: 'Score-based (current)' },
  { value: 'risk_aware', label: 'Risk-aware (beta)' },
]

// Single-name concentration cap. Higher caps accommodate deliberate long-term
// core overweights; the plan still flags everything above the chosen cap.
const CAPS = [10, 15, 20, 25, 30]

export function OptimiserView() {
  const [plan, setPlan] = useState<OptimizerPlan | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [tf, setTf] = useState('day')
  const [method, setMethod] = useState<'heuristic' | 'risk_aware'>('heuristic')
  const [capPct, setCapPct] = useState(15)
  const [open, setOpen] = useState<string | null>(null)

  const load = (timeframe: string = tf, m: 'heuristic' | 'risk_aware' = method, cap: number = capPct) => {
    setLoading(true)
    setErr(null)
    api
      .optimize(timeframe, m, cap)
      .then(setPlan)
      .catch((e) => setErr(String(e.message ?? e)))
      .finally(() => setLoading(false))
  }
  useEffect(() => load(), [])

  if (err) return <ErrorBox msg={err} onRetry={load} />
  if (!plan) return <Loading label="Building your optimised action plan…" />

  const cap = plan.concentration_cap_pct
  return (
    <div>
      <div className="summary-grid">
        <Stat label="Total value" value={`~$${fmtNum(plan.total_value_usd, 0)}`} sub="USD, FX-approx" />
        <Stat label="To buy" value={`$${fmtNum(plan.buy_usd, 0)}`} sub={`${plan.actions.filter((a) => a.action === 'BUY' || a.action === 'ADD').length} names`} />
        <Stat label="To sell / trim" value={`$${fmtNum(plan.sell_usd, 0)}`} sub={`${plan.actions.filter((a) => a.action === 'SELL' || a.action === 'TRIM').length} names`} />
        <Stat label="Projected top weight" value={plan.projected_top_pct != null ? `${fmtNum(plan.projected_top_pct, 1)}%` : '—'} sub={`cap ${fmtNum(cap, 0)}%`} />
      </div>

      {(plan.notes.length > 0 || plan.risk_notes.length > 0) && (
        <div className="notes">
          {plan.notes.map((n, i) => (
            <div className="note" key={`n${i}`}>
              {n}
            </div>
          ))}
          {plan.risk_notes.map((n, i) => (
            <div className="note" key={`r${i}`}>
              {n}
            </div>
          ))}
        </div>
      )}

      <div className="toolbar">
        <label className="tf-select">
          Horizon
          <select
            value={tf}
            onChange={(e) => { setTf(e.target.value); load(e.target.value) }}
            disabled={loading}
          >
            {TFS.map((t) => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </select>
        </label>
        <label className="tf-select">
          Model
          <select
            value={method}
            onChange={(e) => { const m = e.target.value as 'heuristic' | 'risk_aware'; setMethod(m); load(tf, m) }}
            disabled={loading}
          >
            {METHODS.map((m) => (
              <option key={m.value} value={m.value}>{m.label}</option>
            ))}
          </select>
        </label>
        <label className="tf-select">
          Max / name
          <select
            value={capPct}
            onChange={(e) => { const c = Number(e.target.value); setCapPct(c); load(tf, method, c) }}
            disabled={loading}
          >
            {CAPS.map((c) => (
              <option key={c} value={c}>{c}%</option>
            ))}
          </select>
        </label>
        <button onClick={() => load()} disabled={loading}>
          {loading ? 'Recomputing…' : '↻ Recompute'}
        </button>
        {plan.method_used === 'risk_aware' ? (
          <span className="muted small">
            Risk-aware: score-tilt weights scaled by inverse volatility + correlation to the rest
            of the book{plan.portfolio_vol_pct != null ? `, portfolio vol ~${fmtNum(plan.portfolio_vol_pct, 1)}%/yr` : ''}
            {plan.covariance_shrinkage != null ? ` (shrinkage ${fmtNum(plan.covariance_shrinkage * 100, 0)}%)` : ''}.
            Cap {fmtNum(cap, 0)}%, ~{fmtNum(plan.cash_target_pct, 0)}% cash. Decision-support only — you place any trades yourself.
          </span>
        ) : (
          <span className="muted small">
            Model: tilt toward higher-scoring names, exit weak signals, cap any single name at {fmtNum(cap, 0)}%, keep
            ~{fmtNum(plan.cash_target_pct, 0)}% cash. Decision-support only — you place any trades yourself.
          </span>
        )}
      </div>

      <div className="row-list">
        {plan.actions.map((a) => {
          const expanded = open === a.code
          return (
            <Fragment key={a.code}>
              <RowItem a={a} expanded={expanded} onToggle={() => setOpen(expanded ? null : a.code)} />
              {expanded && (
                <div className="row-detail">
                  <div className="chiprow">
                    <span className="chip">Score {fmtNum(a.score, 0)}</span>
                    <span className="chip">{fmtNum(a.current_pct, 1)}% → {fmtNum(a.target_pct, 1)}%</span>
                    <span className="chip">
                      ~Shares {a.est_shares == null || a.action === 'HOLD' ? '—' : `${a.delta_usd > 0 ? '+' : '-'}${fmtNum(Math.abs(a.est_shares), 0)}`}
                    </span>
                    {a.last_price != null && <span className="chip">Last {fmtNum(a.last_price)}</span>}
                    {a.risk_contribution_pct != null && <span className="chip">Risk share {fmtNum(a.risk_contribution_pct, 0)}%</span>}
                  </div>
                  <p className="muted small" style={{ margin: 0 }}>{a.reason}</p>
                </div>
              )}
            </Fragment>
          )
        })}
      </div>
    </div>
  )
}

function RowItem({ a, expanded, onToggle }: { a: OptimizerAction; expanded: boolean; onToggle: () => void }) {
  const buy = a.delta_usd > 0
  return (
    <button
      type="button"
      className={`row-item ${expanded ? 'row-open' : ''}`}
      aria-expanded={expanded}
      onClick={onToggle}
    >
      <div className="row-main">
        <div className="sym">{a.code}</div>
        <div className="muted small">{a.name}</div>
      </div>
      <div className="row-stats">
        <span style={{ color: a.action === 'HOLD' ? 'var(--muted)' : buy ? 'var(--up)' : 'var(--down)' }}>
          {a.action === 'HOLD' ? '—' : `${buy ? '+' : '-'}$${fmtNum(Math.abs(a.delta_usd), 0)}`}
        </span>
        <span className="badge badge-sm" style={{ background: ACTION_COLOR[a.action] ?? 'var(--muted)' }}>
          {a.action}
        </span>
        <span className="chevron">{expanded ? '▾' : '▸'}</span>
      </div>
    </button>
  )
}

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="stat">
      <div className="stat-label">{label}</div>
      <div className="stat-value">{value}</div>
      {sub && <div className="muted small">{sub}</div>}
    </div>
  )
}
