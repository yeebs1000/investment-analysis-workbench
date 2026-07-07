import { Fragment, useEffect, useMemo, useState } from 'react'
import { api } from '../api'
import type { HoldingAnalysis, Performance, PortfolioAnalysis } from '../types'
import { DecisionBadge } from './DecisionBadge'
import { AnalysisCard } from './AnalysisCard'
import { ExplainButton } from './Explain'
import { fmtMoney, fmtNum, fmtPct, plColor } from '../format'

const REFRESH_MS = 30_000 // matches the backend snapshot cache TTL

type SortKey = 'weight_desc' | 'weight_asc' | 'score_desc' | 'score_asc'
const SORTS: { value: SortKey; label: string }[] = [
  { value: 'weight_desc', label: 'Position size (high → low)' },
  { value: 'weight_asc', label: 'Position size (low → high)' },
  { value: 'score_desc', label: 'Signal score (high → low)' },
  { value: 'score_asc', label: 'Signal score (low → high)' },
]

function sortHoldings(holdings: HoldingAnalysis[], sort: SortKey): HoldingAnalysis[] {
  const sorted = [...holdings]
  switch (sort) {
    case 'weight_asc':
      return sorted.sort((a, b) => (a.weight_pct ?? 0) - (b.weight_pct ?? 0))
    case 'score_desc':
      return sorted.sort((a, b) => b.analysis.score - a.analysis.score)
    case 'score_asc':
      return sorted.sort((a, b) => a.analysis.score - b.analysis.score)
    case 'weight_desc':
    default:
      return sorted.sort((a, b) => (b.weight_pct ?? 0) - (a.weight_pct ?? 0))
  }
}

const TFS = [
  { value: 'day', label: 'Daily' },
  { value: 'week', label: 'Weekly' },
  { value: 'month', label: 'Monthly' },
  { value: '60m', label: '1 hour' },
  { value: '15m', label: '15 min' },
]

export function PortfolioView() {
  const [data, setData] = useState<PortfolioAnalysis | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState<string | null>(null)
  const [live, setLive] = useState(false)
  const [tf, setTf] = useState('day')
  const [sort, setSort] = useState<SortKey>('weight_desc')
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)

  // `silent` refreshes (auto-poll) skip the loading state so the table doesn't flash.
  const load = (silent = false, timeframe: string = tf) => {
    if (!silent) setLoading(true)
    setErr(null)
    api
      .portfolio(timeframe)
      .then((d) => {
        setData(d)
        setUpdatedAt(new Date())
      })
      .catch((e) => setErr(String(e.message ?? e)))
      .finally(() => !silent && setLoading(false))
  }
  useEffect(() => load(), [])

  useEffect(() => {
    if (!live) return
    const id = setInterval(() => load(true), REFRESH_MS)
    return () => clearInterval(id)
  }, [live, tf])

  // Hooks must run unconditionally on every render -- compute this before the
  // early returns below, even though it's only rendered once `data` exists.
  const sortedHoldings = useMemo(() => sortHoldings(data?.holdings ?? [], sort), [data, sort])

  if (err) return <Error msg={err} onRetry={() => load()} />
  if (!data) return <Loading label="Analyzing your portfolio (pulling prices + running the analyst team)…" />

  const { account, risk } = data
  return (
    <div>
      <PerformanceCard />
      <div className="summary-grid">
        <Stat
          label="Total assets"
          value={`~$${fmtNum(account.total_assets_usd ?? 0, 0)}`}
          sub={`USD, FX-approx · ${fmtMoney(account.total_assets, account.currency)} native`}
        />
        <Stat label="Invested (positions)" value={`~$${fmtNum(risk.total_value_base, 0)}`} sub="USD, FX-approx" />
        <Stat label="Positions" value={`${risk.num_positions}`} sub={`${risk.winners} up · ${risk.losers} down`} />
        <Stat
          label="Largest position"
          value={risk.concentration_pct ? `${fmtNum(risk.concentration_pct, 0)}%` : '—'}
          sub={risk.top_weights[0]?.name}
        />
      </div>

      <div className="exposure">
        <span className="muted small">Market exposure:</span>{' '}
        {Object.entries(risk.exposure_by_market).map(([m, w]) => (
          <span className="chip" key={m}>
            {m} {fmtNum(w, 0)}%
          </span>
        ))}
      </div>

      <div className="exposure">
        <span className="muted small">Fund mandate:</span>{' '}
        <span className="chip">Target {fmtNum(risk.target_return_pct, 0)}%/yr</span>
        {risk.benchmark?.return_pct != null && (
          <span className="chip" title="Benchmark trailing return over the window">
            {risk.benchmark.label} {risk.benchmark.return_pct >= 0 ? '+' : ''}{fmtNum(risk.benchmark.return_pct, 1)}%
          </span>
        )}
        {risk.avg_score != null && <span className="chip">Avg signal {fmtNum(risk.avg_score, 0)}/100</span>}
        <span className="muted small">· {data.timeframe} basis</span>
      </div>

      {data.macro_regime && (
        <div className="exposure" style={{ marginTop: 6 }}>
          <span className="muted small">Macro regime:</span>{' '}
          <span
            className="chip"
            style={{ borderColor: data.macro_regime.regime === 'risk-off' ? 'var(--down)' : data.macro_regime.regime === 'cautious' ? 'var(--warn)' : 'var(--up)' }}
            title={data.macro_regime.implication}
          >
            <b>{data.macro_regime.regime}</b>
          </span>
          {data.macro_regime.curve_10y2y != null && (
            <span className="chip" title="10yr minus 2yr Treasury yield (negative = inverted)">
              Curve {data.macro_regime.curve_10y2y >= 0 ? '+' : ''}{fmtNum(data.macro_regime.curve_10y2y, 2)}pp
            </span>
          )}
          {data.macro_regime.hy_spread_pct != null && (
            <span className="chip" title="High-yield credit spread (wider = more stress)">
              HY spread {fmtNum(data.macro_regime.hy_spread_pct, 1)}%
            </span>
          )}
          {data.macro_regime.vix != null && (
            <span className="chip" title="CBOE VIX (equity fear gauge)">VIX {fmtNum(data.macro_regime.vix, 0)}</span>
          )}
        </div>
      )}

      {risk.notes.length > 0 && (
        <div className="notes">
          {risk.notes.map((n, i) => (
            <div className="note" key={i}>
              ⚠ {n}
            </div>
          ))}
        </div>
      )}

      <ExplainButton fetcher={api.explainPortfolio} label="AI analyst: whole portfolio" />

      <div className="toolbar">
        <label className="tf-select">
          Analyse on
          <select
            value={tf}
            onChange={(e) => { setTf(e.target.value); load(false, e.target.value) }}
            disabled={loading}
          >
            {TFS.map((t) => (
              <option key={t.value} value={t.value}>{t.label}</option>
            ))}
          </select>
        </label>
        <label className="tf-select">
          Sort by
          <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
            {SORTS.map((s) => (
              <option key={s.value} value={s.value}>{s.label}</option>
            ))}
          </select>
        </label>
        <button onClick={() => load()} disabled={loading}>
          {loading ? 'Refreshing…' : '↻ Refresh'}
        </button>
        <label className="live-toggle" title={`Auto-refresh every ${REFRESH_MS / 1000}s`}>
          <input type="checkbox" checked={live} onChange={(e) => setLive(e.target.checked)} />
          {live && <span className="live-dot" />}
          Live ({REFRESH_MS / 1000}s)
        </label>
        {updatedAt && (
          <span className="muted small">Updated {updatedAt.toLocaleTimeString()}</span>
        )}
        <span className="muted small">Click any row to see the full analyst breakdown.</span>
      </div>

      <div className="row-list">
        {sortedHoldings.map((h) => {
          const p = h.position
          const expanded = open === p.code
          return (
            <Fragment key={p.code}>
              <button
                type="button"
                className={`row-item ${expanded ? 'row-open' : ''}`}
                aria-expanded={expanded}
                onClick={() => setOpen(expanded ? null : p.code)}
              >
                <div className="row-main">
                  <div className="sym">
                    {p.code} <BrokerTag broker={p.broker} />
                  </div>
                  <div className="muted small">{p.name}</div>
                </div>
                <div className="row-stats">
                  <span style={{ color: plColor(p.pl_ratio_pct) }}>{fmtPct(p.pl_ratio_pct)}</span>
                  <DecisionBadge decision={h.action} size="sm" />
                  <span className="chevron">{expanded ? '▾' : '▸'}</span>
                </div>
              </button>
              {expanded && (
                <div className="row-detail">
                  <div className="action-reason">
                    <DecisionBadge decision={h.action} size="sm" /> {h.action_reason}
                  </div>
                  <div className="chiprow">
                    <span className="chip">Weight {fmtNum(h.weight_pct ?? 0, 1)}%</span>
                    <span className="chip">Last {fmtNum(p.last_price)}</span>
                    <span className="chip" style={{ color: plColor(p.pl_value) }}>
                      P&L {fmtMoney(p.pl_value, p.currency)}
                    </span>
                    <span className="chip">Signal {fmtNum(h.analysis.score, 0)}</span>
                  </div>
                  <AnalysisCard ta={h.analysis} />
                </div>
              )}
            </Fragment>
          )
        })}
      </div>
    </div>
  )
}

const BROKER_META: Record<string, { label: string; color: string }> = {
  moomoo: { label: 'Moomoo', color: 'var(--warn)' },
  ibkr: { label: 'IBKR', color: 'var(--accent)' },
  'ibkr+moomoo': { label: 'Both', color: 'var(--up)' },
}

function BrokerTag({ broker }: { broker: string }) {
  const meta = BROKER_META[broker] ?? { label: broker, color: 'var(--muted)' }
  return (
    <span className="broker-tag" style={{ color: meta.color, borderColor: meta.color }}>
      {meta.label}
    </span>
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

// Performance vs SPY — the "am I actually beating the market?" scorecard.
// History accrues forward from first use, so it shows a building-state message
// until there's enough data for risk-adjusted ratios.
function PerformanceCard() {
  const [perf, setPerf] = useState<Performance | null>(null)
  useEffect(() => {
    api.performance().then(setPerf).catch(() => setPerf(null))
  }, [])
  if (!perf || perf.status === 'no_data') return null

  const beating = perf.beating_spy
  const excessColor = (perf.excess_return_pct ?? 0) >= 0 ? 'var(--up)' : 'var(--down)'
  return (
    <div className="perf-card">
      <div className="perf-head">
        <span className="perf-title">
          {beating ? '✅ Beating SPY' : '🔴 Trailing SPY'}
          <span className="muted small"> · since {perf.since} ({perf.days_tracked}d)</span>
        </span>
        <span className="perf-excess" style={{ color: excessColor }}>
          {(perf.excess_return_pct ?? 0) >= 0 ? '+' : ''}{fmtNum(perf.excess_return_pct, 2)}% vs SPY
        </span>
      </div>
      <div className="perf-stats">
        <span className="chip" title="Your book's cumulative return since tracking began">
          You {(perf.account_return_pct ?? 0) >= 0 ? '+' : ''}{fmtNum(perf.account_return_pct, 2)}%
        </span>
        <span className="chip" title="SPY cumulative return over the same window">
          SPY {(perf.spy_return_pct ?? 0) >= 0 ? '+' : ''}{fmtNum(perf.spy_return_pct, 2)}%
        </span>
        {perf.alpha_pct != null && (
          <span className="chip" title="Annualized alpha vs SPY (return not explained by market beta)">
            α {perf.alpha_pct >= 0 ? '+' : ''}{fmtNum(perf.alpha_pct, 1)}%
          </span>
        )}
        {perf.beta != null && <span className="chip" title="Sensitivity to SPY moves">β {fmtNum(perf.beta, 2)}</span>}
        {perf.information_ratio != null && (
          <span className="chip" title="Excess return per unit of tracking error — consistency of outperformance">
            IR {fmtNum(perf.information_ratio, 2)}
          </span>
        )}
        {perf.max_drawdown_pct != null && (
          <span className="chip" title="Worst peak-to-trough decline of your book vs SPY's">
            MaxDD {fmtNum(perf.max_drawdown_pct, 1)}% (SPY {fmtNum(perf.spy_max_drawdown_pct, 1)}%)
          </span>
        )}
      </div>
      {perf.message && <p className="muted small" style={{ margin: '4px 0 0' }}>{perf.message}</p>}
    </div>
  )
}

export function Loading({ label }: { label: string }) {
  return <div className="loading">⏳ {label}</div>
}

export function Error({ msg, onRetry }: { msg: string; onRetry: () => void }) {
  return (
    <div className="error">
      <p>⚠ {msg}</p>
      <p className="muted small">
        Make sure the Moomoo OpenD gateway is running and logged in, then retry.
      </p>
      <button onClick={onRetry}>Retry</button>
    </div>
  )
}
