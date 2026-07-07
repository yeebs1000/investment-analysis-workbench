import { useEffect, useState } from 'react'
import { api } from '../api'
import type { FundamentalMetrics } from '../types'
import { AskBox } from './AskBox'
import { fmtNum } from '../format'

const FIELD_LABELS: Record<string, string> = {
  pe_ttm: 'P/E (TTM)', pb: 'P/B', roe_pct: 'ROE', roic_pct: 'ROIC',
  gross_margin_pct: 'Gross margin', net_margin_pct: 'Net margin', operating_margin_pct: 'Operating margin',
  revenue_growth_yoy_pct: 'Revenue growth YoY', eps_growth_5y_pct: 'EPS growth (5y)',
  debt_to_equity: 'Debt / equity', current_ratio: 'Current ratio', beta: 'Beta',
  week52_high: '52w high', week52_low: '52w low',
}

export function FundamentalsPanel({ code }: { code: string }) {
  const [fm, setFm] = useState<FundamentalMetrics | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const load = () => {
    setLoading(true)
    setErr(null)
    api
      .fundamentals(code)
      .then(setFm)
      .catch((e) => setErr(String(e.message ?? e)))
      .finally(() => setLoading(false))
  }

  // Auto-load the deterministic Finnhub metrics whenever the ticker changes,
  // matching the Options panel's pattern -- the AI verdict below stays on-request.
  useEffect(() => {
    setFm(null)
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [code])

  return (
    <div className="card fund-card">
      <div className="card-head">
        <h3>Fundamentals (value-investing lens)</h3>
        {!fm && (
          <button onClick={load} disabled={loading}>
            {loading ? 'Loading…' : '📚 Load fundamentals'}
          </button>
        )}
      </div>

      {err && <p className="muted small">⚠ {err}</p>}
      {fm?.error && <p className="muted small">{fm.error}</p>}

      {fm && !fm.error && (
        <>
          <div className="opt-meta">
            {fm.sector && <span className="chip">{fm.sector}</span>}
            {fm.exchange && <span className="muted small">{fm.exchange}</span>}
            {fm.market_cap_musd != null && (
              <span className="muted small">Mkt cap ${fmtNum(fm.market_cap_musd / 1000, 1)}B</span>
            )}
          </div>

          <div className="fund-grid">
            <Metric label="P/E (TTM)" value={fm.pe_ttm} />
            <Metric label="P/B" value={fm.pb} />
            <Metric label="ROE" value={fm.roe_pct} suffix="%" />
            <Metric label="ROIC" value={fm.roic_pct} suffix="%" />
            <Metric label="Gross margin" value={fm.gross_margin_pct} suffix="%" />
            <Metric label="Net margin" value={fm.net_margin_pct} suffix="%" />
            <Metric label="Operating margin" value={fm.operating_margin_pct} suffix="%" />
            <Metric label="Revenue growth YoY" value={fm.revenue_growth_yoy_pct} suffix="%" />
            <Metric label="EPS growth (5y)" value={fm.eps_growth_5y_pct} suffix="%" />
            <Metric label="Debt / equity" value={fm.debt_to_equity} />
            <Metric label="Current ratio" value={fm.current_ratio} />
            <Metric label="Beta" value={fm.beta} />
          </div>

          {fm.missing_fields.length > 0 && (
            <p className="muted small" style={{ marginTop: 8 }}>
              Not available for this symbol: {fm.missing_fields.map((f) => FIELD_LABELS[f] ?? f).join(', ')}.
            </p>
          )}

          <div className="ask-section">
            <h4>Ask a value-investing question</h4>
            <AskBox
              fetcher={(q, p) => api.askFundamentals(code, q, p)}
              placeholder={'e.g. "is this a quality business at a fair price?"'}
              suggestions={[
                'Is this a quality business at a reasonable price?',
                'How financially strong is this company?',
                'How does growth compare to valuation?',
              ]}
            />
          </div>
        </>
      )}
    </div>
  )
}

function Metric({ label, value, suffix = '' }: { label: string; value: number | null; suffix?: string }) {
  return (
    <div className="fund-cell">
      <div className="level-label">{label}</div>
      <div className="level-value">{value === null ? '—' : `${fmtNum(value)}${suffix}`}</div>
    </div>
  )
}
