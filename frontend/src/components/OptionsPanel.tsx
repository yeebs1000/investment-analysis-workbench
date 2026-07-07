import { useEffect, useState } from 'react'
import { api } from '../api'
import type { Narrative } from '../api'
import type { OptionsAnalysis, OptionStrategy } from '../types'
import { ExplainButton } from './Explain'
import { AskBox } from './AskBox'
import { fmtNum } from '../format'

const REGIME_COLOR: Record<string, string> = {
  Elevated: 'var(--warn)',
  Normal: 'var(--muted)',
  Cheap: 'var(--up)',
}

export function OptionsPanel({ code }: { code: string }) {
  const [oa, setOa] = useState<OptionsAnalysis | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const load = () => {
    setLoading(true)
    setErr(null)
    api
      .options(code, 35)
      .then(setOa)
      .catch((e) => setErr(String(e.message ?? e)))
      .finally(() => setLoading(false))
  }

  // Auto-load the (deterministic, no-LLM) options strategist whenever the ticker
  // changes, so it's part of the ticker view without an extra click.
  useEffect(() => {
    setOa(null)
    load()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [code])

  return (
    <div className="card opt-card">
      <div className="card-head">
        <h3>Options strategist</h3>
        {!oa && (
          <button onClick={load} disabled={loading}>
            {loading ? 'Loading chain…' : '⚙ Analyze options'}
          </button>
        )}
      </div>

      {err && <p className="muted small">⚠ {err}</p>}
      {oa?.error && <p className="muted small">{oa.error}</p>}

      {oa && !oa.error && (
        <>
          <div className="opt-meta">
            <span className="chip" style={{ borderColor: REGIME_COLOR[oa.iv_regime ?? 'Normal'] }}>
              IV: {oa.iv_regime ?? '—'}
            </span>
            <span className="muted small">
              ATM IV {fmtNum(oa.atm_iv_pct, 1)}% vs{' '}
              {oa.iv_regime_basis === 'garch_forecast' && oa.forecast_vol_pct != null
                ? `${fmtNum(oa.forecast_vol_pct, 1)}% GARCH forecast`
                : `${fmtNum(oa.realized_vol_pct, 1)}% realized`}
              {oa.iv_vs_realized ? ` (×${fmtNum(oa.iv_vs_realized)})` : ''}
            </span>
            <span className="muted small">
              · Tenor {oa.dte}d (exp {oa.expiry_used}) · spot {fmtNum(oa.spot)}
              {oa.holds_underlying ? ` · you hold ${fmtNum(oa.shares_held, 0)} shares` : ''}
            </span>
            {oa.days_to_earnings != null && (
              <span
                className="chip"
                style={{ borderColor: oa.days_to_earnings <= (oa.dte ?? 0) ? 'var(--warn)' : 'var(--muted)' }}
                title="Next confirmed earnings date"
              >
                Earnings {oa.days_to_earnings}d{oa.days_to_earnings <= (oa.dte ?? 0) ? ' ⚠ in tenor' : ''}
              </span>
            )}
            {oa.skew_25d_pts != null && Math.abs(oa.skew_25d_pts) >= 3 && (
              <span className="chip" title="25-delta put IV minus call IV (vol points)">
                {oa.skew_25d_pts > 0 ? 'Put' : 'Call'} skew {fmtNum(Math.abs(oa.skew_25d_pts), 1)}
              </span>
            )}
            {oa.next_atm_iv_pct != null && (
              <span className="chip" title={`Longer tenor ATM IV (${oa.next_expiry_used})`}>
                Term {fmtNum(oa.atm_iv_pct, 0)}→{fmtNum(oa.next_atm_iv_pct, 0)}%
              </span>
            )}
          </div>

          {oa.strategies.length === 0 && (
            <p className="muted small">No high-conviction structure right now — see notes below.</p>
          )}
          {oa.strategies.map((s, i) => (
            <StrategyCard key={i} s={s} />
          ))}

          <ul className="reasons" style={{ marginTop: 8 }}>
            {oa.notes.map((n, i) => (
              <li key={i} className="muted small">
                {n}
              </li>
            ))}
          </ul>

          <ExplainButton
            fetcher={(p) => api.explainOptions(code, p, 35) as Promise<Narrative>}
            label="AI analyst: options ideas"
          />

          <div className="ask-section">
            <h4>Ask about these options</h4>
            <AskBox
              fetcher={(q, p) => api.askOptions(code, q, p, 35)}
              placeholder={'e.g. "should I roll the short put?" or "what is the topside?"'}
              suggestions={['Which strategy fits a long-term holder?', 'When should I roll?', 'What is the max risk here?']}
            />
          </div>
        </>
      )}
    </div>
  )
}

function StrategyCard({ s }: { s: OptionStrategy }) {
  const credit = (s.net_debit_credit ?? 0) >= 0
  return (
    <div className="strategy">
      <div className="strategy-head">
        <b>{s.name}</b>
        <span className="chip">{s.direction}</span>
        {s.tenor_dte && <span className="muted small">{s.tenor_dte} DTE</span>}
      </div>
      <table className="legs">
        <tbody>
          {s.legs.map((l, i) => (
            <tr key={i}>
              <td className={l.action === 'Sell' ? 'sell' : 'buy'}>{l.action}</td>
              <td>
                {l.right} {fmtNum(l.strike, 0)}
              </td>
              <td className="muted small">Δ {l.delta === null ? '—' : fmtNum(l.delta, 2)}</td>
              <td className="muted small">IV {l.iv_pct === null ? '—' : `${fmtNum(l.iv_pct, 0)}%`}</td>
              <td className="num" title={l.bid !== null && l.ask !== null ? `bid ${fmtNum(l.bid)} / ask ${fmtNum(l.ask)}` : 'no live quote — last trade'}>
                ${fmtNum(l.price)}
                {l.bid !== null && l.ask !== null && (
                  <span className="muted small"> ({fmtNum(l.bid)}/{fmtNum(l.ask)})</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="econ">
        <span className={credit ? 'pos' : 'neg'}>
          {credit ? 'Credit' : 'Debit'} ${fmtNum(Math.abs(s.net_debit_credit ?? 0))}/sh
        </span>
        {s.max_profit !== null && <span>Max profit ${fmtNum(s.max_profit)}</span>}
        {s.max_loss !== null && <span>Max loss ${fmtNum(s.max_loss)}</span>}
        {s.breakeven !== null && <span>Breakeven {fmtNum(s.breakeven)}</span>}
        {s.pop_pct !== null && <span>POP {fmtNum(s.pop_pct, 0)}%</span>}
        {s.ev_per_share !== null && (
          <span
            className={s.ev_per_share >= 0 ? 'pos' : 'neg'}
            title="Probability-weighted P&L per share at expiry (same lognormal model as POP) — directional, not precise"
          >
            EV {s.ev_per_share >= 0 ? '+' : '-'}${fmtNum(Math.abs(s.ev_per_share))}/sh
          </span>
        )}
      </div>
      {(s.net_delta !== null || s.net_theta !== null || s.net_vega !== null || s.suggested_contracts !== null) && (
        <div className="opt-meta">
          {s.net_delta !== null && <span className="chip">Net Δ {fmtNum(s.net_delta, 2)}</span>}
          {s.net_theta !== null && <span className="chip">Net θ {fmtNum(s.net_theta, 2)}</span>}
          {s.net_vega !== null && <span className="chip">Net vega {fmtNum(s.net_vega, 2)}</span>}
          {s.suggested_contracts !== null && (
            <span className="chip" style={{ borderColor: 'var(--up)' }} title="Sized so worst-case loss fits ~1% of your book">
              Size {s.suggested_contracts}x
              {s.capital_required_usd !== null ? ` · ~$${fmtNum(s.capital_required_usd, 0)}` : ''}
            </span>
          )}
        </div>
      )}
      {s.warnings.length > 0 && (
        <div className="strategy-warnings">
          {s.warnings.map((w, i) => (
            <p key={i} className="warn small" style={{ color: 'var(--warn)', margin: '2px 0' }}>⚠ {w}</p>
          ))}
        </div>
      )}
      <p className="strategy-why">{s.rationale}</p>
      {s.suited_when && <p className="muted small">Best when: {s.suited_when}</p>}
      {(s.take_profit || s.stop_loss || s.manage) && (
        <div className="manage-grid">
          {s.take_profit && <div><span className="mng-label">Take profit</span> {s.take_profit}</div>}
          {s.stop_loss && <div><span className="mng-label">Stop / risk</span> {s.stop_loss}</div>}
          {s.manage && <div><span className="mng-label">Roll / manage</span> {s.manage}</div>}
        </div>
      )}
    </div>
  )
}
