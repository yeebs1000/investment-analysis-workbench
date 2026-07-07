import type { TechnicalAnalysis } from '../types'
import { DecisionBadge } from './DecisionBadge'
import { ScoreMeter } from './ScoreMeter'
import { AnalystBreakdown } from './AnalystBreakdown'
import { PriceChart } from './PriceChart'
import { ExplainButton } from './Explain'
import { AskBox } from './AskBox'
import { TradePanel } from './TradePanel'
import { api } from '../api'
import { DECISION_META, fmtNum } from '../format'

function consensusColor(score: number): string {
  return score >= 0.15 ? 'var(--up)' : score <= -0.15 ? 'var(--down)' : 'var(--muted)'
}

const TF_LABEL: Record<string, string> = {
  day: 'Daily', week: 'Weekly', month: 'Monthly',
  '60m': '1-hour', '30m': '30-min', '15m': '15-min', '5m': '5-min',
}

function htfColor(trend: number | null): string {
  if (trend === null) return 'var(--muted)'
  return trend > 0.15 ? 'var(--up)' : trend < -0.15 ? 'var(--down)' : 'var(--muted)'
}

export function AnalysisCard({ ta, tf = 'day' }: { ta: TechnicalAnalysis; tf?: string }) {
  if (ta.error) {
    return (
      <div className="card">
        <div className="card-head">
          <div>
            <span className="sym">{ta.code}</span> <span className="name">{ta.name}</span>
          </div>
        </div>
        <p className="muted">{ta.error}</p>
      </div>
    )
  }

  return (
    <div className="card">
      <div className="card-head">
        <div>
          <span className="sym">{ta.code}</span> <span className="name">{ta.name}</span>
          <div className="muted small">
            {ta.price !== null ? `Last ${fmtNum(ta.price)} ${ta.currency ?? ''}` : ''}
            {ta.as_of ? ` · as of ${ta.as_of}` : ''} · {ta.bars_used} bars
          </div>
          <div className="tf-line small">
            <span className="tf-badge">{TF_LABEL[ta.timeframe] ?? ta.timeframe} analysis</span>
            {ta.higher_tf && (
              <span className="muted">
                {' '}higher timeframe ({ta.higher_tf}):{' '}
                <b style={{ color: htfColor(ta.higher_tf_trend) }}>{ta.higher_tf_summary}</b>
              </span>
            )}
          </div>
        </div>
        <DecisionBadge decision={ta.decision} size="lg" />
      </div>

      <ScoreMeter score={ta.score} confidence={ta.confidence_label} />
      <p className="verdict">{DECISION_META[ta.decision].plain}</p>
      {ta.verdict && (
        <p className="verdict small" title={`Two independent lenses, never blended: business quality (${ta.verdict.quality_axis}) × entry timing (${ta.verdict.timing_axis})`}>
          <b>{ta.verdict.quadrant}:</b> {ta.verdict.guidance}
        </p>
      )}
      {ta.entry_risk && (
        <p
          className="verdict small"
          style={{ color: ta.entry_risk.level === 'high' ? 'var(--down)' : 'var(--muted)' }}
          title={ta.entry_risk.reasons.join('; ')}
        >
          ⚠ <b>{ta.entry_risk.label}</b> — {ta.entry_risk.advice}
        </p>
      )}

      <ExplainButton
        key={`${ta.code}-${tf}`}
        fetcher={(p) => api.explainSymbol(ta.code, p, tf)}
        label="AI analyst view (Gemini)"
      />

      <PriceChart code={ta.code} tf={tf} />


      <div className="levels-row">
        <Level label="Suggested stop" value={ta.stop} hint="2× ATR below price" />
        <Level label="Suggested target" value={ta.target} hint="3× ATR above price" />
        <Level label="Volatility (ATR)" value={ta.atr_pct} suffix="%" hint="daily swing size" />
      </div>

      {/* conviction metrics: institutional + quant sizing */}
      <div className="conviction-row">
        {ta.analyst_consensus && (
          <span className="chip" title="Wall Street analyst rating distribution (Finnhub)">
            🏦 Analysts:{' '}
            <b style={{ color: consensusColor(ta.analyst_consensus.score) }}>
              {ta.analyst_consensus.label}
            </b>{' '}
            <span className="muted small">
              ({ta.analyst_consensus.strong_buy + ta.analyst_consensus.buy} buy /{' '}
              {ta.analyst_consensus.hold} hold /{' '}
              {ta.analyst_consensus.sell + ta.analyst_consensus.strong_sell} sell · {ta.analyst_consensus.total})
            </span>
          </span>
        )}
        {ta.fundamental_quality && (
          <span
            className="chip"
            title={`Business-quality lens (separate from the technical score) — ${ta.fundamental_quality.coverage}: ${ta.fundamental_quality.reasons.join('; ')}`}
          >
            ⭐ Quality:{' '}
            <b style={{ color: ta.fundamental_quality.score_0_100 >= 50 ? 'var(--up)' : 'var(--down)' }}>
              {ta.fundamental_quality.label}
            </b>{' '}
            <span className="muted small">({fmtNum(ta.fundamental_quality.score_0_100, 0)}/100)</span>
          </span>
        )}
        {ta.growth_tilt && ta.growth_tilt.label !== 'Neutral' && (
          <span
            className="chip"
            title={`Size/growth conviction tilt (${ta.growth_tilt.size_class}) — suggests sizing ~${ta.growth_tilt.sizing_multiplier}x the technical read: ${ta.growth_tilt.reasons.join('; ')}`}
          >
            ⚖️ Conviction:{' '}
            <b style={{ color: ta.growth_tilt.tilt >= 0 ? 'var(--up)' : 'var(--down)' }}>
              {ta.growth_tilt.label}
            </b>{' '}
            <span className="muted small">({ta.growth_tilt.size_class})</span>
          </span>
        )}
        {ta.earnings_surprise && (
          <span className="chip" title="Recent quarterly EPS beats/misses vs estimate (post-earnings drift tends to follow the sign)">
            📊 Earnings:{' '}
            <b style={{ color: (ta.earnings_surprise.avg_surprise_pct ?? 0) >= 0 ? 'var(--up)' : 'var(--down)' }}>
              {ta.earnings_surprise.beats}B/{ta.earnings_surprise.misses}M
            </b>
            {ta.earnings_surprise.avg_surprise_pct != null && (
              <span className="muted small"> (avg {ta.earnings_surprise.avg_surprise_pct >= 0 ? '+' : ''}{fmtNum(ta.earnings_surprise.avg_surprise_pct, 1)}%)</span>
            )}
          </span>
        )}
        {ta.order_book && ta.order_book.imbalance_pct != null && (
          <span
            className="chip"
            title={`Level-2 order book (${ta.order_book.bid_levels}/${ta.order_book.ask_levels} levels): ${fmtNum(ta.order_book.bid_vol, 0)} bid vs ${fmtNum(ta.order_book.ask_vol, 0)} ask · spread ${ta.order_book.spread_pct ?? '—'}%`}
          >
            📚 Book:{' '}
            <b
              style={{
                color:
                  ta.order_book.imbalance_pct >= 55
                    ? 'var(--up)'
                    : ta.order_book.imbalance_pct <= 45
                      ? 'var(--down)'
                      : 'var(--muted)',
              }}
            >
              {fmtNum(ta.order_book.imbalance_pct, 0)}% bid
            </b>
          </span>
        )}
        {ta.next_earnings && (
          <span className="chip" title="Next confirmed earnings date (Finnhub calendar)">
            📅 Earnings <b>{ta.next_earnings.date}</b>
          </span>
        )}
        {ta.insider && ta.insider.direction !== 'neutral' && (
          <span className="chip" title="Net insider buying/selling over the last ~6 months (Finnhub MSPR)">
            👤 Insiders:{' '}
            <b style={{ color: ta.insider.net_mspr >= 0 ? 'var(--up)' : 'var(--down)' }}>
              {ta.insider.direction}
            </b>
          </span>
        )}
        {ta.reward_risk != null && <span className="chip">Reward:Risk {fmtNum(ta.reward_risk)}:1</span>}
        {ta.kelly_sizing_pct != null && (
          <span className="chip" title="Half-Kelly position size, capped at 15%">
            ½-Kelly size {fmtNum(ta.kelly_sizing_pct, 1)}%
          </span>
        )}
        {ta.rel_strength_pct != null && (
          <span className="chip" title="Excess return vs S&P 500 over the window">
            RS vs SPY{' '}
            <b style={{ color: ta.rel_strength_pct >= 0 ? 'var(--up)' : 'var(--down)' }}>
              {ta.rel_strength_pct >= 0 ? '+' : ''}{fmtNum(ta.rel_strength_pct, 1)}%
            </b>
          </span>
        )}
        {ta.beta != null && <span className="chip">β {fmtNum(ta.beta)}</span>}
      </div>

      <TradePanel ta={ta} />

      {ta.risk_alerts && ta.risk_alerts.length > 0 && (
        <div className="why">
          <h4 style={{ color: 'var(--down)' }}>⚠ Risk alerts</h4>
          <ul className="reasons">
            {ta.risk_alerts.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      )}

      <div className="why">
        <h4>Why</h4>
        <ul className="reasons reasons-top">
          {ta.reasons.map((r, i) => (
            <li key={i}>{r}</li>
          ))}
        </ul>
      </div>

      <details className="breakdown">
        <summary>Analyst team breakdown ({ta.components.length} dimensions)</summary>
        <AnalystBreakdown components={ta.components} />
      </details>

      <div className="ask-section">
        <h4>Ask the analyst</h4>
        <AskBox fetcher={(q, p) => api.askSymbol(ta.code, q, p, tf)} />
      </div>
    </div>
  )
}

function Level({
  label,
  value,
  suffix = '',
  hint,
}: {
  label: string
  value: number | null
  suffix?: string
  hint?: string
}) {
  return (
    <div className="level" title={hint}>
      <div className="level-label">{label}</div>
      <div className="level-value">{value === null ? '—' : `${fmtNum(value)}${suffix}`}</div>
    </div>
  )
}
