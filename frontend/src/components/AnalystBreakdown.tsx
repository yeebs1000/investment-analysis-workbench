import type { SignalComponent } from '../types'

const ICONS: Record<string, string> = {
  Trend: '📈',
  Momentum: '⚡',
  Quant: '🧮',
  Volatility: '🌊',
  Volume: '📊',
  Levels: '🎯',
  'ML Forecast': '🤖',
}

function barColor(score: number): string {
  if (score > 0.15) return 'var(--up)'
  if (score < -0.15) return 'var(--down)'
  return 'var(--muted)'
}

// The "multi-analyst team": each dimension with its own read and number-backed reasons.
export function AnalystBreakdown({ components }: { components: SignalComponent[] }) {
  return (
    <div className="analysts">
      {components.map((c) => (
        <div className="analyst" key={c.name}>
          <div className="analyst-head">
            <span className="analyst-name">
              {ICONS[c.name] ?? '•'} {c.name}
            </span>
            <span className="analyst-summary" style={{ color: barColor(c.score) }}>
              {c.summary}
            </span>
          </div>
          <div className="analyst-bar">
            <div className="analyst-bar-track">
              <div className="analyst-bar-zero" />
              <div
                className="analyst-bar-fill"
                style={{
                  background: barColor(c.score),
                  left: c.score >= 0 ? '50%' : `${50 + c.score * 50}%`,
                  width: `${Math.abs(c.score) * 50}%`,
                }}
              />
            </div>
            <span className="analyst-weight">weight {(c.weight * 100).toFixed(0)}%</span>
          </div>
          <ul className="reasons">
            {c.reasons.map((r, i) => (
              <li key={i}>{r}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  )
}
