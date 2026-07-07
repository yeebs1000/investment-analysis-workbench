import { scoreColor } from '../format'

// A 0-100 horizontal meter with a marker at the score and a neutral line at 50.
export function ScoreMeter({ score, confidence }: { score: number; confidence?: string }) {
  return (
    <div className="meter-wrap">
      <div className="meter">
        <div className="meter-mid" />
        <div
          className="meter-fill"
          style={{ width: `${score}%`, background: scoreColor(score) }}
        />
        <div className="meter-knob" style={{ left: `${score}%`, borderColor: scoreColor(score) }} />
      </div>
      <div className="meter-labels">
        <span>Bearish</span>
        <span className="meter-score" style={{ color: scoreColor(score) }}>
          {score.toFixed(0)}/100{confidence ? ` · ${confidence} confidence` : ''}
        </span>
        <span>Bullish</span>
      </div>
    </div>
  )
}
