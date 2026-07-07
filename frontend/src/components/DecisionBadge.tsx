import type { Decision } from '../types'
import { DECISION_META } from '../format'

export function DecisionBadge({ decision, size = 'md' }: { decision: Decision; size?: 'sm' | 'md' | 'lg' }) {
  const meta = DECISION_META[decision]
  return (
    <span className={`badge badge-${size}`} style={{ background: meta.color }} title={meta.plain}>
      {meta.label}
    </span>
  )
}
