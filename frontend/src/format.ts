import type { Decision } from './types'

export const DECISION_META: Record<
  Decision,
  { label: string; color: string; plain: string }
> = {
  STRONG_BUY: { label: 'Strong Buy', color: 'var(--up)', plain: 'Signals strongly favour buying.' },
  BUY: { label: 'Buy', color: 'var(--up)', plain: 'Conditions favour opening or adding a position.' },
  ACCUMULATE: { label: 'Accumulate', color: 'var(--up)', plain: 'Add gradually on strength or dips.' },
  HOLD: { label: 'Hold', color: 'var(--muted)', plain: 'No strong edge — keep what you have.' },
  REDUCE: { label: 'Reduce / Trim', color: 'var(--warn)', plain: 'Lighten the position; momentum is fading.' },
  SELL: { label: 'Sell', color: 'var(--down)', plain: 'Signals favour exiting the position.' },
}

export function fmtNum(x: number | null | undefined, dp = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  return x.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp })
}

export function fmtMoney(x: number | null | undefined, ccy = '', dp = 2): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  const v = x.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp })
  return ccy ? `${ccy} ${v}` : v
}

export function fmtPct(x: number | null | undefined, dp = 1): string {
  if (x === null || x === undefined || Number.isNaN(x)) return '—'
  const sign = x > 0 ? '+' : ''
  return `${sign}${x.toFixed(dp)}%`
}

export function plColor(x: number | null | undefined): string {
  if (x === null || x === undefined || x === 0) return 'var(--muted)'
  return x > 0 ? 'var(--up)' : 'var(--down)'
}

// Color a 0-100 score from red (0) through grey (50) to green (100).
export function scoreColor(score: number): string {
  if (score >= 60) return 'var(--up)'
  if (score >= 54) return 'var(--up)'
  if (score >= 46) return 'var(--muted)'
  if (score >= 36) return 'var(--warn)'
  return 'var(--down)'
}
