import { useEffect, useState } from 'react'
import { api } from '../api'
import type { ChartSeries } from '../types'

// Pure-SVG price chart: close line + EMA20/50/200 + Bollinger band,
// with a volume strip and an RSI(14) sub-panel. All values are pre-computed
// server-side (deterministic) -- this component only draws them.

type Num = number | null

const W = 760
const PAD_L = 46
const PAD_R = 10
const PRICE_H = 230
const VOL_H = 46
const RSI_H = 70
const GAP = 14
const TOTAL_H = PRICE_H + GAP + VOL_H + GAP + RSI_H

const COL = {
  close: 'var(--text)',
  ema20: 'var(--chart-ema-fast)',
  ema50: 'var(--chart-ema-mid)',
  ema200: 'var(--chart-ema-slow)',
  band: 'rgba(120,130,150,0.16)',
  bandLine: 'rgba(150,160,180,0.45)',
  vol: 'rgba(120,130,150,0.55)',
  rsi: 'var(--up)',
  grid: 'var(--border)',
  muted: 'var(--muted)',
}

const innerW = W - PAD_L - PAD_R

function xAt(i: number, n: number): number {
  if (n <= 1) return PAD_L
  return PAD_L + (i / (n - 1)) * innerW
}

// build an SVG path for a series within [top, top+height], scaled by [min,max]
function linePath(vals: Num[], top: number, height: number, min: number, max: number): string {
  const span = max - min || 1
  let d = ''
  let started = false
  vals.forEach((v, i) => {
    if (v === null || Number.isNaN(v)) {
      started = false
      return
    }
    const x = xAt(i, vals.length)
    const y = top + height - ((v - min) / span) * height
    d += `${started ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)} `
    started = true
  })
  return d.trim()
}

export function PriceChart({ code, tf = 'day' }: { code: string; tf?: string }) {
  const [data, setData] = useState<ChartSeries | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let alive = true
    setLoading(true)
    setErr(null)
    api
      .chart(code, 180, tf)
      .then((d) => {
        if (alive) setData(d)
      })
      .catch((e) => alive && setErr(String(e.message ?? e)))
      .finally(() => alive && setLoading(false))
    return () => {
      alive = false
    }
  }, [code, tf])

  if (loading) return <div className="chart-wrap muted small">Loading chart…</div>
  if (err) return <div className="chart-wrap muted small">Chart unavailable: {err}</div>
  if (!data || data.error) return <div className="chart-wrap muted small">{data?.error ?? 'No chart data.'}</div>
  if (data.close.length < 2) return <div className="chart-wrap muted small">Not enough history to chart.</div>

  const n = data.close.length

  // price scale spans candles + bollinger band + EMAs visible range
  const priceVals: number[] = []
  for (let i = 0; i < n; i++) {
    priceVals.push(data.high[i], data.low[i])
    const u = data.bb_upper[i]
    const l = data.bb_lower[i]
    if (u !== null) priceVals.push(u)
    if (l !== null) priceVals.push(l)
  }
  let pMin = Math.min(...priceVals)
  let pMax = Math.max(...priceVals)
  const padP = (pMax - pMin) * 0.05 || 1
  pMin -= padP
  pMax += padP

  const volMax = Math.max(...data.volume, 1)
  const volTop = PRICE_H + GAP

  // bollinger band as a filled area (upper forward, lower backward)
  let bandD = ''
  const upPts: string[] = []
  const loPts: string[] = []
  for (let i = 0; i < n; i++) {
    const u = data.bb_upper[i]
    const l = data.bb_lower[i]
    if (u === null || l === null) continue
    const x = xAt(i, n)
    const yu = PRICE_H - ((u - pMin) / (pMax - pMin || 1)) * PRICE_H
    const yl = PRICE_H - ((l - pMin) / (pMax - pMin || 1)) * PRICE_H
    upPts.push(`${x.toFixed(1)},${yu.toFixed(1)}`)
    loPts.push(`${x.toFixed(1)},${yl.toFixed(1)}`)
  }
  if (upPts.length > 1) {
    bandD = `M${upPts.join(' L')} L${loPts.reverse().join(' L')} Z`
  }

  // price gridlines (4 horizontal)
  const priceTicks = [0, 0.25, 0.5, 0.75, 1].map((f) => {
    const val = pMax - f * (pMax - pMin)
    const y = f * PRICE_H
    return { y, val }
  })

  // x-axis labels (~6 evenly spaced). Intraday bars carry a time component
  // ("YYYY-MM-DD HH:MM") -> show MM-DD HH:MM; daily/weekly -> show YY-MM-DD.
  const intraday = (data.timeframe ?? 'day').endsWith('m')
  const labelCount = 6
  const xLabels = Array.from({ length: labelCount }, (_, k) => {
    const i = Math.round((k / (labelCount - 1)) * (n - 1))
    const raw = data.time[i] ?? ''
    const t = intraday ? raw.slice(5) : raw.slice(2) // MM-DD HH:MM  vs  YY-MM-DD
    return { x: xAt(i, n), t }
  })

  const lastClose = data.close[n - 1]
  const firstClose = data.close[0]
  const chg = ((lastClose - firstClose) / firstClose) * 100

  const closeColor =
    lastClose >= firstClose ? 'var(--up)' : 'var(--down)'

  // RSI panel
  const rsiTop = volTop + VOL_H + GAP
  const rsiPath = linePath(data.rsi14, rsiTop, RSI_H, 0, 100)
  const rsiY = (v: number) => rsiTop + RSI_H - (v / 100) * RSI_H

  return (
    <div className="chart-wrap">
      <div className="chart-legend">
        <span style={{ color: closeColor }}>
          {data.name} · {lastClose.toFixed(2)}{' '}
          <span className="small">({chg >= 0 ? '+' : ''}{chg.toFixed(1)}% over {n} bars)</span>
        </span>
        <span className="chart-keys small">
          <i style={{ background: COL.ema20 }} /> EMA20
          <i style={{ background: COL.ema50 }} /> EMA50
          <i style={{ background: COL.ema200 }} /> EMA200
          <i style={{ background: COL.bandLine }} /> Bollinger(20,2)
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${TOTAL_H}`} width="100%" className="price-chart" role="img"
           aria-label={`Price chart for ${data.name}`}>
        {/* price gridlines + labels */}
        {priceTicks.map((t, i) => (
          <g key={i}>
            <line x1={PAD_L} y1={t.y} x2={W - PAD_R} y2={t.y} stroke={COL.grid} strokeWidth={0.5} />
            <text x={PAD_L - 6} y={t.y + 3} textAnchor="end" fontSize={10} fill={COL.muted}>
              {t.val.toFixed(t.val >= 100 ? 0 : 2)}
            </text>
          </g>
        ))}

        {/* bollinger band fill */}
        {bandD && <path d={bandD} fill={COL.band} stroke="none" />}
        <path d={linePath(data.bb_upper, 0, PRICE_H, pMin, pMax)} fill="none"
              stroke={COL.bandLine} strokeWidth={0.7} strokeDasharray="3 3" />
        <path d={linePath(data.bb_lower, 0, PRICE_H, pMin, pMax)} fill="none"
              stroke={COL.bandLine} strokeWidth={0.7} strokeDasharray="3 3" />

        {/* EMAs */}
        <path d={linePath(data.ema200, 0, PRICE_H, pMin, pMax)} fill="none" stroke={COL.ema200} strokeWidth={1.2} />
        <path d={linePath(data.ema50, 0, PRICE_H, pMin, pMax)} fill="none" stroke={COL.ema50} strokeWidth={1.2} />
        <path d={linePath(data.ema20, 0, PRICE_H, pMin, pMax)} fill="none" stroke={COL.ema20} strokeWidth={1.2} />

        {/* close line */}
        <path d={linePath(data.close, 0, PRICE_H, pMin, pMax)} fill="none" stroke={COL.close} strokeWidth={1.6} />

        {/* volume strip */}
        <line x1={PAD_L} y1={volTop + VOL_H} x2={W - PAD_R} y2={volTop + VOL_H} stroke={COL.grid} strokeWidth={0.5} />
        {data.volume.map((v, i) => {
          const x = xAt(i, n)
          const h = (v / volMax) * VOL_H
          const bw = Math.max(innerW / n - 0.6, 0.6)
          const up = i === 0 || data.close[i] >= data.close[i - 1]
          return (
            <rect key={i} x={x - bw / 2} y={volTop + VOL_H - h} width={bw} height={h}
                  fill={up ? 'rgba(52,168,83,0.5)' : 'rgba(234,67,53,0.5)'} />
          )
        })}
        <text x={PAD_L - 6} y={volTop + 9} textAnchor="end" fontSize={9} fill={COL.muted}>Vol</text>

        {/* RSI panel */}
        <line x1={PAD_L} y1={rsiY(70)} x2={W - PAD_R} y2={rsiY(70)} stroke={COL.grid} strokeWidth={0.5} strokeDasharray="2 3" />
        <line x1={PAD_L} y1={rsiY(30)} x2={W - PAD_R} y2={rsiY(30)} stroke={COL.grid} strokeWidth={0.5} strokeDasharray="2 3" />
        <line x1={PAD_L} y1={rsiY(50)} x2={W - PAD_R} y2={rsiY(50)} stroke={COL.grid} strokeWidth={0.4} />
        <path d={rsiPath} fill="none" stroke={COL.rsi} strokeWidth={1.3} />
        <text x={PAD_L - 6} y={rsiY(70) + 3} textAnchor="end" fontSize={9} fill={COL.muted}>70</text>
        <text x={PAD_L - 6} y={rsiY(30) + 3} textAnchor="end" fontSize={9} fill={COL.muted}>30</text>
        <text x={W - PAD_R} y={rsiTop - 3} textAnchor="end" fontSize={9} fill={COL.muted}>
          RSI {data.rsi14[n - 1]?.toFixed(0) ?? '—'}
        </text>

        {/* x-axis date labels */}
        {xLabels.map((l, i) => (
          <text key={i} x={l.x} y={TOTAL_H - 1} textAnchor="middle" fontSize={9} fill={COL.muted}>
            {l.t}
          </text>
        ))}
      </svg>
    </div>
  )
}
