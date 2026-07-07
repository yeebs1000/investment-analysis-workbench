import { useEffect, useState } from 'react'
import { api } from '../api'
import type { TechnicalAnalysis, WatchlistGroup } from '../types'
import { fmtNum } from '../format'

const NEW_LIST = '__new__'

// Maps a Moomoo code to its moomoo.com stock page for a manual order hand-off.
function moomooUrl(code: string): string {
  const [mkt, sym] = code.includes('.') ? code.split('.', 2) : ['US', code]
  return `https://www.moomoo.com/stock/${sym}-${mkt}`
}

function side(ta: TechnicalAnalysis): { label: string; color: string } {
  const d = ta.decision
  if (d === 'STRONG_BUY' || d === 'BUY' || d === 'ACCUMULATE') return { label: 'BUY', color: 'var(--up)' }
  if (d === 'SELL' || d === 'REDUCE') return { label: 'SELL / TRIM', color: 'var(--down)' }
  return { label: 'HOLD', color: 'var(--muted)' }
}

export function TradePanel({ ta }: { ta: TechnicalAnalysis }) {
  const [msg, setMsg] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [groups, setGroups] = useState<WatchlistGroup[]>([])
  const [sel, setSel] = useState<string>(NEW_LIST)
  const [newName, setNewName] = useState('')
  const s = side(ta)

  useEffect(() => {
    api
      .watchlists()
      .then((g) => {
        setGroups(g)
        if (g.length) setSel(g[0].name)
      })
      .catch(() => setGroups([])) // no groups yet -> stays on "new list"
  }, [])

  const addWatch = () => {
    let group: string | undefined
    let source: string | undefined
    if (sel === NEW_LIST) {
      group = newName.trim() || undefined
      source = 'local'
    } else {
      const g = groups.find((x) => x.name === sel)
      group = g?.name
      source = g?.source
    }
    setBusy(true)
    setMsg(null)
    api
      .watchlistAdd(ta.code, group, source)
      .then((r) => {
        const { group: grp, source: src } = (r as { group?: string; source?: string }) ?? {}
        setMsg(`Added to ${src === 'local' ? 'app' : 'Moomoo'} watchlist${grp ? ` "${grp}"` : ''}.`)
        if (src === 'local' && grp && !groups.some((x) => x.name === grp)) {
          setGroups((prev) => [...prev, { name: grp, count: null, source: 'local' }])
          setSel(grp)
          setNewName('')
        }
      })
      .catch((e) => setMsg(`Could not add: ${String((e as Error).message ?? e)}`))
      .finally(() => setBusy(false))
  }

  return (
    <div className="trade-panel">
      <div className="trade-head">
        <b>Trade ticket</b>
        <span className="muted small">prepare here · review &amp; place in Moomoo yourself</span>
      </div>
      <div className="ticket-grid">
        <Cell label="Side" value={s.label} color={s.color} />
        <Cell label="Entry (ref)" value={ta.price != null ? fmtNum(ta.price) : '—'} />
        <Cell label="Stop" value={ta.stop != null ? fmtNum(ta.stop) : '—'} color="var(--down)" />
        <Cell label="Target" value={ta.target != null ? fmtNum(ta.target) : '—'} color="var(--up)" />
        <Cell label="Reward:Risk" value={ta.reward_risk != null ? `${fmtNum(ta.reward_risk)}:1` : '—'} />
        <Cell
          label="Size (½-Kelly)"
          value={ta.kelly_sizing_pct != null ? `${fmtNum(ta.kelly_sizing_pct, 1)}% of book` : '—'}
        />
      </div>
      <div className="trade-actions">
        <button onClick={addWatch} disabled={busy || (sel === NEW_LIST && !newName.trim())}>
          {busy ? 'Adding…' : '☆ Add to watchlist'}
        </button>
        <select value={sel} onChange={(e) => setSel(e.target.value)} disabled={busy}>
          {groups.map((g) => (
            <option key={g.name} value={g.name}>
              {g.name}{g.source === 'local' ? ' (app)' : ''}
            </option>
          ))}
          <option value={NEW_LIST}>＋ New list…</option>
        </select>
        {sel === NEW_LIST && (
          <input
            className="wl-newname"
            placeholder="New list name"
            value={newName}
            onChange={(e) => setNewName(e.target.value)}
            disabled={busy}
          />
        )}
        <a className="btn-link" href={moomooUrl(ta.code)} target="_blank" rel="noreferrer">
          Open in Moomoo ↗
        </a>
      </div>
      {msg && <div className="muted small" style={{ marginTop: 6 }}>{msg}</div>}
    </div>
  )
}

function Cell({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="ticket-cell">
      <div className="level-label">{label}</div>
      <div className="level-value" style={color ? { color } : undefined}>{value}</div>
    </div>
  )
}
