import { Fragment, useEffect, useState } from 'react'
import { api } from '../api'
import type { TechnicalAnalysis, WatchlistAnalysis, WatchlistGroup } from '../types'
import { DecisionBadge } from './DecisionBadge'
import { AnalysisCard } from './AnalysisCard'
import { Error, Loading } from './PortfolioView'
import { fmtNum } from '../format'

type SortKey = 'score_desc' | 'score_asc'
const SORTS: { value: SortKey; label: string }[] = [
  { value: 'score_desc', label: 'Signal score (high → low)' },
  { value: 'score_asc', label: 'Signal score (low → high)' },
]

function sortItems(items: TechnicalAnalysis[], sort: SortKey): TechnicalAnalysis[] {
  // errored (no-data) items always sink to the bottom, regardless of direction.
  const sorted = [...items].sort((a, b) => {
    if (!!a.error !== !!b.error) return a.error ? 1 : -1
    return sort === 'score_asc' ? a.score - b.score : b.score - a.score
  })
  return sorted
}

export function WatchlistView() {
  const [groups, setGroups] = useState<WatchlistGroup[]>([])
  const [group, setGroup] = useState<string>('')
  const [data, setData] = useState<WatchlistAnalysis | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [open, setOpen] = useState<string | null>(null)
  const [sort, setSort] = useState<SortKey>('score_desc')

  useEffect(() => {
    api
      .watchlists()
      .then((g) => {
        setGroups(g)
        if (g.length && !group) setGroup(g[0].name)
      })
      .catch((e) => setErr(String(e.message ?? e)))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const current = groups.find((x) => x.name === group)
  const isLocal = current?.source === 'local'

  const run = (g: string) => {
    if (!g) return
    setLoading(true)
    setErr(null)
    setData(null)
    setOpen(null)
    api
      .watchlist(g, 30, 'day', groups.find((x) => x.name === g)?.source)
      .then(setData)
      .catch((e) => setErr(String(e.message ?? e)))
      .finally(() => setLoading(false))
  }

  const reloadGroups = (select?: string) =>
    api.watchlists().then((g) => {
      setGroups(g)
      setGroup(select && g.some((x) => x.name === select) ? select : g[0]?.name ?? '')
    })

  const removeSymbol = (code: string) => {
    api
      .watchlistRemove(code, group)
      .then(() => run(group))
      .catch((e) => setErr(String(e.message ?? e)))
  }

  const deleteList = () => {
    if (!isLocal || !confirm(`Delete the "${group}" list? (symbols only — no positions touched)`)) return
    api
      .watchlistDelete(group)
      .then(() => {
        setData(null)
        return reloadGroups()
      })
      .catch((e) => setErr(String(e.message ?? e)))
  }

  return (
    <div>
      <div className="toolbar">
        <label className="muted small">Watchlist group:</label>
        <select value={group} onChange={(e) => setGroup(e.target.value)}>
          {groups.map((g) => (
            <option key={g.name} value={g.name}>
              {g.name}{g.source === 'local' ? ' (app)' : ''}
            </option>
          ))}
        </select>
        <button onClick={() => run(group)} disabled={loading || !group}>
          {loading ? 'Analyzing…' : 'Analyze group'}
        </button>
        {isLocal && (
          <button className="btn-link" onClick={deleteList} disabled={loading}>
            🗑 Delete list
          </button>
        )}
        {data && (
          <label className="tf-select">
            Sort by
            <select value={sort} onChange={(e) => setSort(e.target.value as SortKey)}>
              {SORTS.map((s) => (
                <option key={s.value} value={s.value}>{s.label}</option>
              ))}
            </select>
          </label>
        )}
        <span className="muted small">Ranks every symbol in the group by signal strength.</span>
      </div>

      {err && <Error msg={err} onRetry={() => run(group)} />}
      {loading && <Loading label={`Running the analyst team across "${group}"…`} />}

      {data && (
        <div className="row-list">
          {sortItems(data.items, sort).map((ta, i) => {
            const expanded = open === ta.code
            return (
              <Fragment key={ta.code}>
                <button
                  type="button"
                  className={`row-item ${expanded ? 'row-open' : ''} ${ta.error ? 'row-err' : ''}`}
                  aria-expanded={expanded}
                  disabled={!!ta.error}
                  onClick={() => !ta.error && setOpen(expanded ? null : ta.code)}
                >
                  <div className="row-main">
                    <div className="sym">
                      <span className="muted small">#{i + 1}</span> {ta.code}
                    </div>
                    <div className="muted small">{ta.name}</div>
                  </div>
                  <div className="row-stats">
                    {ta.error ? (
                      <span className="muted small">no data</span>
                    ) : (
                      <>
                        <span className="score-pill">{fmtNum(ta.score, 0)}</span>
                        <DecisionBadge decision={ta.decision} size="sm" />
                        <span className="chevron">{expanded ? '▾' : '▸'}</span>
                      </>
                    )}
                  </div>
                </button>
                {expanded && (
                  <div className="row-detail">
                    <div className="chiprow">
                      <span className="chip">Last {fmtNum(ta.price)}</span>
                      <span className="chip">Confidence {ta.confidence_label}</span>
                      {isLocal && (
                        <button className="btn-link" onClick={() => removeSymbol(ta.code)}>
                          ✕ Remove from list
                        </button>
                      )}
                    </div>
                    <AnalysisCard ta={ta} />
                  </div>
                )}
              </Fragment>
            )
          })}
        </div>
      )}
      {data && data.errors.length > 0 && (
        <p className="muted small">
          {data.errors.length} symbol(s) had no market-data permission and were skipped (e.g. China A-shares).
        </p>
      )}
    </div>
  )
}
