import { useEffect, useRef, useState } from 'react'
import { api } from '../api'
import type { SearchResult, TechnicalAnalysis } from '../types'
import { AnalysisCard } from './AnalysisCard'
import { OptionsPanel } from './OptionsPanel'
import { FundamentalsPanel } from './FundamentalsPanel'
import { Loading } from './PortfolioView'

const EXAMPLES = ['US.AAPL', 'US.NVDA', 'HK.00700', 'SG.D05', 'JP.6981']
const CODE_RE = /^[A-Za-z]{2}\.[A-Za-z0-9]+$/   // looks like MARKET.CODE already

const FALLBACK_TFS = [
  { value: 'day', label: 'Daily' },
  { value: 'week', label: 'Weekly' },
  { value: 'month', label: 'Monthly' },
  { value: '60m', label: '1 hour' },
  { value: '15m', label: '15 min' },
]

export function SymbolView() {
  const [code, setCode] = useState('US.AAPL')
  const [query, setQuery] = useState('US.AAPL')
  const [tf, setTf] = useState('day')
  const [tfs, setTfs] = useState(FALLBACK_TFS)
  const [ta, setTa] = useState<TechnicalAnalysis | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState<SearchResult[]>([])
  const [searching, setSearching] = useState(false)
  const debounce = useRef<number | undefined>(undefined)

  useEffect(() => {
    api.timeframes().then((t) => setTfs(t.items)).catch(() => {})
  }, [])

  const run = (c: string, timeframe: string = tf) => {
    const sym = c.trim().toUpperCase()
    if (!sym) return
    setCode(sym)
    setQuery(sym)
    setResults([])
    setLoading(true)
    setErr(null)
    api
      .analyze(sym, timeframe)
      .then(setTa)
      .catch((e) => setErr(String(e.message ?? e)))
      .finally(() => setLoading(false))
  }

  // Free-text search (company name or bare ticker) -> resolve to MARKET.CODE.
  const onQueryChange = (v: string) => {
    setQuery(v)
    window.clearTimeout(debounce.current)
    const q = v.trim()
    if (q.length < 2 || CODE_RE.test(q)) {
      setResults([])
      return
    }
    debounce.current = window.setTimeout(() => {
      setSearching(true)
      api.search(q).then(setResults).catch(() => setResults([])).finally(() => setSearching(false))
    }, 300)
  }

  const submit = () => {
    const q = query.trim()
    if (CODE_RE.test(q)) run(q)
    else if (results.length) run(results[0].code)
    else run(q) // last resort: treat as a code
  }

  const onTfChange = (v: string) => {
    setTf(v)
    if (ta && !loading) run(code, v) // re-analyze the current symbol on the new timeframe
  }

  return (
    <div>
      <div className="toolbar">
        <div className="search-wrap">
          <input
            value={query}
            onChange={(e) => onQueryChange(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && submit()}
            placeholder="Search ticker or company — e.g. nvda, Tencent, US.AAPL"
            className="sym-input"
          />
          {(results.length > 0 || searching) && (
            <div className="search-results">
              {searching && <div className="search-item muted small">Searching…</div>}
              {results.map((r) => (
                <button key={r.code} className="search-item" onClick={() => run(r.code)}>
                  <span className="sym">{r.code}</span> <span className="name">{r.name}</span>
                  {r.type && <span className="muted small"> · {r.type}</span>}
                </button>
              ))}
            </div>
          )}
        </div>
        <label className="tf-select">
          Timeframe
          <select value={tf} onChange={(e) => onTfChange(e.target.value)} disabled={loading}>
            {tfs.map((t) => (
              <option key={t.value} value={t.value}>
                {t.label}
              </option>
            ))}
          </select>
        </label>
        <button onClick={submit} disabled={loading}>
          {loading ? 'Analyzing…' : 'Analyze'}
        </button>
      </div>
      <div className="examples">
        {EXAMPLES.map((e) => (
          <button key={e} className="chip-btn" onClick={() => { setCode(e); run(e) }}>
            {e}
          </button>
        ))}
      </div>

      {loading && <Loading label={`Analyzing ${code}…`} />}
      {err && <div className="error"><p>⚠ {err}</p></div>}
      {ta && !loading && (
        <>
          <AnalysisCard ta={ta} tf={tf} />
          {!ta.error && <FundamentalsPanel code={ta.code} />}
          {!ta.error && <OptionsPanel code={ta.code} />}
        </>
      )}
    </div>
  )
}
