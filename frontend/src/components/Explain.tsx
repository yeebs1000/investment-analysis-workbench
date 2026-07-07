import { useEffect, useRef, useState } from 'react'
import type { Narrative, Provider } from '../api'
import { useProvider } from '../ProviderContext'
import { renderMarkdown } from '../markdown'

export function NarrativeView({ n }: { n: Narrative }) {
  if (!n.available) return <div className="nar nar-muted">{n.message}</div>
  return (
    <div className="nar">
      <div className="nar-meta">
        🧠 {n.provider} · {n.model}
        {n.cached && <span className="nar-cached">cached · $0</span>}
      </div>
      <div className="nar-body">{renderMarkdown(n.text)}</div>
    </div>
  )
}

export function ExplainButton({
  fetcher,
  label = 'AI analyst view',
  auto = false,
}: {
  fetcher: (provider: Provider) => Promise<Narrative>
  label?: string
  auto?: boolean
}) {
  const { provider, notifyUsage } = useProvider()
  const [n, setN] = useState<Narrative | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  // keep the latest fetcher without retriggering the auto-run effect
  const fetcherRef = useRef(fetcher)
  fetcherRef.current = fetcher

  const run = async () => {
    setLoading(true)
    setErr(null)
    try {
      const res = await fetcherRef.current(provider)
      setN(res)
      notifyUsage()
    } catch (e) {
      setErr(String((e as Error).message ?? e))
    } finally {
      setLoading(false)
    }
  }

  // Auto-run once when mounted with a live provider selected (this component is
  // keyed per symbol+timeframe, so it re-runs whenever those change).
  useEffect(() => {
    if (auto && provider !== 'none') run()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [auto, provider])

  const busyLabel = auto ? 'AI analyst thinking…' : 'Thinking…'
  return (
    <div className="explain">
      <button onClick={run} disabled={loading} className="explain-btn">
        {loading ? busyLabel : n ? '↻ Regenerate' : `🔬 ${label}`}
        <span className="muted small"> ({provider === 'none' ? 'deterministic — pick Gemini/Claude above' : provider})</span>
      </button>
      {err && <div className="nar nar-muted">⚠ {err}</div>}
      {n && <NarrativeView n={n} />}
    </div>
  )
}
