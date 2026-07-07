import { useState } from 'react'
import type { Provider } from '../api'
import type { AskResponse } from '../types'
import { useProvider } from '../ProviderContext'
import { renderMarkdown } from '../markdown'

const SUGGESTIONS = ['What is the realistic topside and what supports it?', 'What is the bear case?', 'Where do I cut the position?']

export function AskBox({
  fetcher,
  placeholder = 'Ask about this setup — e.g. "what\'s the topside?"',
  suggestions = SUGGESTIONS,
}: {
  fetcher: (q: string, provider: Provider) => Promise<AskResponse>
  placeholder?: string
  suggestions?: string[]
}) {
  const { provider, notifyUsage } = useProvider()
  const [q, setQ] = useState('')
  const [ans, setAns] = useState<AskResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)

  const ask = (question: string) => {
    const text = question.trim()
    if (!text) return
    setLoading(true)
    setErr(null)
    fetcher(text, provider)
      .then((r) => { setAns(r); notifyUsage() })
      .catch((e) => setErr(String((e as Error).message ?? e)))
      .finally(() => setLoading(false))
  }

  const disabled = provider === 'none'
  return (
    <div className="askbox">
      <div className="askbox-row">
        <input
          className="ask-input"
          value={q}
          onChange={(e) => setQ(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && ask(q)}
          placeholder={disabled ? 'Pick Gemini/Claude above to ask the analyst' : placeholder}
          disabled={disabled || loading}
        />
        <button onClick={() => ask(q)} disabled={disabled || loading}>
          {loading ? 'Thinking…' : '🔎 Ask'}
        </button>
      </div>
      {!disabled && (
        <div className="ask-suggestions">
          {suggestions.map((s) => (
            <button key={s} className="chip-btn" onClick={() => { setQ(s); ask(s) }} disabled={loading}>
              {s}
            </button>
          ))}
        </div>
      )}
      {err && <div className="nar nar-muted">⚠ {err}</div>}
      {ans && !ans.available && <div className="nar nar-muted">{ans.message}</div>}
      {ans && ans.available && (
        <div className="nar">
          <div className="nar-meta">
            🧠 {ans.provider} · {ans.model}
            {ans.cached && <span className="nar-cached">cached · $0</span>}
          </div>
          <div className="nar-body">{renderMarkdown(ans.answer)}</div>
        </div>
      )}
    </div>
  )
}
