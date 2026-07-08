import { useEffect, useState } from 'react'
import { api, type LLMStatus, type Provider } from '../api'
import { useProvider } from '../ProviderContext'

const OPTIONS: { id: Provider; label: string }[] = [
  { id: 'none', label: 'Deterministic (free)' },
  { id: 'gemini', label: 'Gemini' },
  { id: 'claude', label: 'Claude' },
]

export function LLMToggle() {
  const { provider, setProvider } = useProvider()
  const [status, setStatus] = useState<LLMStatus | null>(null)

  useEffect(() => {
    api.llmStatus().then(setStatus).catch(() => setStatus(null))
  }, [])

  // If the selected provider isn't actually available (e.g. Claude with no API
  // key), fall back to Gemini (if keyed) or Deterministic. Covers the load race
  // where a provider was picked before status arrived.
  useEffect(() => {
    if (status && provider !== 'none' && !status.available[provider]) {
      setProvider(status.available.gemini ? 'gemini' : 'none')
    }
  }, [status, provider, setProvider])

  // Lock non-free providers until status confirms a key exists (unknown = locked,
  // so nothing is clickable before we know it's usable).
  const disabled = (id: Provider) => id !== 'none' && (!status || !status.available[id])

  return (
    <div className="llm-toggle" title="Toggle the AI that explains the analysis (controls API cost).">
      <span className="muted small">Explain with:</span>
      <div className="seg">
        {OPTIONS.map((o) => {
          const off = disabled(o.id)
          return (
            <button
              key={o.id}
              className={`seg-btn ${provider === o.id ? 'seg-on' : ''}`}
              onClick={() => setProvider(o.id)}
              disabled={off}
              title={
                off
                  ? `${o.label} needs an API key in backend/.env`
                  : status?.models[o.id] ?? ''
              }
            >
              {o.label}
              {off ? ' 🔒' : ''}
            </button>
          )
        })}
      </div>
      {provider !== 'none' && status?.options?.[provider] && (
        <label className="tf-select" title={`Which ${provider} model answers analysis requests`}>
          <select
            value={status.models[provider]}
            onChange={(e) =>
              api.llmSetModel(provider, e.target.value).then(setStatus).catch(() => {})
            }
          >
            {status.options[provider].map((m) => (
              <option key={m} value={m}>
                {m}
              </option>
            ))}
          </select>
        </label>
      )}
    </div>
  )
}
