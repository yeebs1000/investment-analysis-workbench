import { useEffect, useState } from 'react'
import { api, type LLMUsage } from '../api'
import { useProvider } from '../ProviderContext'

export function CostMeter() {
  const { bump, notifyUsage } = useProvider()
  const [usage, setUsage] = useState<LLMUsage | null>(null)

  const load = () => api.llmUsage().then(setUsage).catch(() => setUsage(null))
  useEffect(() => {
    load()
  }, [bump])

  const reset = async () => {
    await api.llmReset()
    notifyUsage()
  }

  const cost = usage?.total_est_cost_usd ?? 0
  return (
    <div className="cost-meter" title="Estimated LLM spend this session (from approximate per-token pricing).">
      <span className="cost-dot" />
      <span>
        ~${cost.toFixed(4)} · {usage?.total_calls ?? 0} calls
        {usage && usage.cached_entries > 0 ? ` · ${usage.cached_entries} cached` : ''}
      </span>
      {(usage?.total_calls ?? 0) > 0 && (
        <button className="cost-reset" onClick={reset} title="Reset the session cost counter">
          reset
        </button>
      )}
    </div>
  )
}
