import type {
  Account,
  AskResponse,
  ChartSeries,
  FundamentalMetrics,
  OptimizerPlan,
  OptionsAnalysis,
  Performance,
  PortfolioAnalysis,
  Position,
  SearchResult,
  TechnicalAnalysis,
  WatchlistAnalysis,
  WatchlistGroup,
} from './types'

async function get<T>(url: string): Promise<T> {
  const res = await fetch(url)
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  return res.json() as Promise<T>
}

export interface Health {
  status: string
  broker: string
  brokers?: string[]
  broker_status?: Record<string, string> // e.g. { moomoo: 'unreachable', ibkr: 'connected' }
  ibkr_enabled?: boolean
  tiger_enabled?: boolean
  security_firm: string
  read_only: boolean
  disclaimer: string
}

export type Provider = 'gemini' | 'claude' | 'none'

export interface Narrative {
  available: boolean
  provider: string | null
  model: string | null
  cached: boolean
  text: string
  message: string
}

export interface TimeframeList {
  default: string
  items: { value: string; label: string }[]
}

export interface LLMStatus {
  default: string
  available: Record<string, boolean>
  models: Record<string, string>
  options: Record<string, string[]>
}

export interface LLMUsage {
  by_provider: Record<
    string,
    { calls: number; input_tokens: number; output_tokens: number; est_cost_usd: number }
  >
  total_est_cost_usd: number
  total_calls: number
  cached_entries: number
  note: string
}

export const api = {
  health: () => get<Health>('/api/health'),
  account: () => get<Account>('/api/account'),
  positions: () => get<Position[]>('/api/positions'),
  portfolio: (tf = 'day') => get<PortfolioAnalysis>(`/api/portfolio?tf=${tf}`),
  performance: () => get<Performance>('/api/performance'),
  optimize: (tf = 'day', method: 'heuristic' | 'risk_aware' = 'heuristic', capPct = 15) =>
    get<OptimizerPlan>(`/api/optimize?tf=${tf}&method=${method}&cap_pct=${capPct}`),
  analyze: (code: string, tf = 'day') =>
    get<TechnicalAnalysis>(`/api/analyze/${encodeURIComponent(code)}?tf=${tf}`),
  timeframes: () => get<TimeframeList>('/api/timeframes'),
  chart: (code: string, lookback = 180, tf = 'day') =>
    get<ChartSeries>(`/api/chart/${encodeURIComponent(code)}?lookback=${lookback}&tf=${tf}`),
  search: (q: string) => get<SearchResult[]>(`/api/search?q=${encodeURIComponent(q)}`),
  watchlistAdd: (code: string, group?: string, source?: string) =>
    post(`/api/watchlist/add?code=${encodeURIComponent(code)}${group ? `&group=${encodeURIComponent(group)}` : ''}${source ? `&source=${source}` : ''}`),
  watchlistRemove: (code: string, group: string) =>
    post(`/api/watchlist/remove?code=${encodeURIComponent(code)}&group=${encodeURIComponent(group)}`),
  watchlistDelete: (group: string) =>
    post(`/api/watchlist/delete?group=${encodeURIComponent(group)}`),
  watchlists: () => get<WatchlistGroup[]>('/api/watchlists'),
  watchlist: (group: string, limit = 30, tf = 'day', source?: string) =>
    get<WatchlistAnalysis>(`/api/watchlist?group=${encodeURIComponent(group)}&limit=${limit}&tf=${tf}${source ? `&source=${source}` : ''}`),
  askSymbol: (code: string, q: string, provider: Provider, tf = 'day') =>
    get<AskResponse>(`/api/ask/${encodeURIComponent(code)}?q=${encodeURIComponent(q)}&tf=${tf}&provider=${provider}`),
  askOptions: (code: string, q: string, provider: Provider, dte = 35) =>
    get<AskResponse>(`/api/options/${encodeURIComponent(code)}/ask?q=${encodeURIComponent(q)}&dte=${dte}&provider=${provider}`),
  fundamentals: (code: string) => get<FundamentalMetrics>(`/api/fundamentals/${encodeURIComponent(code)}`),
  askFundamentals: (code: string, q: string, provider: Provider) =>
    get<AskResponse>(`/api/fundamentals/${encodeURIComponent(code)}/ask?q=${encodeURIComponent(q)}&provider=${provider}`),

  llmStatus: () => get<LLMStatus>('/api/llm/status'),
  llmUsage: () => get<LLMUsage>('/api/llm/usage'),
  llmReset: () => post('/api/llm/reset'),
  llmSetModel: (provider: string, model: string) =>
    post<LLMStatus>(`/api/llm/model?provider=${provider}&model=${encodeURIComponent(model)}`),
  explainSymbol: (code: string, provider: Provider, tf = 'day') =>
    get<Narrative>(`/api/explain/${encodeURIComponent(code)}?provider=${provider}&tf=${tf}`),
  explainPortfolio: (provider: Provider) =>
    get<Narrative>(`/api/portfolio/explain?provider=${provider}`),
  options: (code: string, dte = 35) =>
    get<OptionsAnalysis>(`/api/options/${encodeURIComponent(code)}?dte=${dte}`),
  explainOptions: (code: string, provider: Provider, dte = 35) =>
    get<Narrative>(`/api/options/${encodeURIComponent(code)}/explain?dte=${dte}&provider=${provider}`),
}

async function post<T = unknown>(url: string): Promise<T> {
  const res = await fetch(url, { method: 'POST' })
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      detail = body.detail ?? detail
    } catch {
      /* ignore */
    }
    throw new Error(detail)
  }
  try {
    return (await res.json()) as T
  } catch {
    return undefined as T
  }
}
