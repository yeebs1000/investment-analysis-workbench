// Mirrors the backend pydantic models (app/data/models.py).

export type Decision =
  | 'STRONG_BUY'
  | 'BUY'
  | 'ACCUMULATE'
  | 'HOLD'
  | 'REDUCE'
  | 'SELL'

export interface SignalComponent {
  name: string
  score: number
  weight: number
  summary: string
  reasons: string[]
  metrics: Record<string, number | null>
}

export interface TechnicalAnalysis {
  code: string
  name: string
  as_of: string | null
  market: string | null
  currency: string | null
  timeframe: string
  higher_tf: string | null
  higher_tf_trend: number | null
  higher_tf_summary: string | null
  price: number | null
  score: number
  decision: Decision
  confidence: number
  confidence_label: string
  components: SignalComponent[]
  reasons: string[]
  stop: number | null
  target: number | null
  atr_pct: number | null
  reward_risk: number | null
  kelly_fraction: number | null
  kelly_sizing_pct: number | null
  rel_strength_pct: number | null
  beta: number | null
  alpha_pct: number | null
  analyst_consensus: AnalystConsensus | null
  earnings_surprise: EarningsSurprise | null
  insider: InsiderSentiment | null
  fundamental_quality: FundamentalQuality | null
  growth_tilt: GrowthTilt | null
  entry_risk: EntryRisk | null
  verdict: TwoAxisVerdict | null
  next_earnings: NextEarnings | null
  order_book: OrderBook | null
  risk_alerts: string[]
  indicators: Record<string, number | null>
  bars_used: number
  error: string | null
}

export interface EntryRisk {
  level: 'caution' | 'high'
  direction: 'up' | 'down'
  label: string
  move_atr_10: number
  stretch_atr: number
  event_gap: boolean
  attribution: string | null
  reasons: string[]
  advice: string
}

export interface TwoAxisVerdict {
  quality_axis: string
  timing_axis: string
  quadrant: string
  guidance: string
}

export interface NextEarnings {
  date: string
  hour: string
  eps_estimate: number | null
}

export interface OrderBook {
  bid_levels: number
  ask_levels: number
  bid_vol: number
  ask_vol: number
  imbalance_pct: number | null
  best_bid: number
  best_ask: number
  spread_pct: number | null
}

export interface EarningsSurprise {
  beats: number
  misses: number
  avg_surprise_pct: number | null
  last_surprise_pct: number | null
  last_period: string | null
  quarters: { period: string | null; actual: number; estimate: number; surprise_pct: number | null }[]
}

export interface InsiderSentiment {
  net_mspr: number
  net_change: number
  months: number
  direction: string
}

export interface FundamentalQuality {
  score_0_100: number
  label: string
  coverage: string
  reasons: string[]
  missing: string[]
}

export interface GrowthTilt {
  tilt: number
  label: string
  size_class: string
  sizing_multiplier: number
  reasons: string[]
}

export interface AnalystConsensus {
  strong_buy: number
  buy: number
  hold: number
  sell: number
  strong_sell: number
  total: number
  score: number
  label: string
  as_of: string | null
}

export interface Position {
  code: string
  name: string
  market: string
  currency: string
  broker: string
  side: string
  qty: number
  cost_price: number | null
  last_price: number | null
  market_value: number
  pl_ratio_pct: number | null
  pl_value: number | null
  today_pl_value: number | null
}

export interface HoldingAnalysis {
  position: Position
  analysis: TechnicalAnalysis
  weight_pct: number | null
  action: Decision
  action_reason: string
}

export interface Account {
  currency: string
  total_assets: number
  cash: number
  market_value: number
  total_assets_usd: number | null
  cash_usd: number | null
  available_funds: number | null
  buying_power: number | null
  unrealized_pl: number | null
  realized_pl: number | null
  risk_level: string | null
  by_currency: Record<string, Record<string, number>>
}

export interface PortfolioRisk {
  base_currency: string
  total_value_base: number
  num_positions: number
  top_weights: { code: string; name: string; weight_pct: number }[]
  concentration_pct: number | null
  exposure_by_market: Record<string, number>
  exposure_by_currency: Record<string, number>
  winners: number
  losers: number
  target_return_pct: number
  benchmark: { code: string; return_pct: number | null; label: string } | null
  avg_score: number | null
  notes: string[]
}

export interface Performance {
  status: 'no_data' | 'building' | 'ready'
  days_tracked: number
  since?: string
  as_of?: string
  equity_usd?: number
  account_return_pct?: number
  spy_return_pct?: number
  excess_return_pct?: number
  beating_spy?: boolean
  beta?: number | null
  alpha_pct?: number | null
  tracking_error_pct?: number | null
  information_ratio?: number | null
  max_drawdown_pct?: number
  spy_max_drawdown_pct?: number
  message?: string
}

export interface MacroRegime {
  regime: string
  curve_10y2y: number | null
  hy_spread_pct: number | null
  vix: number | null
  flags: string[]
  implication: string
}

export interface PortfolioAnalysis {
  account: Account
  risk: PortfolioRisk
  timeframe: string
  holdings: HoldingAnalysis[]
  macro_regime: MacroRegime | null
  generated_at: string | null
}

export interface OptionLeg {
  action: string
  right: string
  strike: number
  expiry: string
  dte: number | null
  delta: number | null
  iv_pct: number | null
  price: number | null
  bid: number | null
  ask: number | null
  code: string | null
}

export interface OptionStrategy {
  name: string
  direction: string
  legs: OptionLeg[]
  tenor_dte: number | null
  rationale: string
  net_debit_credit: number | null
  max_profit: number | null
  max_loss: number | null
  breakeven: number | null
  suited_when: string
  take_profit: string | null
  stop_loss: string | null
  manage: string | null
  pop_pct: number | null
  ev_per_share: number | null
  net_delta: number | null
  net_theta: number | null
  net_vega: number | null
  warnings: string[]
  suggested_contracts: number | null
  capital_required_usd: number | null
}

export interface OptionsAnalysis {
  code: string
  name: string
  as_of: string | null
  spot: number | null
  technical_decision: Decision | null
  technical_score: number | null
  realized_vol_pct: number | null
  vol_estimator: string
  forecast_vol_pct: number | null
  iv_regime_basis: string
  atm_iv_pct: number | null
  iv_regime: string | null
  iv_vs_realized: number | null
  expiry_used: string | null
  dte: number | null
  holds_underlying: boolean
  shares_held: number
  analyst_consensus: AnalystConsensus | null
  earnings_date: string | null
  days_to_earnings: number | null
  skew_25d_pts: number | null
  next_expiry_used: string | null
  next_atm_iv_pct: number | null
  strategies: OptionStrategy[]
  notes: string[]
  error: string | null
}

export interface SearchResult {
  code: string
  name: string
  finnhub_symbol: string
  type: string
}

export interface AskResponse {
  available: boolean
  provider: string | null
  model: string | null
  cached: boolean
  answer: string
  message: string
}

export interface FundamentalMetrics {
  code: string
  name: string | null
  finnhub_symbol: string | null
  sector: string | null
  exchange: string | null
  market_cap_musd: number | null
  pe_ttm: number | null
  pb: number | null
  roe_pct: number | null
  roic_pct: number | null
  gross_margin_pct: number | null
  net_margin_pct: number | null
  operating_margin_pct: number | null
  revenue_growth_yoy_pct: number | null
  eps_growth_5y_pct: number | null
  debt_to_equity: number | null
  current_ratio: number | null
  beta: number | null
  week52_high: number | null
  week52_low: number | null
  available_fields: string[]
  missing_fields: string[]
  error: string | null
}

export interface ChartSeries {
  code: string
  name: string
  timeframe: string
  time: string[]
  open: number[]
  high: number[]
  low: number[]
  close: number[]
  volume: number[]
  ema20: (number | null)[]
  ema50: (number | null)[]
  ema200: (number | null)[]
  bb_upper: (number | null)[]
  bb_mid: (number | null)[]
  bb_lower: (number | null)[]
  rsi14: (number | null)[]
  error: string | null
}

export interface OptimizerAction {
  code: string
  name: string
  broker: string
  action: string // BUY | ADD | HOLD | TRIM | SELL
  decision: Decision
  score: number
  currency: string
  current_pct: number
  target_pct: number
  current_usd: number
  delta_usd: number
  est_shares: number | null
  last_price: number | null
  reason: string
  risk_contribution_pct: number | null
}

export interface OptimizerPlan {
  base_currency: string
  timeframe: string
  total_value_usd: number
  invested_usd: number
  cash_usd: number
  cash_target_pct: number
  concentration_cap_pct: number
  projected_top_pct: number | null
  buy_usd: number
  sell_usd: number
  actions: OptimizerAction[]
  notes: string[]
  generated_at: string | null
  method_used: string
  portfolio_vol_pct: number | null
  covariance_shrinkage: number | null
  risk_notes: string[]
}

export interface WatchlistGroup {
  name: string
  count: number | null
  source: 'moomoo' | 'local'
}

export interface WatchlistAnalysis {
  group: string
  items: TechnicalAnalysis[]
  generated_at: string | null
  errors: string[]
}
