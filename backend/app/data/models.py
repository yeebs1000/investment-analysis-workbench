"""Typed domain models shared across the analytics and API layers."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Decision(str, Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    ACCUMULATE = "ACCUMULATE"
    HOLD = "HOLD"
    REDUCE = "REDUCE"
    SELL = "SELL"

    @property
    def label(self) -> str:
        return {
            "STRONG_BUY": "Strong Buy",
            "BUY": "Buy",
            "ACCUMULATE": "Accumulate",
            "HOLD": "Hold",
            "REDUCE": "Reduce / Trim",
            "SELL": "Sell",
        }[self.value]


class Account(BaseModel):
    currency: str = "HKD"
    total_assets: float = 0.0
    cash: float = 0.0
    market_value: float = 0.0
    total_assets_usd: float | None = None   # FX-approx total in USD (for display)
    cash_usd: float | None = None
    available_funds: float | None = None
    buying_power: float | None = None
    unrealized_pl: float | None = None
    realized_pl: float | None = None
    risk_level: str | None = None
    # native cash/asset balances per currency, e.g. {"USD": {"cash": .., "assets": ..}}
    by_currency: dict[str, dict[str, float]] = Field(default_factory=dict)


class Position(BaseModel):
    code: str
    name: str
    market: str
    currency: str
    broker: str = "moomoo"      # "moomoo" | "ibkr" | "moomoo+ibkr" (held in both)
    side: str = "LONG"
    qty: float = 0.0
    cost_price: float | None = None
    last_price: float | None = None
    market_value: float = 0.0
    pl_ratio_pct: float | None = None   # percent, e.g. 3.02 == +3.02%
    pl_value: float | None = None
    today_pl_value: float | None = None


class SignalComponent(BaseModel):
    """One analyst dimension (trend/momentum/volatility/volume/levels)."""
    name: str
    score: float            # normalized to [-1, +1]
    weight: float           # contribution weight in the blend
    summary: str            # one-line plain-English read
    reasons: list[str] = Field(default_factory=list)   # number-backed bullets, tagged (+)/(-)
    metrics: dict[str, float | None] = Field(default_factory=dict)


class MLSignal(BaseModel):
    """Carrier for the optional ML forecast, passed into technical.analyze().

    Decouples technical.py from the optional app.ml package -- if no model has
    been trained (or the ML deps aren't installed), this is simply None and the
    engine behaves exactly as it did before ML existed."""
    score: float             # normalized to [-1, +1], 2*probability - 1
    probability: float       # raw model output: P(beats cross-sectional median fwd return)
    reliability: float       # 0-1, sample-size/fold-count discounted -- see app/ml/train.py
    reasons: list[str] = Field(default_factory=list)


class TechnicalAnalysis(BaseModel):
    code: str
    name: str
    as_of: str | None = None
    market: str | None = None
    currency: str | None = None
    timeframe: str = "day"
    price: float | None = None
    score: float            # 0-100, 50 = neutral
    decision: Decision
    confidence: float       # 0-1
    confidence_label: str   # Low / Medium / High
    higher_tf: str | None = None          # the corroborating higher timeframe (e.g. "week")
    higher_tf_trend: float | None = None  # its trend score, [-1, 1]
    higher_tf_summary: str | None = None  # plain read of the higher-tf trend
    components: list[SignalComponent] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)        # top aggregated bullets
    stop: float | None = None        # ATR-based suggested stop (for a long)
    target: float | None = None      # ATR-based suggested target
    atr_pct: float | None = None
    reward_risk: float | None = None         # (target-price)/(price-stop)
    kelly_fraction: float | None = None       # full-Kelly edge estimate [-1, 1]
    kelly_sizing_pct: float | None = None     # suggested book weight (half-Kelly, capped)
    rel_strength_pct: float | None = None     # excess return vs benchmark over window
    beta: float | None = None
    alpha_pct: float | None = None            # annualized alpha vs benchmark
    analyst_consensus: dict | None = None     # Finnhub analyst rating distribution
    earnings_surprise: dict | None = None     # recent EPS beats/misses + PEAD read (context, not scored)
    insider: dict | None = None               # net insider buying/selling (context, not scored)
    fundamental_quality: dict | None = None   # Buffett/Munger quality score, SEPARATE from technical score
    growth_tilt: dict | None = None           # size/growth-stage conviction tilt (underweight mega, overweight small growers)
    entry_risk: dict | None = None            # categorical chase/capitulation flag; None = normal behaviour
    verdict: dict | None = None               # two-axis read: business quality x entry timing (never conflated)
    next_earnings: dict | None = None         # next confirmed earnings date (None = unknown, NOT "none coming")
    order_book: dict | None = None            # IBKR Level-2 depth snapshot (context, not scored)
    risk_alerts: list[str] = Field(default_factory=list)  # one consolidated list of every active risk flag
    indicators: dict[str, float | None] = Field(default_factory=dict)
    bars_used: int = 0
    error: str | None = None         # set when data was insufficient/unavailable


class HoldingAnalysis(BaseModel):
    position: Position
    analysis: TechnicalAnalysis
    weight_pct: float | None = None      # share of portfolio (approx, FX-normalized)
    action: Decision                      # position-aware action (held context)
    action_reason: str = ""


class PortfolioRisk(BaseModel):
    base_currency: str = "USD"
    total_value_base: float = 0.0
    num_positions: int = 0
    top_weights: list[dict[str, float | str]] = Field(default_factory=list)
    concentration_pct: float | None = None   # weight of largest position
    exposure_by_market: dict[str, float] = Field(default_factory=dict)
    exposure_by_currency: dict[str, float] = Field(default_factory=dict)
    winners: int = 0
    losers: int = 0
    target_return_pct: float = 12.5          # fund mandate: 10-15%/yr
    benchmark: dict | None = None            # {code, return_1y_pct, label}
    avg_score: float | None = None           # mean technical score across the book
    notes: list[str] = Field(default_factory=list)


class PortfolioAnalysis(BaseModel):
    account: Account
    risk: PortfolioRisk
    timeframe: str = "day"
    holdings: list[HoldingAnalysis] = Field(default_factory=list)
    macro_regime: dict | None = None      # FRED regime read (yield curve / credit / VIX)
    generated_at: str | None = None


class OptionLeg(BaseModel):
    action: str              # "Buy" or "Sell"
    right: str               # "Call" or "Put"
    strike: float
    expiry: str
    dte: int | None = None
    delta: float | None = None
    iv_pct: float | None = None      # implied vol, percent
    price: float | None = None       # mark premium per share (bid/ask mid; falls back to last trade)
    bid: float | None = None
    ask: float | None = None
    oi: float | None = None          # open interest (None = feed didn't supply it)
    code: str | None = None


class OptionStrategy(BaseModel):
    name: str
    direction: str           # Bullish / Bearish / Neutral / Income
    legs: list[OptionLeg] = Field(default_factory=list)
    tenor_dte: int | None = None
    rationale: str = ""
    net_debit_credit: float | None = None   # per share; +credit / -debit
    max_profit: float | None = None
    max_loss: float | None = None
    breakeven: float | None = None
    suited_when: str = ""
    take_profit: str | None = None           # profit-taking rule for the setup
    stop_loss: str | None = None             # loss-management rule
    manage: str | None = None                # roll / adjustment guidance
    pop_pct: float | None = None             # probability of profit at expiry, 0-100
    ev_per_share: float | None = None        # probability-weighted P&L/share at expiry (same lognormal model as POP)
    net_delta: float | None = None           # position Greeks (native BSM, sum across legs)
    net_theta: float | None = None
    net_vega: float | None = None
    warnings: list[str] = Field(default_factory=list)   # event-risk / liquidity flags for THIS structure
    suggested_contracts: int | None = None   # sized so max loss fits the per-trade risk budget
    capital_required_usd: float | None = None  # cash/collateral needed for suggested_contracts


class OptionsAnalysis(BaseModel):
    code: str
    name: str
    as_of: str | None = None
    spot: float | None = None
    technical_decision: Decision | None = None
    technical_score: float | None = None
    realized_vol_pct: float | None = None    # annualized, percent (trailing Yang-Zhang)
    vol_estimator: str = "close_to_close"    # "yang_zhang" | "close_to_close" (fallback used)
    forecast_vol_pct: float | None = None    # GARCH(1,1) forecast over the option tenor, annualized %
    iv_regime_basis: str = "realized"        # "garch_forecast" | "realized" (what the regime compared IV to)
    atm_iv_pct: float | None = None
    iv_regime: str | None = None             # Elevated / Normal / Cheap
    iv_vs_realized: float | None = None       # ratio (IV / regime-basis vol)
    expiry_used: str | None = None
    dte: int | None = None
    holds_underlying: bool = False
    shares_held: float = 0.0
    analyst_consensus: dict | None = None
    earnings_date: str | None = None          # next confirmed earnings (None = unknown, NOT "none coming")
    days_to_earnings: int | None = None
    skew_25d_pts: float | None = None         # 25-delta put IV minus 25-delta call IV, vol points
    next_expiry_used: str | None = None       # second tenor sampled for term structure
    next_atm_iv_pct: float | None = None
    strategies: list[OptionStrategy] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    error: str | None = None


class FundamentalMetrics(BaseModel):
    """Curated value-investing metrics from Finnhub (free tier). Coverage
    skews heavily to US-listed names -- `missing_fields` is always populated
    honestly rather than silently omitted, so the UI/LLM never implies data
    that isn't actually there."""
    code: str
    name: str | None = None
    finnhub_symbol: str | None = None
    sector: str | None = None
    exchange: str | None = None
    market_cap_musd: float | None = None
    pe_ttm: float | None = None
    pb: float | None = None
    roe_pct: float | None = None
    roic_pct: float | None = None
    gross_margin_pct: float | None = None
    net_margin_pct: float | None = None
    operating_margin_pct: float | None = None
    revenue_growth_yoy_pct: float | None = None
    eps_growth_5y_pct: float | None = None
    debt_to_equity: float | None = None
    current_ratio: float | None = None
    beta: float | None = None
    week52_high: float | None = None
    week52_low: float | None = None
    available_fields: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)
    error: str | None = None


class SearchResult(BaseModel):
    code: str               # Moomoo MARKET.CODE
    name: str
    finnhub_symbol: str = ""
    type: str = ""


class AskResponse(BaseModel):
    available: bool
    provider: str | None = None
    model: str | None = None
    cached: bool = False
    answer: str = ""
    message: str = ""


class ChartSeries(BaseModel):
    code: str
    name: str
    timeframe: str = "day"
    time: list[str] = Field(default_factory=list)
    open: list[float] = Field(default_factory=list)
    high: list[float] = Field(default_factory=list)
    low: list[float] = Field(default_factory=list)
    close: list[float] = Field(default_factory=list)
    volume: list[float] = Field(default_factory=list)
    ema20: list[float | None] = Field(default_factory=list)
    ema50: list[float | None] = Field(default_factory=list)
    ema200: list[float | None] = Field(default_factory=list)
    bb_upper: list[float | None] = Field(default_factory=list)
    bb_mid: list[float | None] = Field(default_factory=list)
    bb_lower: list[float | None] = Field(default_factory=list)
    rsi14: list[float | None] = Field(default_factory=list)
    error: str | None = None


class OptimizerAction(BaseModel):
    code: str
    name: str
    broker: str = "moomoo"
    action: str                      # BUY | ADD | HOLD | TRIM | SELL
    decision: Decision               # the underlying technical decision
    score: float
    currency: str = "USD"
    current_pct: float = 0.0         # current weight of the book
    target_pct: float = 0.0          # recommended weight
    current_usd: float = 0.0
    delta_usd: float = 0.0           # +buy / -sell, in USD
    est_shares: float | None = None  # approx share count to trade (native price)
    last_price: float | None = None
    reason: str = ""
    risk_contribution_pct: float | None = None  # share of portfolio variance (risk_aware only)


class OptimizerPlan(BaseModel):
    base_currency: str = "USD"
    timeframe: str = "day"
    total_value_usd: float = 0.0
    invested_usd: float = 0.0
    cash_usd: float = 0.0
    cash_target_pct: float = 5.0
    concentration_cap_pct: float = 15.0
    projected_top_pct: float | None = None
    buy_usd: float = 0.0
    sell_usd: float = 0.0
    actions: list[OptimizerAction] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    generated_at: str | None = None
    method_used: str = "heuristic"                # heuristic | risk_aware -- what actually ran
    portfolio_vol_pct: float | None = None         # annualized book vol (risk_aware only)
    covariance_shrinkage: float | None = None      # Ledoit-Wolf shrinkage delta, 0-1 (risk_aware only)
    risk_notes: list[str] = Field(default_factory=list)  # caveats specific to the risk model


class WatchlistGroup(BaseModel):
    name: str
    count: int | None = None
    source: str = "moomoo"   # "moomoo" (broker-side) or "local" (app-owned JSON)


class WatchlistAnalysis(BaseModel):
    group: str
    items: list[TechnicalAnalysis] = Field(default_factory=list)
    generated_at: str | None = None
    errors: list[str] = Field(default_factory=list)
