"""Builds compact JSON payloads from the deterministic analysis and asks the
selected LLM to narrate them. Only pre-computed numbers are passed in."""
from __future__ import annotations

from pydantic import BaseModel

from app.data.models import AskResponse, FundamentalMetrics, OptionsAnalysis, PortfolioAnalysis, TechnicalAnalysis
from app.llm import prompts
from app.llm.router import router


class Narrative(BaseModel):
    available: bool
    provider: str | None = None
    model: str | None = None
    cached: bool = False
    text: str = ""
    message: str = ""   # set when unavailable (e.g. no API key)


# A curated indicator subset keeps the prompt compact so thinking models answer
# fast and reliably (the full ~22-field dict bloats the context and stalls them).
_KEY_INDICATORS = (
    "rsi", "rsi2", "macd", "macd_signal", "ema20", "ema50", "ema200",
    "sma50", "sma200", "adx", "pct_b", "atr", "volume_ratio",
    "ts_momentum_pct", "donchian_high", "donchian_low", "rsi_divergence",
    "sharpe", "max_drawdown_pct", "support_20d", "resistance_20d",
)


def _symbol_payload(ta: TechnicalAnalysis) -> dict:
    return {
        "symbol": ta.code,
        "name": ta.name,
        "analysis_timeframe": ta.timeframe,
        "higher_timeframe": ta.higher_tf,
        "higher_timeframe_read": ta.higher_tf_summary,
        "price": ta.price,
        "currency": ta.currency,
        "decision": ta.decision.value,
        "score_0_100": ta.score,
        "confidence": ta.confidence_label,
        "suggested_stop": ta.stop,
        "suggested_target": ta.target,
        "atr_pct": ta.atr_pct,
        "reward_risk": ta.reward_risk,
        "kelly_sizing_pct": ta.kelly_sizing_pct,
        "rel_strength_vs_spy_pct": ta.rel_strength_pct,
        "beta": ta.beta,
        "alpha_pct": ta.alpha_pct,
        "analyst_consensus": ta.analyst_consensus,
        "earnings_surprise": ta.earnings_surprise,
        "insider_sentiment": ta.insider,
        "fundamental_quality": ta.fundamental_quality,
        "size_growth_tilt": ta.growth_tilt,
        "entry_risk": ta.entry_risk,
        "verdict_two_axis": ta.verdict,
        "next_earnings": ta.next_earnings,
        "order_book_level2": ta.order_book,
        "risk_alerts": ta.risk_alerts,
        "analysts": [
            {
                "dimension": c.name,
                "score_-1_to_1": round(c.score, 2),
                "reasons": c.reasons[:3],   # top number-backed drivers only
            }
            for c in ta.components
        ],
        "key_indicators": {k: ta.indicators.get(k) for k in _KEY_INDICATORS},
    }


def _portfolio_payload(pa: PortfolioAnalysis) -> dict:
    return {
        "base_currency": pa.risk.base_currency,
        "approx_value": pa.risk.total_value_base,
        "positions": pa.risk.num_positions,
        "winners": pa.risk.winners,
        "losers": pa.risk.losers,
        "concentration_pct": pa.risk.concentration_pct,
        "exposure_by_market_pct": pa.risk.exposure_by_market,
        "risk_notes": pa.risk.notes,
        "holdings": [
            {
                "symbol": h.position.code,
                "name": h.position.name,
                "weight_pct": h.weight_pct,
                "pl_pct": h.position.pl_ratio_pct,
                "action": h.action.value,
                "technical_decision": h.analysis.decision.value,
                "score_0_100": h.analysis.score,
            }
            for h in pa.holdings
        ],
    }


def _unavailable(requested: str | None) -> Narrative:
    eff = router.resolve(requested)
    status = router.providers_status()
    if eff == "none" and (requested or "").lower() not in ("none", "deterministic", ""):
        msg = (
            f"'{requested}' is not configured - add its API key to backend/.env. "
            f"Available: {[k for k, v in status['available'].items() if v] or 'none'}."
        )
    else:
        msg = "Deterministic mode - no LLM call made (free)."
    return Narrative(available=False, message=msg)


def explain_symbol(ta: TechnicalAnalysis, provider: str | None) -> Narrative:
    if ta.error:
        return Narrative(available=False, message=ta.error)
    result = router.narrate(
        provider, prompts.ANALYST_SYSTEM,
        prompts.symbol_user_message(_symbol_payload(ta)), max_tokens=1100,
    )
    if result is None:
        return _unavailable(provider)
    return Narrative(
        available=True, provider=result.provider, model=result.model,
        cached=result.cached, text=result.text,
    )


def _options_payload(oa: OptionsAnalysis) -> dict:
    return {
        "symbol": oa.code,
        "name": oa.name,
        "spot": oa.spot,
        "technical_view": oa.technical_decision.value if oa.technical_decision else None,
        "iv_regime": oa.iv_regime,
        "iv_regime_basis": oa.iv_regime_basis,
        "atm_iv_pct": oa.atm_iv_pct,
        "realized_vol_pct": oa.realized_vol_pct,
        "forecast_vol_pct": oa.forecast_vol_pct,
        "skew_25d_pts": oa.skew_25d_pts,
        "expiry": oa.expiry_used,
        "dte": oa.dte,
        "earnings_date": oa.earnings_date,
        "days_to_earnings": oa.days_to_earnings,
        "holds_shares": oa.shares_held,
        "strategies": [
            {
                "name": s.name,
                "direction": s.direction,
                "legs": [
                    {"action": l.action, "right": l.right, "strike": l.strike, "delta": l.delta}
                    for l in s.legs
                ],
                "net_credit_or_debit": s.net_debit_credit,
                "max_profit": s.max_profit,
                "max_loss": s.max_loss,
                "breakeven": s.breakeven,
                "pop_pct": s.pop_pct,
                "ev_per_share": s.ev_per_share,
                "net_delta": s.net_delta,
                "net_theta": s.net_theta,
                "net_vega": s.net_vega,
                "suggested_contracts": s.suggested_contracts,
                "capital_required_usd": s.capital_required_usd,
                "warnings": s.warnings,
                "suited_when": s.suited_when,
                "take_profit": s.take_profit,
                "stop_loss": s.stop_loss,
                "manage": s.manage,
            }
            for s in oa.strategies
        ],
        "analyst_consensus": oa.analyst_consensus,
        "notes": oa.notes,
    }


def explain_options(oa: OptionsAnalysis, provider: str | None) -> Narrative:
    if oa.error:
        return Narrative(available=False, message=oa.error)
    result = router.narrate(
        provider, prompts.OPTIONS_SYSTEM,
        prompts.options_user_message(_options_payload(oa)), max_tokens=900,
    )
    if result is None:
        return _unavailable(provider)
    return Narrative(
        available=True, provider=result.provider, model=result.model,
        cached=result.cached, text=result.text,
    )


def _ask_unavailable(requested: str | None) -> AskResponse:
    n = _unavailable(requested)
    return AskResponse(available=False, message=n.message)


def ask_symbol(ta: TechnicalAnalysis, question: str, provider: str | None) -> AskResponse:
    if ta.error:
        return AskResponse(available=False, message=ta.error)
    # report-format answers (headed sections + a snapshot table) need headroom
    # beyond the tight-brief prompts; the prompt itself caps length at ~400 words
    result = router.narrate(
        provider, prompts.ASK_SYSTEM,
        prompts.ask_user_message(_symbol_payload(ta), question), max_tokens=1400,
    )
    if result is None:
        return _ask_unavailable(provider)
    return AskResponse(available=True, provider=result.provider, model=result.model,
                       cached=result.cached, answer=result.text)


def ask_options(oa: OptionsAnalysis, question: str, provider: str | None) -> AskResponse:
    if oa.error:
        return AskResponse(available=False, message=oa.error)
    result = router.narrate(
        provider, prompts.ASK_SYSTEM,
        prompts.ask_user_message(_options_payload(oa), question), max_tokens=1400,
    )
    if result is None:
        return _ask_unavailable(provider)
    return AskResponse(available=True, provider=result.provider, model=result.model,
                       cached=result.cached, answer=result.text)


def _fundamentals_payload(fm: FundamentalMetrics) -> dict:
    payload = fm.model_dump(exclude={"error"})
    return payload


def ask_fundamentals(fm: FundamentalMetrics, question: str, provider: str | None) -> AskResponse:
    if fm.error:
        return AskResponse(available=False, message=fm.error)
    result = router.narrate(
        provider, prompts.FUNDAMENTAL_ASK_SYSTEM,
        prompts.ask_user_message(_fundamentals_payload(fm), question), max_tokens=700,
    )
    if result is None:
        return _ask_unavailable(provider)
    return AskResponse(available=True, provider=result.provider, model=result.model,
                       cached=result.cached, answer=result.text)


def explain_portfolio(pa: PortfolioAnalysis, provider: str | None) -> Narrative:
    result = router.narrate(
        provider,
        prompts.PORTFOLIO_SYSTEM,
        prompts.portfolio_user_message(_portfolio_payload(pa)),
        max_tokens=800,
    )
    if result is None:
        return _unavailable(provider)
    return Narrative(
        available=True, provider=result.provider, model=result.model,
        cached=result.cached, text=result.text,
    )
