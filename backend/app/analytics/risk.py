"""PortfolioRiskAnalyst: turns positions + per-name technical reads into
position-aware actions and portfolio-level risk context (FX-normalized)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from app.analytics.indicators import log_returns
from app.data.models import (
    Account,
    Decision,
    HoldingAnalysis,
    OptimizerAction,
    OptimizerPlan,
    PortfolioAnalysis,
    PortfolioRisk,
    Position,
    TechnicalAnalysis,
)
from app.data.normalize import fx_to_usd, is_option_code

# How a standalone technical decision maps to an action when you ALREADY hold it.
_HELD_ACTION = {
    Decision.STRONG_BUY: Decision.ACCUMULATE,
    Decision.BUY: Decision.ACCUMULATE,
    Decision.ACCUMULATE: Decision.ACCUMULATE,
    Decision.HOLD: Decision.HOLD,
    Decision.REDUCE: Decision.REDUCE,
    Decision.SELL: Decision.SELL,
}

CONCENTRATION_LIMIT_PCT = 15.0   # default single-name weight cap (overridable per request
                                  # via /api/optimize?cap_pct= for deliberate core overweights)
DEFAULT_CASH_TARGET_PCT = 5.0    # default idle-cash buffer the optimizer plans around
SCORE_TILT_MIDLINE = 46.0        # score above this earns weight; at/below -> zero raw weight
                                  # (46 = the HOLD midline of the 0-100 scoring scale)
REDUCE_WEIGHT_MULT = 0.4         # REDUCE-rated names keep this fraction of their score weight
WINNER_LOCK_GAINS_PL_PCT = 25.0  # reduce/sell signal on a winner up this much -> "lock gains" nudge
LOSER_CUT_PL_PCT = -15.0         # sell signal on a loser down this much -> "cut if thesis broken"
MARKET_CONCENTRATION_NOTE_PCT = 70.0  # single-market exposure above this gets a diversification note
MIN_TRADE_FLOOR_FRAC = 0.02      # ignore rebalance trades under 2% of book...
MIN_TRADE_FLOOR_USD = 100.0      # ...or under $100, whichever is larger
CASH_DEPLOY_NUDGE_PCT = 2.0      # idle cash > target+this -> "deploy" note

# --- risk-aware optimizer mode: gating thresholds + tuning constants --------
MIN_HOLDINGS_FOR_RISK_MODEL = 7     # below this, correlation structure isn't meaningfully
                                     # estimable -- fall back, always. Also the smallest count
                                     # that can still fully deploy capital under the default
                                     # 15% cap (7 * ~15.8% investable-adjusted >= 100%); fewer
                                     # names would pass this gate but then always hit the
                                     # cap-infeasibility fallback below, which is confusing.
MIN_BARS_PER_HOLDING = 60           # ~3 months; shorter history excludes a name
                                     # from the covariance estimate (not the plan)
CORR_PENALTY_STRENGTH = 1.0         # how hard avg-correlation-to-book discounts a weight
SHRINKAGE_CAVEAT_THRESHOLD = 0.4    # Ledoit-Wolf delta above this -> "noisy" note
SANITY_DIFF_THRESHOLD_PCT = 5.0     # heuristic-vs-risk-aware disagreement worth flagging
PERIODS_PER_YEAR = 252.0
_FLOOR_VOL = 0.02                   # 2% annualized -- avoids inverse-vol blowup near-zero


def _value_usd(p: Position) -> float:
    return abs(p.market_value) * fx_to_usd(p.currency)


def position_action(
    ta: TechnicalAnalysis, position: Position, weight_pct: float
) -> tuple[Decision, str]:
    """Decide buy/sell/accumulate/hold/reduce for a held position."""
    base = _HELD_ACTION.get(ta.decision, Decision.HOLD)
    reason_bits: list[str] = [f"signal {ta.decision.label.lower()} (score {ta.score:.0f}/100)"]

    # Don't add to an already-oversized position.
    if base == Decision.ACCUMULATE and weight_pct >= CONCENTRATION_LIMIT_PCT:
        base = Decision.HOLD
        reason_bits.append(f"already {weight_pct:.0f}% of book — hold rather than add")

    # Lock gains: a reduce/sell signal on a strong winner -> trim.
    pl = position.pl_ratio_pct or 0.0
    if base in (Decision.REDUCE, Decision.SELL) and pl > WINNER_LOCK_GAINS_PL_PCT:
        reason_bits.append(f"up {pl:.0f}% — consider locking gains")
    if base == Decision.SELL and pl < LOSER_CUT_PL_PCT:
        reason_bits.append(f"down {pl:.0f}% — cut if thesis broken")

    return base, "; ".join(reason_bits)


def analyze_portfolio(
    account: Account,
    positions: list[Position],
    analyses: dict[str, TechnicalAnalysis],
    base_currency: str = "USD",
) -> PortfolioAnalysis:
    total_usd = sum(_value_usd(p) for p in positions) or 1.0

    holdings: list[HoldingAnalysis] = []
    exposure_mkt: dict[str, float] = {}
    exposure_ccy: dict[str, float] = {}
    winners = losers = 0

    for p in positions:
        w = _value_usd(p) / total_usd * 100.0
        exposure_mkt[p.market] = exposure_mkt.get(p.market, 0.0) + w
        exposure_ccy[p.currency] = exposure_ccy.get(p.currency, 0.0) + w
        if (p.pl_value or 0) > 0:
            winners += 1
        elif (p.pl_value or 0) < 0:
            losers += 1

        ta = analyses.get(p.code) or TechnicalAnalysis(
            code=p.code, name=p.name, score=50.0, decision=Decision.HOLD,
            confidence=0.0, confidence_label="Low", error="No analysis available.",
        )
        action, action_reason = position_action(ta, p, w)
        holdings.append(HoldingAnalysis(
            position=p, analysis=ta, weight_pct=round(w, 2),
            action=action, action_reason=action_reason,
        ))

    holdings.sort(key=lambda h: h.weight_pct or 0, reverse=True)
    top_weights = [
        {"code": h.position.code, "name": h.position.name, "weight_pct": h.weight_pct or 0.0}
        for h in holdings[:5]
    ]
    concentration = holdings[0].weight_pct if holdings else None

    notes: list[str] = []
    if concentration and concentration >= CONCENTRATION_LIMIT_PCT:
        notes.append(
            f"Concentration risk: {holdings[0].position.name} is "
            f"{concentration:.0f}% of the book (> {CONCENTRATION_LIMIT_PCT:.0f}% guide)."
        )
    biggest_mkt = max(exposure_mkt.items(), key=lambda kv: kv[1], default=None)
    if biggest_mkt and biggest_mkt[1] >= MARKET_CONCENTRATION_NOTE_PCT:
        notes.append(f"{biggest_mkt[1]:.0f}% of the book sits in {biggest_mkt[0]} — limited diversification.")
    sells = [h for h in holdings if h.action in (Decision.SELL, Decision.REDUCE)]
    if sells:
        notes.append("Trim/exit candidates: " + ", ".join(h.position.code for h in sells[:6]) + ".")
    adds = [h for h in holdings if h.action == Decision.ACCUMULATE]
    if adds:
        notes.append("Add candidates among holdings: " + ", ".join(h.position.code for h in adds[:6]) + ".")

    risk = PortfolioRisk(
        base_currency=base_currency,
        total_value_base=round(total_usd, 2),
        num_positions=len(positions),
        top_weights=top_weights,
        concentration_pct=round(concentration, 2) if concentration else None,
        exposure_by_market={k: round(v, 1) for k, v in sorted(exposure_mkt.items(), key=lambda x: -x[1])},
        exposure_by_currency={k: round(v, 1) for k, v in sorted(exposure_ccy.items(), key=lambda x: -x[1])},
        winners=winners, losers=losers, notes=notes,
    )
    return PortfolioAnalysis(account=account, risk=risk, holdings=holdings)


def rank_watchlist(analyses: list[TechnicalAnalysis]) -> list[TechnicalAnalysis]:
    """Best buy candidates first; un-analyzable names sink to the bottom."""
    return sorted(analyses, key=lambda a: (a.error is not None, -a.score))


def _cap_weights(targets: dict[str, float], cap_usd: float, total_target: float) -> dict[str, float]:
    """Clamp each target to cap_usd, redistributing the excess proportionally to
    names still below the cap. A few passes converge for realistic books."""
    t = dict(targets)
    for _ in range(5):
        over = {k: v for k, v in t.items() if v > cap_usd + 1e-6}
        if not over:
            break
        excess = sum(v - cap_usd for v in over.values())
        for k in over:
            t[k] = cap_usd
        room = {k: cap_usd - v for k, v in t.items() if v < cap_usd - 1e-6}
        room_sum = sum(room.values())
        if room_sum <= 0:
            break
        for k, r in room.items():
            t[k] += excess * (r / room_sum)
    return t


def optimize_portfolio(
    pa: PortfolioAnalysis,
    cash_usd: float = 0.0,
    cap_pct: float = CONCENTRATION_LIMIT_PCT,
    cash_target_pct: float = DEFAULT_CASH_TARGET_PCT,
    now_iso: str | None = None,
    method: str = "heuristic",
    bars_by_code: dict[str, pd.DataFrame] | None = None,
) -> OptimizerPlan:
    """Turn positions + signals into concrete BUY/ADD/TRIM/SELL actions.

    method="heuristic" (default, unchanged): tilt the book toward
    higher-scoring names, exit weak ones, respect a single-name concentration
    cap, and deploy idle cash.

    method="risk_aware": additionally scales each name's weight by inverse
    volatility and a correlation penalty (Ledoit-Wolf shrunk covariance across
    `bars_by_code`), solved subject to the same cap/no-short constraints. Never
    replaces the heuristic silently -- on any gating failure or solver
    non-convergence it falls back to the heuristic path and says why in
    `plan.risk_notes` / `plan.method_used`.
    """
    holdings = pa.holdings
    invested = sum(_value_usd(h.position) for h in holdings)
    cash = max(cash_usd or 0.0, 0.0)
    total = invested + cash
    plan = OptimizerPlan(
        total_value_usd=round(total, 2), invested_usd=round(invested, 2),
        cash_usd=round(cash, 2), cash_target_pct=cash_target_pct,
        concentration_cap_pct=cap_pct, generated_at=now_iso,
    )
    if total <= 0 or not holdings:
        plan.notes.append("No positions to optimise.")
        return plan

    # 0) Option-contract positions (e.g. a held call) are outside a stock-weight
    #    optimizer's scope: their value stays where it is (HOLD, sized/managed via
    #    the options strategist), and their capital is excluded from the pool the
    #    optimizer re-allocates -- otherwise equity technicals on the option code
    #    would produce BUY/SELL calls on a decaying derivative.
    option_codes = {h.position.code for h in holdings if is_option_code(h.position.code)}
    option_usd = sum(_value_usd(h.position) for h in holdings if h.position.code in option_codes)

    # 1) Raw quality weight per name: grows with score above the HOLD midline;
    #    bearish names are zeroed (exit) or shrunk (reduce). Shared by both methods.
    raw: dict[str, float] = {}
    for h in holdings:
        if h.position.code in option_codes:
            continue
        s = h.analysis.score
        q = max(0.0, s - SCORE_TILT_MIDLINE)
        if h.analysis.decision == Decision.SELL:
            q = 0.0
        elif h.analysis.decision == Decision.REDUCE:
            q *= REDUCE_WEIGHT_MULT
        raw[h.position.code] = q
    sum_raw = sum(raw.values())

    investable = max(0.0, total * (1.0 - cash_target_pct / 100.0) - option_usd)
    cap_usd = total * cap_pct / 100.0

    heuristic_targets: dict[str, float]
    if sum_raw > 0:
        heuristic_targets = {c: investable * (q / sum_raw) for c, q in raw.items()}
        heuristic_targets = _cap_weights(heuristic_targets, cap_usd, investable)
    else:
        heuristic_targets = {c: 0.0 for c in raw}  # everything bearish -> go to cash

    targets = heuristic_targets
    method_used = "heuristic"
    risk_notes: list[str] = []
    portfolio_vol_pct: float | None = None
    covariance_shrinkage: float | None = None
    risk_contrib: dict[str, float] = {}

    if method == "risk_aware":
        if sum_raw <= 0:
            risk_notes.append(
                "Every holding is bearish-rated, so there's nothing to risk-optimise "
                "— showing the cash-out plan."
            )
        else:
            try:
                result = _risk_aware_targets(
                    holdings=holdings, raw=raw, cap_usd=cap_usd, investable=investable,
                    bars_by_code=bars_by_code or {},
                )
                risk_targets, r_notes, r_vol, r_shrink, r_contrib = result
                risk_notes = r_notes
                if risk_targets is not None:
                    targets = risk_targets
                    method_used = "risk_aware"
                    portfolio_vol_pct = r_vol
                    covariance_shrinkage = r_shrink
                    risk_contrib = r_contrib
            except Exception as exc:  # noqa: BLE001 - never let the risk overlay break the endpoint
                risk_notes = [
                    f"Risk-aware allocation failed unexpectedly ({exc}) — showing the "
                    f"score-based plan instead."
                ]

        # Sanity-check diff vs the heuristic -- only meaningful once risk-aware ran.
        if method_used == "risk_aware":
            max_diff = 0.0
            for c in raw:
                h_pct = heuristic_targets.get(c, 0.0) / total * 100.0
                r_pct = targets.get(c, 0.0) / total * 100.0
                max_diff = max(max_diff, abs(h_pct - r_pct))
            if max_diff > SANITY_DIFF_THRESHOLD_PCT:
                risk_notes.append(
                    f"Risk-aware and score-based weights disagree by up to {max_diff:.0f} "
                    f"percentage points on individual names — when models diverge this "
                    f"much, treat both as estimates and size conservatively."
                )

    # 2) Turn target vs current into actions. Ignore trades below a noise floor.
    floor = max(total * MIN_TRADE_FLOOR_FRAC, MIN_TRADE_FLOOR_USD)
    actions: list[OptimizerAction] = []
    buy_usd = sell_usd = 0.0
    for h in holdings:
        p = h.position
        cur = _value_usd(p)
        cur_pct = cur / total * 100.0
        if p.code in option_codes:
            actions.append(OptimizerAction(
                code=p.code, name=p.name, broker=p.broker, action="HOLD",
                decision=Decision.HOLD, score=round(h.analysis.score, 1),
                currency=p.currency, current_pct=round(cur_pct, 1), target_pct=round(cur_pct, 1),
                current_usd=round(cur, 2), delta_usd=0.0, est_shares=None,
                last_price=p.last_price,
                reason="option contract — outside the stock optimiser's scope; manage via the options strategist (expiry/theta, not equity weights)",
            ))
            continue
        tgt = targets.get(p.code, 0.0)
        delta = tgt - cur
        tgt_pct = tgt / total * 100.0
        px_usd = (p.last_price or 0.0) * fx_to_usd(p.currency)
        shares = (delta / px_usd) if px_usd > 0 else None

        if abs(delta) < floor:
            act = "HOLD"
        elif tgt <= 1e-6:
            act = "SELL"
        elif delta > 0:
            act = "ADD" if cur > floor else "BUY"
        else:
            act = "TRIM"

        if act in ("ADD", "BUY"):
            buy_usd += delta
        elif act in ("TRIM", "SELL"):
            sell_usd += -delta

        actions.append(OptimizerAction(
            code=p.code, name=p.name, broker=p.broker, action=act,
            decision=h.analysis.decision, score=round(h.analysis.score, 1),
            currency=p.currency, current_pct=round(cur_pct, 1), target_pct=round(tgt_pct, 1),
            current_usd=round(cur, 2), delta_usd=round(delta, 2),
            est_shares=round(shares, 1) if shares is not None else None,
            last_price=p.last_price, reason=_opt_reason(h, act, cur_pct, tgt_pct, cap_pct),
            risk_contribution_pct=round(risk_contrib[p.code], 1) if p.code in risk_contrib else None,
        ))

    # order: biggest sells first, then biggest buys, holds last
    order = {"SELL": 0, "TRIM": 1, "BUY": 2, "ADD": 3, "HOLD": 4}
    actions.sort(key=lambda a: (order.get(a.action, 9), -abs(a.delta_usd)))

    plan.actions = actions
    plan.buy_usd = round(buy_usd, 2)
    plan.sell_usd = round(sell_usd, 2)
    plan.projected_top_pct = round(max((a.target_pct for a in actions), default=0.0), 1)
    plan.method_used = method_used
    plan.portfolio_vol_pct = round(portfolio_vol_pct, 1) if portfolio_vol_pct is not None else None
    plan.covariance_shrinkage = round(covariance_shrinkage, 2) if covariance_shrinkage is not None else None
    plan.risk_notes = risk_notes

    # 3) Plain-English summary notes.
    sells = [a for a in actions if a.action in ("SELL", "TRIM")]
    buys = [a for a in actions if a.action in ("BUY", "ADD")]
    if sells:
        plan.notes.append("Reduce/exit: " + ", ".join(f"{a.code} ({a.action.lower()})" for a in sells[:6]) + ".")
    if buys:
        plan.notes.append("Add/build: " + ", ".join(f"{a.code}" for a in buys[:6]) + ".")
    over = [a for a in actions if a.current_pct > cap_pct]
    if over:
        plan.notes.append(
            f"Trim to the {cap_pct:.0f}% cap: "
            + ", ".join(f"{a.code} ({a.current_pct:.0f}%→{a.target_pct:.0f}%)" for a in over) + "."
        )
    if cash > total * (cash_target_pct + CASH_DEPLOY_NUDGE_PCT) / 100.0:
        plan.notes.append(f"~${cash:,.0f} idle cash — deploy into the add list above (keeping ~{cash_target_pct:.0f}% buffer).")
    if not sells and not buys:
        plan.notes.append("Book is already close to the model's target weights — no material trades.")
    return plan


# --- risk-aware optimizer: covariance-aware overlay on top of the score tilt ---

def _build_returns_frame(bars_by_code: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, list[str]]:
    """Daily log-return series per code, aligned on common dates. Codes with
    fewer than MIN_BARS_PER_HOLDING usable observations are dropped (the
    caller applies a neutral risk adjustment for them instead of dropping
    them from the plan)."""
    series: dict[str, pd.Series] = {}
    for code, bars in bars_by_code.items():
        if bars is None or bars.empty or "close" not in bars.columns:
            continue
        r = log_returns(bars["close"]).dropna()
        if len(r) < MIN_BARS_PER_HOLDING:
            continue
        series[code] = r
    if not series:
        return pd.DataFrame(), []
    frame = pd.concat(series, axis=1, join="inner").dropna(how="any")
    return frame, list(frame.columns)


def _shrunk_covariance(returns: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Ledoit-Wolf shrunk covariance (annualized) + correlation matrix + the
    shrinkage intensity actually used (0 = no shrinkage, 1 = fully toward the
    structured target -- high values flag a noisy sample matrix)."""
    x = returns.to_numpy()
    lw = LedoitWolf().fit(x)
    cov_daily = lw.covariance_
    delta = float(lw.shrinkage_)
    cov = cov_daily * PERIODS_PER_YEAR
    d = np.sqrt(np.clip(np.diagonal(cov_daily), 1e-12, None))
    corr = cov_daily / np.outer(d, d)
    return cov, corr, delta


def _risk_overlay_weights(
    raw: dict[str, float], cov: np.ndarray, corr: np.ndarray, cov_codes: list[str],
) -> dict[str, float]:
    """Stage A: scale each name's score-tilt weight by inverse volatility and
    a penalty for high average correlation to the rest of the (positively
    weighted) book. Names outside the covariance estimate (thin history) or
    already zeroed (SELL) pass through unchanged -- deterministic, no solver."""
    idx = {c: i for i, c in enumerate(cov_codes)}
    positive_idx = [idx[c] for c in cov_codes if raw.get(c, 0.0) > 0]
    risk_adj: dict[str, float] = {}
    for code, q in raw.items():
        if q <= 0 or code not in idx:
            risk_adj[code] = q
            continue
        i = idx[code]
        sigma_i = float(np.sqrt(max(cov[i, i], 0.0)))
        inv_vol_i = 1.0 / max(sigma_i, _FLOOR_VOL)
        others = [j for j in positive_idx if j != i]
        avg_corr_i = float(np.mean([corr[i, j] for j in others])) if others else 0.0
        corr_penalty_i = 1.0 / (1.0 + max(0.0, avg_corr_i) * CORR_PENALTY_STRENGTH)
        risk_adj[code] = q * inv_vol_i * corr_penalty_i
    return risk_adj


def _solve_risk_adjusted_weights(
    codes: list[str], risk_adj: np.ndarray, cap_frac: float, investable: float,
) -> tuple[dict[str, float], bool, str | None]:
    """Stage B: find the weight vector closest (squared distance) to the
    Stage-A proportional target, subject to sum-to-1, [0, cap_frac] bounds
    (no-short + concentration cap), and hard-zero bounds for already-excluded
    names. Deliberately NOT a variance-minimization/mean-variance objective --
    this keeps the solve a well-behaved convex QP whose only job is constraint
    satisfaction around an already-sane target, not return-seeking."""
    n = len(codes)
    if n == 0 or risk_adj.sum() <= 0:
        return {c: 0.0 for c in codes}, True, None

    n_positive = int(np.sum(risk_adj > 0))
    if cap_frac * n_positive < 1.0 - 1e-9:
        return {}, False, (
            f"the {cap_frac:.0%} cap can't be fully deployed across only {n_positive} "
            f"positively-rated name(s) (max reachable {cap_frac * n_positive:.0%})"
        )

    w0 = risk_adj / risk_adj.sum()
    bounds = [(0.0, cap_frac) if risk_adj[i] > 0 else (0.0, 0.0) for i in range(n)]
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    # Clip the warm start into the box bounds -- SLSQP's line search is fragile
    # when x0 badly violates a bound (e.g. one dominant score-tilt weight far
    # above the cap), even though the equality constraint alone is satisfiable.
    x_start = np.clip(w0, lo, hi)
    room = hi - x_start
    shortfall = 1.0 - x_start.sum()
    if shortfall > 1e-9 and room.sum() > 0:
        x_start = x_start + shortfall * (room / room.sum())

    def objective(w: np.ndarray) -> float:
        return float(np.sum((w - w0) ** 2))

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        objective, x_start, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"maxiter": 200, "ftol": 1e-9},
    )
    if not res.success or np.any(np.isnan(res.x)):
        return {}, False, f"solver did not converge: {res.message}"
    w = np.clip(res.x, lo, hi)
    s = w.sum()
    if s <= 0:
        return {}, False, "solver produced a degenerate (all-zero) allocation"
    w = w / s
    return {c: investable * wi for c, wi in zip(codes, w)}, True, None


def _risk_contributions(
    codes: list[str], weight_frac: dict[str, float], cov: np.ndarray, cov_codes: list[str],
) -> tuple[dict[str, float], float | None]:
    """Euler risk-contribution decomposition (each name's share of variance,
    scale-invariant so it always sums to 100% across the names it covers) plus
    the resulting annualized portfolio vol -- both computed only across names
    with a covariance estimate (thin-history names are excluded from both, so
    this modestly understates true book vol; disclosed via risk_notes)."""
    idx = {c: i for i, c in enumerate(cov_codes)}
    sub_codes = [c for c in codes if c in idx and weight_frac.get(c, 0.0) > 0]
    if not sub_codes:
        return {}, None
    w = np.array([weight_frac[c] for c in sub_codes])
    sub_idx = [idx[c] for c in sub_codes]
    sub_cov = cov[np.ix_(sub_idx, sub_idx)]
    port_var = float(w @ sub_cov @ w)
    if port_var <= 0:
        return {}, None
    marginal = sub_cov @ w
    rc = w * marginal / port_var * 100.0
    port_vol_pct = float(np.sqrt(port_var) * 100.0)
    return {c: float(v) for c, v in zip(sub_codes, rc)}, port_vol_pct


def _risk_aware_targets(
    holdings: list[HoldingAnalysis],
    raw: dict[str, float],
    cap_usd: float,
    investable: float,
    bars_by_code: dict[str, pd.DataFrame],
) -> tuple[dict[str, float] | None, list[str], float | None, float | None, dict[str, float]]:
    """Runs the Stage A + Stage B risk-aware pipeline. Returns
    (targets_usd_or_None, notes, portfolio_vol_pct, covariance_shrinkage, risk_contribution_pct).
    A None target means "could not run -- caller falls back to the heuristic";
    notes always explains why, whether it succeeded or fell back."""
    codes = [h.position.code for h in holdings]
    if len(codes) < MIN_HOLDINGS_FOR_RISK_MODEL:
        return None, [
            f"Risk-aware mode needs at least {MIN_HOLDINGS_FOR_RISK_MODEL} holdings with "
            f"price history to estimate correlations ({len(codes)} held) — showing the "
            f"score-based plan instead."
        ], None, None, {}

    returns, cov_codes = _build_returns_frame(bars_by_code)
    if len(cov_codes) < MIN_HOLDINGS_FOR_RISK_MODEL:
        return None, [
            f"Only {len(cov_codes)} holding(s) had enough price history "
            f"(≥{MIN_BARS_PER_HOLDING} days) to estimate correlations — showing the "
            f"score-based plan instead."
        ], None, None, {}
    if len(returns) < MIN_BARS_PER_HOLDING:
        return None, [
            f"Only {len(returns)} overlapping trading days across holdings — too short a "
            f"window to estimate correlations reliably — showing the score-based plan instead."
        ], None, None, {}

    cov, corr, shrinkage = _shrunk_covariance(returns)
    risk_adj_map = _risk_overlay_weights(raw, cov, corr, cov_codes)

    notes: list[str] = [
        f"Risk-aware weights use ~{len(returns)} days of price history across "
        f"{len(cov_codes)} of {len(codes)} holdings — a directional read on correlation, "
        f"not a precise one. Treat this as a second opinion alongside the score-based plan."
    ]
    thin = [c for c in codes if c not in cov_codes and raw.get(c, 0.0) > 0]
    if thin:
        notes.append(
            "Limited price history for " + ", ".join(thin[:6]) +
            " — used their score-based weight as-is (no risk adjustment applied)."
        )
    if shrinkage > SHRINKAGE_CAVEAT_THRESHOLD:
        notes.append(
            f"Correlations look noisy/concentrated (shrinkage {shrinkage:.0%}) — "
            f"risk estimates here are directional, not precise."
        )

    order = list(codes)   # stable order matching `raw` / the holdings list
    risk_adj_arr = np.array([max(0.0, risk_adj_map.get(c, 0.0)) for c in order])
    if risk_adj_arr.sum() <= 0:
        notes.append("Every holding's risk-adjusted weight collapsed to zero — showing the score-based plan instead.")
        return None, notes, None, shrinkage, {}

    cap_frac = cap_usd / investable if investable > 0 else 0.0
    targets_usd, converged, solver_note = _solve_risk_adjusted_weights(order, risk_adj_arr, cap_frac, investable)
    if not converged:
        notes.append(f"Risk-aware allocation couldn't be computed ({solver_note}) — showing the score-based plan instead.")
        return None, notes, None, shrinkage, {}

    weight_frac = {c: (targets_usd.get(c, 0.0) / investable if investable > 0 else 0.0) for c in order}
    risk_contrib, port_vol_pct = _risk_contributions(order, weight_frac, cov, cov_codes)

    return targets_usd, notes, port_vol_pct, shrinkage, risk_contrib


def _opt_reason(h: HoldingAnalysis, act: str, cur_pct: float, tgt_pct: float, cap: float) -> str:
    a = h.analysis
    bits = [f"signal {a.decision.label.lower()} (score {a.score:.0f}, {a.confidence_label.lower()} conf)"]
    if act == "SELL":
        bits.append("bearish read — exit to cash")
    elif act == "TRIM":
        if cur_pct > cap:
            bits.append(f"{cur_pct:.0f}% of book exceeds the {cap:.0f}% cap — trim")
        else:
            bits.append("trim toward target weight")
    elif act in ("ADD", "BUY"):
        bits.append(f"raise weight {cur_pct:.0f}%→{tgt_pct:.0f}% on strength")
    else:
        bits.append("near target — hold")
    return "; ".join(bits)
