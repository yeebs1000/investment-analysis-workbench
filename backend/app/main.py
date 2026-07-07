"""FastAPI app exposing the read-only analysis API.

Endpoints are plain `def` (run in a threadpool) because the broker SDK is
blocking. The frontend dev server (Vite) is allowed via CORS.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.data.models import (
    Account,
    AskResponse,
    ChartSeries,
    FundamentalMetrics,
    OptimizerPlan,
    OptionsAnalysis,
    PortfolioAnalysis,
    Position,
    SearchResult,
    TechnicalAnalysis,
    WatchlistAnalysis,
    WatchlistGroup,
)
from app.llm import orchestrator
from app.llm.orchestrator import Narrative
from app.llm.router import router as llm_router
from app.services.analysis_service import DEFAULT_TF, TIMEFRAMES, service

DISCLAIMER = (
    "Decision-support only — not financial advice. All analysis is generated from "
    "your own brokerage data and is read-only; no orders are ever placed. You make and "
    "execute all decisions yourself."
)

app = FastAPI(title="Investment Analysis Workbench", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    # `brokers` = actually wired up (SDK installed + enabled); `broker_status` =
    # live reachability (e.g. Moomoo "unreachable" when OpenD isn't running),
    # so the UI can stop cheerfully claiming a broker works when it doesn't.
    configured = service.configured_brokers()
    try:
        status = service.broker_status()
    except Exception:  # noqa: BLE001 - health must never 500
        status = {}
    return {
        "status": "ok",
        "broker": configured[0] if configured else "none",
        "brokers": configured,
        "broker_status": status,
        "ibkr_enabled": settings.ibkr_enabled,
        "tiger_enabled": settings.tiger_enabled,
        "security_firm": settings.security_firm,
        "read_only": True,
        "disclaimer": DISCLAIMER,
    }


@app.get("/api/account", response_model=Account)
def account() -> Account:
    try:
        return service.get_account()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/positions", response_model=list[Position])
def positions() -> list[Position]:
    try:
        return service.get_positions()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/portfolio", response_model=PortfolioAnalysis)
def portfolio(tf: str = Query(DEFAULT_TF)) -> PortfolioAnalysis:
    try:
        return service.analyze_portfolio(tf=tf)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/performance")
def performance() -> dict:
    try:
        return service.get_performance()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/timeframes")
def timeframes() -> dict:
    return {"default": DEFAULT_TF,
            "items": [{"value": k, "label": v["label"]} for k, v in TIMEFRAMES.items()]}


_OPTIMIZE_METHODS = {"heuristic", "risk_aware"}


@app.get("/api/optimize", response_model=OptimizerPlan)
def optimize(
    tf: str = Query(DEFAULT_TF),
    method: str = Query("heuristic"),
    cap_pct: float = Query(15.0, ge=5.0, le=50.0,
                           description="single-name concentration cap; raise for deliberate core overweights"),
    cash_target_pct: float = Query(5.0, ge=0.0, le=50.0),
) -> OptimizerPlan:
    if method not in _OPTIMIZE_METHODS:
        raise HTTPException(status_code=400, detail=f"method must be one of {sorted(_OPTIMIZE_METHODS)}")
    try:
        return service.optimize(tf=tf, method=method, cap_pct=cap_pct, cash_target_pct=cash_target_pct)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/search", response_model=list[SearchResult])
def search(q: str = Query(..., min_length=1)) -> list[SearchResult]:
    try:
        return service.search(q)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/watchlist/add")
def watchlist_add(code: str = Query(...), group: str | None = Query(None),
                  source: str | None = Query(None)) -> dict:
    try:
        return service.add_to_watchlist(code, group, source)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/watchlist/remove")
def watchlist_remove(code: str = Query(...), group: str = Query(...)) -> dict:
    try:
        return service.remove_from_watchlist(code, group)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.post("/api/watchlist/delete")
def watchlist_delete(group: str = Query(...)) -> dict:
    try:
        return service.delete_watchlist(group)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/analyze/{code}", response_model=TechnicalAnalysis)
def analyze(code: str, tf: str = Query(DEFAULT_TF)) -> TechnicalAnalysis:
    try:
        return service.analyze_symbol(code, tf=tf)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/watchlists", response_model=list[WatchlistGroup])
def watchlists() -> list[WatchlistGroup]:
    try:
        return service.list_watchlists()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


# group names contain spaces and '/', so pass via query param (not path).
@app.get("/api/watchlist", response_model=WatchlistAnalysis)
def watchlist(group: str = Query(...), limit: int = Query(30, ge=1, le=60),
              tf: str = Query(DEFAULT_TF), source: str | None = Query(None)) -> WatchlistAnalysis:
    try:
        return service.analyze_watchlist(group, limit=limit, tf=tf, source=source)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/chart/{code}", response_model=ChartSeries)
def chart(code: str, lookback: int = Query(180, ge=30, le=500),
          tf: str = Query(DEFAULT_TF)) -> ChartSeries:
    try:
        return service.get_chart_series(code, lookback=lookback, tf=tf)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


# ---------------- options ----------------
@app.get("/api/options/{code}", response_model=OptionsAnalysis)
def options(code: str, dte: int = Query(35, ge=5, le=180)) -> OptionsAnalysis:
    try:
        return service.analyze_options(code, target_dte=dte)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/options/{code}/explain", response_model=Narrative)
def explain_options(code: str, dte: int = Query(35, ge=5, le=180),
                    provider: str | None = Query(None)) -> Narrative:
    try:
        oa = service.analyze_options(code, target_dte=dte)
        return orchestrator.explain_options(oa, provider)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/options/{code}/ask", response_model=AskResponse)
def ask_options(code: str, q: str = Query(..., min_length=2),
                dte: int = Query(35, ge=5, le=180),
                provider: str | None = Query(None)) -> AskResponse:
    try:
        oa = service.analyze_options(code, target_dte=dte)
        return orchestrator.ask_options(oa, q, provider)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


# ---------------- LLM Q&A about a symbol ----------------
@app.get("/api/ask/{code}", response_model=AskResponse)
def ask_symbol(code: str, q: str = Query(..., min_length=2),
               tf: str = Query(DEFAULT_TF), provider: str | None = Query(None)) -> AskResponse:
    try:
        ta = service.analyze_symbol(code, tf=tf)
        return orchestrator.ask_symbol(ta, q, provider)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


# ---------------- fundamentals (value-investing lens) ----------------
@app.get("/api/fundamentals/{code}", response_model=FundamentalMetrics)
def fundamentals(code: str) -> FundamentalMetrics:
    try:
        return service.get_fundamentals(code)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/fundamentals/{code}/ask", response_model=AskResponse)
def ask_fundamentals(code: str, q: str = Query(..., min_length=2),
                     provider: str | None = Query(None)) -> AskResponse:
    try:
        fm = service.get_fundamentals(code)
        return orchestrator.ask_fundamentals(fm, q, provider)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


# ---------------- LLM narration ----------------
@app.get("/api/llm/status")
def llm_status() -> dict:
    return llm_router.providers_status()


@app.get("/api/llm/usage")
def llm_usage() -> dict:
    return llm_router.usage()


@app.post("/api/llm/reset")
def llm_reset() -> dict:
    llm_router.reset_usage()
    return {"status": "reset"}


@app.post("/api/llm/model")
def llm_set_model(provider: str = Query(...), model: str = Query(...)) -> dict:
    try:
        return llm_router.set_model(provider, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/explain/{code}", response_model=Narrative)
def explain_symbol(code: str, provider: str | None = Query(None),
                   tf: str = Query(DEFAULT_TF)) -> Narrative:
    try:
        ta = service.analyze_symbol(code, tf=tf)
        return orchestrator.explain_symbol(ta, provider)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))


@app.get("/api/portfolio/explain", response_model=Narrative)
def explain_portfolio(provider: str | None = Query(None)) -> Narrative:
    try:
        pa = service.analyze_portfolio()
        return orchestrator.explain_portfolio(pa, provider)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc))
