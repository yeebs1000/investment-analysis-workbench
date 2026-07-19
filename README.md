# 📊 Investment Analysis Workbench

[![CI](https://github.com/yeebs1000/investment-analysis-workbench/actions/workflows/ci.yml/badge.svg)](https://github.com/yeebs1000/investment-analysis-workbench/actions/workflows/ci.yml) ![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.11%2B-blue) ![Read--only](https://img.shields.io/badge/broker%20access-read--only-brightgreen)

A local, **read-only** workbench that reads your brokerage account (Moomoo, and
optionally IBKR / Tiger) and turns it into hedge-fund-style analysis: number-backed
technical signals, BUY / SELL / ACCUMULATE / HOLD / REDUCE decisions per holding,
a risk-aware rebalance plan, an options strategist, and performance tracking vs the
S&P 500 — all on a clean dashboard a non-finance reader can follow.

<!-- SCREENSHOT: save a PNG to docs/screenshot.png, then uncomment the line
     below. Tip: the Symbol-lookup or Watchlist view shows no account balances,
     so it's the safe choice for a public README (the Portfolio view shows your
     real holdings/dollar amounts). -->
<!-- ![Dashboard — per-symbol technical analysis](docs/screenshot.png) -->

Every number on screen is computed by deterministic Python you can read line by
line — indicators, scoring, options math, portfolio optimizer. An LLM (optional)
only narrates that precomputed JSON in plain English; it never invents a figure,
and the app works fully with zero AI key. It does not place real trades, does
not predict prices, and does not scrape news — it reads your actual positions
and tells you what the numbers already say.

The one exception to read-only is an optional **paper-trading loop**,
hard-locked to Moomoo's SIMULATE environment by construction (see the
guardrails header in `backend/app/brokers/paper_broker.py`) — it cannot touch
real money. The daily recorder → signal-log → paper-trade scripts live in
`backend/scripts/`; each script's module docstring documents its flags. All
personal data (bar caches, chain archives, signal logs, trades, journals,
result reports) stays in the gitignored `backend/data_store/` and never leaves
your machine — the repo carries the base version only.

## Why this is different

- **Auditable, not a black box.** Every score, stop, target, and Greek traces to
  Python you can read in one sitting — no vendor API, no opaque "AI says buy."
- **True multi-broker, not a skin.** Moomoo, IBKR, and Tiger each have a real
  fallback chain for bars, quotes, and options — link one, several, or none of
  the account-linked brokers and the app degrades honestly instead of failing.
- **ML that's allowed to say no.** The optional forecast trains with purged
  walk-forward validation, a shuffled-label leak check, and a block-bootstrap
  confidence interval — it only activates if it clears its own gate, and ships
  at 0 score-weight otherwise. Most retail "AI trading" tools skip this and
  ship whatever overfit backtest looked good.
- **Real options math, not a vibe.** Black-Scholes Greeks, a GARCH forward-vol
  forecast compared against implied, probability-of-profit and expected value
  per structure — computed, not guessed.
- **Free to run fully.** The deterministic engine needs zero API keys; AI
  narration and macro/fundamentals context are opt-in extras, not the product.
- **Read-only, always.** No order-placement code path exists anywhere in the
  app — there's nothing to accidentally wire up.

## 👉 New here? Read [SETUP.md](SETUP.md)

[SETUP.md](SETUP.md) is a step-by-step guide that assumes **zero technical background** —
installing Python/Node, connecting your broker, and running the app. Start there.

## What it does

- **Deterministic engine (always on, free, no API key).** Computes every indicator
  (RSI + slope, MACD, EMA20/50/200 + slope, ADX/DI, Bollinger/%B, ATR, OBV, Stochastic,
  Donchian, momentum, RSI divergence, support/resistance) and blends them into a
  transparent 0–100 score and a discrete decision. Five "analysts" (Trend, Momentum,
  Volatility, Volume, Levels) contribute weighted, number-tagged reasons; conviction is
  corroborated against a higher timeframe, trend strength, and volume.
- **Portfolio & risk.** FX-normalized combined book across brokers, concentration and
  exposure checks, and a **rebalance optimiser** with a configurable per-name cap
  (score-based, plus an opt-in risk-aware mode using Ledoit-Wolf covariance).
- **Options strategist.** IV regime (implied vs a **GARCH forward vol forecast** over the
  tenor), native Black-Scholes Greeks, probability-of-profit and expected value, concrete
  defined-risk structures (spreads, covered call, iron condor, straddle), earnings/liquidity
  warnings, and risk-budgeted position sizing.
- **Context (optional, Finnhub/FRED).** Fundamental quality score, analyst consensus,
  earnings-surprise history, insider sentiment, and a macro "market weather" regime.
- **Performance vs SPY.** Persists a daily equity snapshot and reports cumulative return,
  excess vs SPY, alpha/beta, tracking error, information ratio, and drawdown.
- **AI narration (optional).** Toggle Gemini or Claude to add plain-English briefs and an
  **Ask** box that answers situation-specific questions ("I hold 900 shares — what for
  income?") using only the pre-computed numbers. Cached and cost-metered; deterministic
  mode needs no key.

## ⚠️ Broker compatibility — read before picking your setup

Moomoo was the original broker; **every subsystem now has an automatic
fallback chain (Moomoo → IBKR → Tiger)** so any single broker, or any
combination, gets the full feature set — technical scores, options
strategist, and all. Live-verified end to end against a real Moomoo + IBKR
account this session, including forcing Moomoo's options path to fail to
confirm the IBKR fallback actually engages and degrades honestly (no
fabricated numbers) when the account lacks a live options-data subscription.
**Tiger's market-data fallback (bars, options) is code-complete but not
live-tested** — no real Tiger account was available this session; it's built
strictly from the column names documented in the installed `tigeropen` SDK's
own source, but hasn't been run against real data. If something looks off on
a Tiger-only setup, that's the first place to check.

| Feature | Moomoo | IBKR-only (no Moomoo) | Tiger-only (no Moomoo/IBKR) |
|---|---|---|---|
| Balances & positions (combined, FX-normalized) | ✅ | ✅ | ✅ |
| Technical score / charts / decisions | ✅ | ✅ (fallback for US/HK/JP/China-Connect/SG; other markets may still fail to chart) | ✅ *(new; not live-tested)* |
| Options strategist (chain, Greeks, IV, structures) | ✅ | ✅ *(new — needs an options market-data subscription for live quotes; degrades to Greeks-only, no fabricated prices, if you only have delayed/no options data)* | ✅ *(new; not live-tested)* |
| Live quote freshness (display name + 52wk-high/low flavor text only — price/score always come from bars, on every broker) | ✅ | ✅ *(new)* | ✅ *(new; not live-tested)* |
| Level-2 order book depth chip | ✅ full 10-level depth on US stocks/ETFs, no paid subscription needed; top-of-book only on some other markets | ✅ fallback (needs a paid deep-book subscription) | ❌ not implemented |
| Watchlists (view / analyze / add / remove) | ✅ native groups + app-local lists | ✅ app-local lists (broker-independent) | ✅ app-local lists (broker-independent) |
| Fundamentals, analyst consensus, earnings, insider, macro regime, AI narration | ✅ | ✅ (Finnhub/FRED/Gemini/Claude — broker-independent) | ✅ |

The **Level-2 depth chip** tries Moomoo first (full 10-level book on US
stocks/ETFs, included free) and falls back to IBKR's deep book (needs a paid
subscription there) — so most users get it with no extra cost.

## Quick start (see [SETUP.md](SETUP.md) for the full walkthrough)

Prerequisites: **Python 3.11**, **Node.js**, and **Moomoo OpenD** running + logged in.

```powershell
# Backend
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env            # then edit .env (see SETUP.md)
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8010

# Frontend (second terminal)
cd frontend
npm install
npm run dev                       # open http://localhost:5173
```

## Project layout

```
backend/    FastAPI + analytics
  app/analytics/   deterministic indicators, technical/options/risk/performance engines
  app/brokers/     read-only clients: moomoo_client, ibkr_client, tiger_client
  app/data/        typed models, normalizers, Finnhub + FRED providers
  app/llm/         optional Gemini/Claude narration (router, cache, cost meter)
  app/ml/          optional offline-trained signal (never auto-applied)
  tests/           indicator/options/risk/ML/quality/performance correctness tests
frontend/   Vite + React + TypeScript dashboard
```

## Tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m tests.test_indicators
.\.venv\Scripts\python.exe -m tests.test_options_math
.\.venv\Scripts\python.exe -m tests.test_risk_optimizer
.\.venv\Scripts\python.exe -m tests.test_fundamental_quality
.\.venv\Scripts\python.exe -m tests.test_performance
.\.venv\Scripts\python.exe -m tests.test_ml_features
```

## Configuration

All settings live in `backend/.env` (git-ignored; copy from `.env.example`). Every field
has a safe default and the app runs fully in free deterministic mode with no keys. Optional
integrations: IBKR and Tiger (read-only broker merges), Finnhub and FRED (context data),
Gemini and Claude (AI narration). See [SETUP.md](SETUP.md) for where to get each key.

## Safety

Read-only by design — the app cannot trade. API keys and broker private keys live only in
your local `.env`/key files and are never committed. Run it locally or over a private VPN
(e.g. Tailscale); never expose it to the public internet.

## Disclaimer

**Decision-support only — not financial advice.** This is a personal tool for reading your
own accounts and computing indicators; it is not a recommendation to buy, sell, or hold
anything. The system is read-only and never places, stages, or modifies an order — every
decision and every execution is yours. Markets are risky and past performance (including
anything this app reports vs. SPY) does not predict future results.

## License

MIT — see [LICENSE](LICENSE).
