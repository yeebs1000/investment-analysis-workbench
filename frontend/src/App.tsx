import { useEffect, useState } from 'react'
import './App.css'
import { api, type Health } from './api'
import { PortfolioView } from './components/PortfolioView'
import { OptimiserView } from './components/OptimiserView'
import { WatchlistView } from './components/WatchlistView'
import { SymbolView } from './components/SymbolView'
import { LLMToggle } from './components/LLMToggle'
import { CostMeter } from './components/CostMeter'
import { ProviderProvider } from './ProviderContext'
import { ThemeProvider, useTheme } from './ThemeContext'
import { DECISION_META } from './format'
import type { Decision } from './types'

type Tab = 'portfolio' | 'optimiser' | 'watchlists' | 'symbol'

const LEGEND_ORDER: Decision[] = ['STRONG_BUY', 'BUY', 'ACCUMULATE', 'HOLD', 'REDUCE', 'SELL']

export default function App() {
  const [tab, setTab] = useState<Tab>('portfolio')
  const [health, setHealth] = useState<Health | null>(null)

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null))
  }, [])

  return (
    <ThemeProvider>
    <ProviderProvider>
      <div className="app">
        <header className="app-header">
          <div className="brand">
            <h1>Investment Analysis Workbench</h1>
            <span className="tag-readonly">read-only</span>
            {health?.brokers &&
              health.brokers.map((b) => {
                const label = b === 'ibkr' ? 'IBKR' : b === 'tiger' ? 'Tiger' : 'Moomoo'
                const st = health.broker_status?.[b]
                const unreachable = st === 'unreachable'
                return (
                  <span
                    key={b}
                    className={`tag-brokers${unreachable ? ' tag-broker-down' : ''}`}
                    title={
                      unreachable
                        ? `${label} is configured but not reachable — is its gateway running and logged in?`
                        : `${label}: ${st ?? 'configured'}`
                    }
                  >
                    {label}
                    {unreachable ? ' ⚠' : ''}
                  </span>
                )
              })}
          </div>
          <div className="header-right">
            <nav className="tabs">
              <button className={tab === 'portfolio' ? 'on' : ''} onClick={() => setTab('portfolio')}>
                Portfolio
              </button>
              <button className={tab === 'optimiser' ? 'on' : ''} onClick={() => setTab('optimiser')}>
                Optimiser
              </button>
              <button className={tab === 'watchlists' ? 'on' : ''} onClick={() => setTab('watchlists')}>
                Watchlists
              </button>
              <button className={tab === 'symbol' ? 'on' : ''} onClick={() => setTab('symbol')}>
                Symbol lookup
              </button>
            </nav>
            <ThemeToggle />
          </div>
        </header>

        <div className="llm-bar">
          <LLMToggle />
          <CostMeter />
        </div>

        <Legend />

        <main className="content">
          {tab === 'portfolio' && <PortfolioView />}
          {tab === 'optimiser' && <OptimiserView />}
          {tab === 'watchlists' && <WatchlistView />}
          {tab === 'symbol' && <SymbolView />}
        </main>

        <footer className="disclaimer">
          {health?.disclaimer ??
            'Decision-support only — not financial advice. Read-only; no orders are ever placed.'}
        </footer>
      </div>
    </ProviderProvider>
    </ThemeProvider>
  )
}

function ThemeToggle() {
  const { theme, toggleTheme } = useTheme()
  return (
    <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme" title="Toggle theme">
      {theme === 'dark' ? '☀' : '☾'}
    </button>
  )
}

function Legend() {
  return (
    <div className="legend">
      <span className="muted small">What the calls mean:</span>
      {LEGEND_ORDER.map((d) => (
        <span className="legend-item" key={d} title={DECISION_META[d].plain}>
          <span className="legend-dot" style={{ background: DECISION_META[d].color }} />
          <b>{DECISION_META[d].label}</b>
        </span>
      ))}
    </div>
  )
}
