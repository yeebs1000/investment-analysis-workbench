import { createContext, useContext, useState, type ReactNode } from 'react'
import type { Provider } from './api'

interface Ctx {
  provider: Provider
  setProvider: (p: Provider) => void
  bump: number // increments to signal "usage changed" so the cost meter refreshes
  notifyUsage: () => void
}

const ProviderCtx = createContext<Ctx>({
  provider: 'none',
  setProvider: () => {},
  bump: 0,
  notifyUsage: () => {},
})

export function ProviderProvider({ children }: { children: ReactNode }) {
  const [provider, setProvider] = useState<Provider>('none')
  const [bump, setBump] = useState(0)
  return (
    <ProviderCtx.Provider
      value={{ provider, setProvider, bump, notifyUsage: () => setBump((b) => b + 1) }}
    >
      {children}
    </ProviderCtx.Provider>
  )
}

// eslint-disable-next-line react-refresh/only-export-components
export function useProvider() {
  return useContext(ProviderCtx)
}
