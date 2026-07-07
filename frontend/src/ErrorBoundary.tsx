import { Component, type ReactNode } from 'react'

interface Props { children: ReactNode }
interface State { error: Error | null }

// Without this, an uncaught render error unmounts the whole app to a blank
// white page with no clue why (e.g. backend on the wrong port, malformed
// API response). This just makes failures visible instead of silent.
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  render() {
    if (this.state.error) {
      return (
        <div style={{ padding: 40, fontFamily: 'sans-serif', maxWidth: 640, margin: '0 auto' }}>
          <h2>Something went wrong.</h2>
          <p>
            The dashboard hit an unexpected error. This is usually the backend being
            unreachable, on the wrong port, or another process squatting :8010.
          </p>
          <pre style={{ whiteSpace: 'pre-wrap', opacity: 0.7 }}>{this.state.error.message}</pre>
          <button onClick={() => window.location.reload()}>Reload</button>
        </div>
      )
    }
    return this.props.children
  }
}
