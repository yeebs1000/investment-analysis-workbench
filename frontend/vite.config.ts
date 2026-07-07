import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy /api to the FastAPI backend so the frontend can use relative URLs.
export default defineConfig({
  plugins: [react()],
  server: {
    // host: true binds 0.0.0.0 so the app is reachable from a phone on the
    // same LAN (http://<pc-ip>:5173) — or from anywhere via Tailscale. The
    // backend stays on 127.0.0.1; phone traffic reaches it only through this
    // dev server's /api proxy below. Never port-forward this to the internet.
    host: true,
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8010',
        changeOrigin: true,
      },
    },
  },
})
