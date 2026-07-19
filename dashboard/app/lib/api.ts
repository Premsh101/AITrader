/**
 * lib/api.ts – typed helpers for the AITrader FastAPI backend.
 */

const API_URL =
  process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

// Shared secret for mutating endpoints; must match AITRADER_API_KEY on the
// backend. Embedded at build time, so keep the dashboard itself private.
const API_KEY = process.env.NEXT_PUBLIC_API_KEY ?? ''

export interface Stats {
  total_pnl: number
  daily_pnl: number
  win_rate: number
  active_positions: number
  total_trades: number
  winning_trades: number
}

export interface Trade {
  id: number
  symbol: string
  quantity: number
  mode: string
  status: string
  buy_price: string | null
  sell_price: string | null
  pnl: string | null
  created_at: string | null
  updated_at: string | null
}

export interface ActivityEntry {
  agent: string
  message: string
  timestamp: string
}

export interface Config {
  is_live_mode: boolean
  last_sync_time: string | null
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`API ${path} → ${res.status}: ${body}`)
  }
  return res.json() as Promise<T>
}

export const fetchStats = () => apiFetch<Stats>('/stats')
export const fetchTrades = (limit = 100) =>
  apiFetch<Trade[]>(`/trades?limit=${limit}`)
export const fetchActivity = (limit = 50) =>
  apiFetch<ActivityEntry[]>(`/activity?limit=${limit}`)
export const fetchConfig = () => apiFetch<Config>('/config')

/** Headers for mutating requests (API-key protected on the backend). */
const mutatingHeaders = { 'X-API-Key': API_KEY }

export const toggleMode = (mode: 'PAPER' | 'LIVE') =>
  apiFetch<{ mode: string; is_live_mode: boolean }>('/toggle-mode', {
    method: 'POST',
    headers: mutatingHeaders,
    // confirm is required by the backend when switching to LIVE; the UI has
    // already asked the user before this call is made.
    body: JSON.stringify({ mode, confirm: true }),
  })

export const placeOrder = (order: {
  symbol: string
  quantity: number
  side: 'BUY' | 'SELL'
  reference_price?: number
}) =>
  apiFetch<Record<string, unknown>>('/order', {
    method: 'POST',
    headers: mutatingHeaders,
    body: JSON.stringify(order),
  })
