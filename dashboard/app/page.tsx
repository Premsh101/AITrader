'use client'

import { useEffect, useState, useCallback } from 'react'
import {
  TrendingUp,
  TrendingDown,
  BarChart2,
  Activity,
  RefreshCw,
} from 'lucide-react'

import Header from './components/Header'
import StatsCard from './components/StatsCard'
import ActivityLog from './components/ActivityLog'
import TradeTable from './components/TradeTable'
import {
  fetchStats,
  fetchTrades,
  fetchActivity,
  fetchConfig,
  Stats,
  Trade,
  ActivityEntry,
} from './lib/api'

export default function Dashboard() {
  const [isLive, setIsLive] = useState(false)
  const [stats, setStats] = useState<Stats | null>(null)
  const [trades, setTrades] = useState<Trade[]>([])
  const [activity, setActivity] = useState<ActivityEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null)
  const [error, setError] = useState<string | null>(null)

  const loadAll = useCallback(async (showSpinner = false) => {
    if (showSpinner) setRefreshing(true)
    setError(null)
    try {
      const [cfg, s, t, a] = await Promise.all([
        fetchConfig(),
        fetchStats(),
        fetchTrades(100),
        fetchActivity(50),
      ])
      setIsLive(cfg.is_live_mode)
      setStats(s)
      setTrades(t)
      setActivity(a)
      setLastUpdated(new Date())
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [])

  // Initial load
  useEffect(() => {
    loadAll()
  }, [loadAll])

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const timer = setInterval(() => loadAll(), 30_000)
    return () => clearInterval(timer)
  }, [loadAll])

  const pnlPositive =
    stats?.total_pnl != null ? stats.total_pnl > 0 : null
  const dailyPositive =
    stats?.daily_pnl != null ? stats.daily_pnl > 0 : null

  return (
    <div className="flex flex-col min-h-screen bg-[#0f1117]">
      <Header isLive={isLive} onModeChange={setIsLive} />

      <main className="flex-1 p-6 space-y-6 max-w-[1400px] mx-auto w-full">
        {/* Error banner */}
        {error && (
          <div className="bg-red-500/10 border border-red-500/30 text-red-400 px-4 py-3 rounded-lg text-sm">
            ⚠️ {error}
          </div>
        )}

        {/* Refresh bar */}
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-slate-200">Dashboard</h2>
          <div className="flex items-center gap-3">
            {lastUpdated && (
              <span className="text-xs text-slate-500">
                Updated {lastUpdated.toLocaleTimeString()}
              </span>
            )}
            <button
              onClick={() => loadAll(true)}
              disabled={refreshing}
              className="flex items-center gap-2 px-4 py-2 bg-slate-700 hover:bg-slate-600 disabled:opacity-50 text-slate-200 text-sm rounded-lg transition-colors"
            >
              <RefreshCw
                size={14}
                className={refreshing ? 'animate-spin' : ''}
              />
              {refreshing ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
        </div>

        {/* Stats row */}
        {loading ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div
                key={i}
                className="bg-[#161b27] border border-slate-700 rounded-xl p-5 h-24 animate-pulse"
              />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatsCard
              title="Daily P&L"
              value={
                stats
                  ? `${stats.daily_pnl >= 0 ? '+' : ''}₹${stats.daily_pnl.toFixed(2)}`
                  : '—'
              }
              icon={<TrendingUp size={20} />}
              positive={dailyPositive}
            />
            <StatsCard
              title="Total Profit"
              value={
                stats
                  ? `${stats.total_pnl >= 0 ? '+' : ''}₹${stats.total_pnl.toFixed(2)}`
                  : '—'
              }
              icon={<BarChart2 size={20} />}
              positive={pnlPositive}
            />
            <StatsCard
              title="Win Rate %"
              value={stats ? `${stats.win_rate.toFixed(1)}%` : '—'}
              icon={<TrendingDown size={20} />}
              positive={stats ? stats.win_rate >= 50 : null}
            />
            <StatsCard
              title="Active Positions"
              value={stats?.active_positions ?? '—'}
              icon={<Activity size={20} />}
              positive={null}
            />
          </div>
        )}

        {/* Activity log + Trade table */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <ActivityLog entries={activity} />
          <div className="xl:col-span-1 overflow-hidden">
            {/* placeholder for second column on smaller screens */}
          </div>
        </div>

        <TradeTable trades={trades} />
      </main>

      <footer className="text-center text-xs text-slate-600 py-4 border-t border-slate-800">
        AITrader © {new Date().getFullYear()} — Auto-refreshes every 30 s
      </footer>
    </div>
  )
}
