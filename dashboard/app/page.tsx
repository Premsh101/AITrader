'use client'

import { useEffect, useState, useCallback } from 'react'
import {
  TrendingUp,
  TrendingDown,
  BarChart2,
  Activity,
  RefreshCw,
  Trophy,
  Layers,
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

  useEffect(() => { loadAll() }, [loadAll])

  useEffect(() => {
    const timer = setInterval(() => loadAll(), 30_000)
    return () => clearInterval(timer)
  }, [loadAll])

  const pnlPositive   = stats?.total_pnl != null ? stats.total_pnl > 0 : null
  const dailyPositive = stats?.daily_pnl  != null ? stats.daily_pnl  > 0 : null
  const winGood       = stats ? stats.win_rate >= 50 : null

  const openTrades   = trades.filter(t => t.status === 'Open')
  const closedTrades = trades.filter(t => t.status !== 'Open')

  return (
    <div className="flex flex-col min-h-screen bg-[#0d1117] text-slate-200">
      <Header isLive={isLive} onModeChange={setIsLive} />

      <main className="flex-1 p-6 space-y-6 max-w-[1440px] mx-auto w-full">

        {/* Error banner */}
        {error && (
          <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/20 text-red-400 px-4 py-3 rounded-xl text-sm">
            <span className="text-lg">⚠️</span>
            <span>{error}</span>
          </div>
        )}

        {/* Toolbar */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-base font-semibold text-slate-100">Overview</h2>
            {lastUpdated && (
              <p className="text-xs text-slate-600 mt-0.5">
                Last updated {lastUpdated.toLocaleTimeString()}
              </p>
            )}
          </div>
          <button
            onClick={() => loadAll(true)}
            disabled={refreshing}
            className="flex items-center gap-2 px-4 py-2 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-300 text-xs rounded-xl border border-slate-700 hover:border-slate-600 transition-all duration-200"
          >
            <RefreshCw size={13} className={refreshing ? 'animate-spin' : ''} />
            {refreshing ? 'Refreshing…' : 'Refresh'}
          </button>
        </div>

        {/* Stats grid */}
        {loading ? (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="bg-[#161b27] border border-slate-800 rounded-2xl p-5 h-28 animate-pulse" />
            ))}
          </div>
        ) : (
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <StatsCard
              title="Daily P&L"
              value={stats ? `${stats.daily_pnl >= 0 ? '+' : ''}₹${stats.daily_pnl.toFixed(2)}` : '—'}
              icon={<TrendingUp size={18} />}
              positive={dailyPositive}
              subtitle="Today's realised gains"
            />
            <StatsCard
              title="Total Profit"
              value={stats ? `${stats.total_pnl >= 0 ? '+' : ''}₹${stats.total_pnl.toFixed(2)}` : '—'}
              icon={<BarChart2 size={18} />}
              positive={pnlPositive}
              subtitle={`${stats?.total_trades ?? 0} closed trades`}
            />
            <StatsCard
              title="Win Rate"
              value={stats ? `${stats.win_rate.toFixed(1)}%` : '—'}
              icon={<Trophy size={18} />}
              positive={winGood}
              subtitle={`${stats?.winning_trades ?? 0} winning trades`}
            />
            <StatsCard
              title="Open Positions"
              value={stats?.active_positions ?? '—'}
              icon={<Activity size={18} />}
              positive={null}
              subtitle={`${closedTrades.length} positions closed`}
            />
          </div>
        )}

        {/* Secondary stats bar */}
        {!loading && stats && (
          <div className="grid grid-cols-3 gap-4">
            {[
              { label: 'Total Trades',    value: stats.total_trades,    icon: <Layers size={14} /> },
              { label: 'Winning Trades',  value: stats.winning_trades,  icon: <TrendingUp size={14} /> },
              { label: 'Losing Trades',   value: stats.total_trades - stats.winning_trades, icon: <TrendingDown size={14} /> },
            ].map(({ label, value, icon }) => (
              <div key={label} className="bg-[#161b27] border border-slate-800 rounded-xl px-5 py-3 flex items-center justify-between">
                <span className="text-xs text-slate-500 font-medium">{label}</span>
                <div className="flex items-center gap-1.5 text-slate-300">
                  <span className="text-slate-600">{icon}</span>
                  <span className="text-sm font-semibold">{value}</span>
                </div>
              </div>
            ))}
          </div>
        )}

        {/* Open Positions quick strip */}
        {!loading && openTrades.length > 0 && (
          <div className="bg-[#161b27] border border-amber-500/20 rounded-2xl px-5 py-4 shadow-lg">
            <div className="flex items-center gap-2 mb-3">
              <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse" />
              <h3 className="text-sm font-semibold text-amber-300">
                Open Positions ({openTrades.length})
              </h3>
            </div>
            <div className="flex flex-wrap gap-2">
              {openTrades.map(t => (
                <div
                  key={t.id}
                  className="flex items-center gap-2 bg-slate-800/60 border border-slate-700/50 rounded-lg px-3 py-1.5 text-xs"
                >
                  <span className="font-semibold text-white">{t.symbol.replace('.NS', '')}</span>
                  {t.buy_price && (
                    <span className="text-slate-400">@ ₹{parseFloat(t.buy_price).toFixed(2)}</span>
                  )}
                  <span className="text-amber-400 text-[10px] font-medium uppercase">
                    {t.mode}
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Activity + Trade table side by side on xl */}
        <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
          <div className="xl:col-span-2">
            <ActivityLog entries={activity} />
          </div>
          <div className="xl:col-span-3">
            <TradeTable trades={trades} />
          </div>
        </div>

      </main>

      <footer className="text-center text-[11px] text-slate-700 py-5 border-t border-slate-800/60">
        AITrader &copy; {new Date().getFullYear()} — Auto-refreshes every 30 s
      </footer>
    </div>
  )
}
