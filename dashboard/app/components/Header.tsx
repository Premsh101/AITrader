'use client'

import { toggleMode } from '../lib/api'

interface HeaderProps {
  isLive: boolean
  onModeChange: (live: boolean) => void
}

export default function Header({ isLive, onModeChange }: HeaderProps) {
  const handleToggle = async () => {
    const newMode = isLive ? 'PAPER' : 'LIVE'
    try {
      await toggleMode(newMode)
      onModeChange(!isLive)
    } catch (err) {
      console.error('Failed to toggle mode:', err)
    }
  }

  return (
    <header className="flex items-center justify-between px-6 py-4 bg-[#161b27] border-b border-slate-700 shadow-md">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <span className="text-2xl">🤖</span>
        <div>
          <h1 className="text-xl font-bold text-white tracking-tight">AITrader</h1>
          <p className="text-xs text-slate-400">Autonomous AI Trading Bot</p>
        </div>
      </div>

      {/* Mode controls */}
      <div className="flex items-center gap-4">
        {/* Status badge */}
        <span
          className={`px-3 py-1 rounded-full text-xs font-semibold tracking-widest uppercase ${
            isLive
              ? 'bg-red-500/20 text-red-400 border border-red-500/40'
              : 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40'
          }`}
        >
          {isLive ? '🔴 LIVE' : '🟢 PAPER'}
        </span>

        {/* Toggle switch */}
        <div className="flex items-center gap-2 text-sm text-slate-300">
          <span className={!isLive ? 'text-emerald-400 font-semibold' : 'text-slate-500'}>
            Paper
          </span>
          <button
            onClick={handleToggle}
            className={`relative inline-flex h-6 w-12 items-center rounded-full transition-colors duration-300 focus:outline-none ${
              isLive ? 'bg-red-500' : 'bg-emerald-500'
            }`}
            aria-label="Toggle trading mode"
          >
            <span
              className={`inline-block h-4 w-4 rounded-full bg-white shadow transform transition-transform duration-300 ${
                isLive ? 'translate-x-7' : 'translate-x-1'
              }`}
            />
          </button>
          <span className={isLive ? 'text-red-400 font-semibold' : 'text-slate-500'}>
            Live
          </span>
        </div>
      </div>
    </header>
  )
}
