'use client'

import { toggleMode } from '../lib/api'
import { Zap } from 'lucide-react'

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
    <header className="sticky top-0 z-30 flex items-center justify-between px-6 py-4 bg-[#0d1117]/90 backdrop-blur border-b border-slate-800 shadow-lg">
      {/* Brand */}
      <div className="flex items-center gap-3">
        <div className="flex items-center justify-center w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 shadow-lg shadow-indigo-500/30">
          <Zap size={18} className="text-white" />
        </div>
        <div>
          <h1 className="text-lg font-bold text-white tracking-tight leading-none">AITrader</h1>
          <p className="text-[11px] text-slate-500 leading-none mt-0.5">Autonomous AI Trading System</p>
        </div>
      </div>

      {/* Mode controls */}
      <div className="flex items-center gap-4">
        {/* Animated status pill */}
        <div
          className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold tracking-wider uppercase border transition-all duration-500 ${
            isLive
              ? 'bg-red-500/10 text-red-400 border-red-500/30 shadow-red-500/20 shadow-sm'
              : 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30 shadow-emerald-500/20 shadow-sm'
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              isLive ? 'bg-red-400 animate-pulse' : 'bg-emerald-400'
            }`}
          />
          {isLive ? 'Live Trading' : 'Paper Mode'}
        </div>

        {/* Toggle */}
        <div className="flex items-center gap-2 text-sm text-slate-400">
          <span className={`text-xs transition-colors ${!isLive ? 'text-emerald-400 font-semibold' : 'text-slate-600'}`}>
            Paper
          </span>
          <button
            onClick={handleToggle}
            className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-[#0d1117] ${
              isLive ? 'bg-red-500 focus:ring-red-500' : 'bg-emerald-500 focus:ring-emerald-500'
            }`}
            aria-label="Toggle trading mode"
          >
            <span
              className={`inline-block h-4 w-4 rounded-full bg-white shadow-md transform transition-transform duration-300 ${
                isLive ? 'translate-x-6' : 'translate-x-1'
              }`}
            />
          </button>
          <span className={`text-xs transition-colors ${isLive ? 'text-red-400 font-semibold' : 'text-slate-600'}`}>
            Live
          </span>
        </div>
      </div>
    </header>
  )
}
