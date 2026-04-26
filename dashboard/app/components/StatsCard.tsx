import { ReactNode } from 'react'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

interface StatsCardProps {
  title: string
  value: string | number
  icon: ReactNode
  positive?: boolean | null
  subtitle?: string
}

export default function StatsCard({ title, value, icon, positive, subtitle }: StatsCardProps) {
  const valueColor =
    positive === true
      ? 'text-emerald-400'
      : positive === false
      ? 'text-red-400'
      : 'text-white'

  const glowClass =
    positive === true
      ? 'shadow-emerald-500/10'
      : positive === false
      ? 'shadow-red-500/10'
      : ''

  const iconBg =
    positive === true
      ? 'bg-emerald-500/10 text-emerald-400'
      : positive === false
      ? 'bg-red-500/10 text-red-400'
      : 'bg-slate-700/60 text-slate-400'

  const TrendIcon =
    positive === true ? TrendingUp : positive === false ? TrendingDown : Minus

  return (
    <div
      className={`relative bg-[#161b27] border border-slate-800 rounded-2xl p-5 flex flex-col gap-3 shadow-lg ${glowClass} hover:border-slate-600 hover:-translate-y-0.5 transition-all duration-200 overflow-hidden`}
    >
      {/* subtle top-left glow orb */}
      {positive !== null && positive !== undefined && (
        <div
          className={`absolute -top-6 -left-6 w-24 h-24 rounded-full blur-2xl opacity-20 pointer-events-none ${
            positive ? 'bg-emerald-500' : 'bg-red-500'
          }`}
        />
      )}

      <div className="flex items-center justify-between">
        <div className={`p-2.5 rounded-xl ${iconBg}`}>{icon}</div>
        <TrendIcon
          size={14}
          className={`${
            positive === true
              ? 'text-emerald-400'
              : positive === false
              ? 'text-red-400'
              : 'text-slate-600'
          }`}
        />
      </div>

      <div>
        <p className="text-xs text-slate-500 uppercase tracking-widest font-medium mb-1">
          {title}
        </p>
        <p className={`text-2xl font-bold tracking-tight ${valueColor}`}>{value}</p>
        {subtitle && (
          <p className="text-xs text-slate-600 mt-1">{subtitle}</p>
        )}
      </div>
    </div>
  )
}
