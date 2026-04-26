import { ReactNode } from 'react'

interface StatsCardProps {
  title: string
  value: string | number
  icon: ReactNode
  positive?: boolean | null
}

export default function StatsCard({ title, value, icon, positive }: StatsCardProps) {
  const valueColor =
    positive === true
      ? 'text-emerald-400'
      : positive === false
      ? 'text-red-400'
      : 'text-white'

  return (
    <div className="bg-[#161b27] border border-slate-700 rounded-xl p-5 flex items-center gap-4 shadow-sm hover:border-slate-500 transition-colors">
      <div className="p-3 rounded-lg bg-slate-700/50 text-slate-300">{icon}</div>
      <div>
        <p className="text-xs text-slate-400 uppercase tracking-wider mb-1">{title}</p>
        <p className={`text-2xl font-bold ${valueColor}`}>{value}</p>
      </div>
    </div>
  )
}
