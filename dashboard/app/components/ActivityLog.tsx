import { ActivityEntry } from '../lib/api'

const AGENT_COLORS: Record<string, string> = {
  Hunter: 'text-amber-400',
  Guardian: 'text-blue-400',
  Executive: 'text-purple-400',
  System: 'text-slate-400',
}

const AGENT_ICONS: Record<string, string> = {
  Hunter: '🎯',
  Guardian: '🛡️',
  Executive: '📊',
  System: '⚙️',
}

interface ActivityLogProps {
  entries: ActivityEntry[]
}

export default function ActivityLog({ entries }: ActivityLogProps) {
  return (
    <div className="bg-[#161b27] border border-slate-700 rounded-xl overflow-hidden shadow-sm">
      <div className="px-5 py-3 border-b border-slate-700 flex items-center justify-between">
        <h2 className="font-semibold text-slate-200">⚡ Live Activity Log</h2>
        <span className="text-xs text-slate-500">{entries.length} entries</span>
      </div>

      <div className="overflow-y-auto max-h-80 divide-y divide-slate-800">
        {entries.length === 0 ? (
          <div className="px-5 py-8 text-center text-slate-500 text-sm">
            Waiting for first trading cycle…
          </div>
        ) : (
          entries.map((entry, i) => {
            const color = AGENT_COLORS[entry.agent] ?? 'text-slate-300'
            const icon = AGENT_ICONS[entry.agent] ?? '•'
            const time = new Date(entry.timestamp).toLocaleTimeString()
            return (
              <div key={i} className="px-5 py-2.5 flex items-start gap-3 hover:bg-slate-800/40 transition-colors">
                <span className="text-base mt-0.5">{icon}</span>
                <div className="flex-1 min-w-0">
                  <span className={`text-xs font-semibold uppercase ${color}`}>
                    {entry.agent}
                  </span>
                  <p className="text-sm text-slate-300 leading-snug">{entry.message}</p>
                </div>
                <time className="text-xs text-slate-600 whitespace-nowrap mt-0.5">{time}</time>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
