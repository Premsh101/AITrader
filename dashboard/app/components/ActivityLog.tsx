import { ActivityEntry } from '../lib/api'

const AGENT_META: Record<string, { color: string; bg: string; icon: string }> = {
  Hunter:    { color: 'text-amber-300',   bg: 'bg-amber-500/10 border-amber-500/20',   icon: '🎯' },
  Guardian:  { color: 'text-sky-300',     bg: 'bg-sky-500/10 border-sky-500/20',       icon: '🛡️' },
  Executive: { color: 'text-violet-300',  bg: 'bg-violet-500/10 border-violet-500/20', icon: '📊' },
  System:    { color: 'text-slate-400',   bg: 'bg-slate-700/30 border-slate-700/40',   icon: '⚙️' },
}

interface ActivityLogProps {
  entries: ActivityEntry[]
}

export default function ActivityLog({ entries }: ActivityLogProps) {
  return (
    <div className="bg-[#161b27] border border-slate-800 rounded-2xl overflow-hidden shadow-lg flex flex-col">
      {/* Header */}
      <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="relative flex h-2.5 w-2.5">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-60" />
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500" />
          </span>
          <h2 className="font-semibold text-slate-200 text-sm">Live Activity Log</h2>
        </div>
        <span className="text-xs text-slate-600 bg-slate-800/60 px-2 py-0.5 rounded-full">
          {entries.length} entries
        </span>
      </div>

      {/* Entries */}
      <div className="overflow-y-auto max-h-80 divide-y divide-slate-800/60 flex-1">
        {entries.length === 0 ? (
          <div className="px-5 py-10 text-center text-slate-600 text-sm flex flex-col items-center gap-2">
            <span className="text-2xl">⏳</span>
            Waiting for first trading cycle…
          </div>
        ) : (
          entries.map((entry, i) => {
            const meta = AGENT_META[entry.agent] ?? {
              color: 'text-slate-300',
              bg: 'bg-slate-700/20 border-slate-700/30',
              icon: '•',
            }
            const time = new Date(entry.timestamp).toLocaleTimeString()
            return (
              <div
                key={i}
                className="px-4 py-2.5 flex items-start gap-3 hover:bg-white/[0.02] transition-colors"
              >
                <span className="text-sm mt-0.5 shrink-0">{meta.icon}</span>
                <div className="flex-1 min-w-0">
                  <span
                    className={`inline-block text-[10px] font-semibold uppercase tracking-widest px-1.5 py-0.5 rounded border ${meta.bg} ${meta.color} mb-1`}
                  >
                    {entry.agent}
                  </span>
                  <p className="text-xs text-slate-300 leading-relaxed">{entry.message}</p>
                </div>
                <time className="text-[10px] text-slate-600 whitespace-nowrap mt-0.5 shrink-0">
                  {time}
                </time>
              </div>
            )
          })
        )}
      </div>
    </div>
  )
}
