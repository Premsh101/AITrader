import { Trade } from '../lib/api'

interface TradeTableProps {
  trades: Trade[]
}

export default function TradeTable({ trades }: TradeTableProps) {
  return (
    <div className="bg-[#161b27] border border-slate-800 rounded-2xl overflow-hidden shadow-lg">
      {/* Header */}
      <div className="px-5 py-3.5 border-b border-slate-800 flex items-center justify-between">
        <h2 className="font-semibold text-slate-200 text-sm">Trade History</h2>
        <span className="text-xs text-slate-600 bg-slate-800/60 px-2 py-0.5 rounded-full">
          {trades.length} records
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-900/40">
              <th className="px-5 py-3 text-left text-[11px] text-slate-500 uppercase tracking-widest font-medium">ID</th>
              <th className="px-5 py-3 text-left text-[11px] text-slate-500 uppercase tracking-widest font-medium">Symbol</th>
              <th className="px-5 py-3 text-right text-[11px] text-slate-500 uppercase tracking-widest font-medium">Qty</th>
              <th className="px-5 py-3 text-right text-[11px] text-slate-500 uppercase tracking-widest font-medium">Buy ₹</th>
              <th className="px-5 py-3 text-right text-[11px] text-slate-500 uppercase tracking-widest font-medium">Sell ₹</th>
              <th className="px-5 py-3 text-right text-[11px] text-slate-500 uppercase tracking-widest font-medium">P&amp;L</th>
              <th className="px-5 py-3 text-center text-[11px] text-slate-500 uppercase tracking-widest font-medium">Status</th>
              <th className="px-5 py-3 text-center text-[11px] text-slate-500 uppercase tracking-widest font-medium">Mode</th>
              <th className="px-5 py-3 text-right text-[11px] text-slate-500 uppercase tracking-widest font-medium">Date</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/60">
            {trades.length === 0 ? (
              <tr>
                <td colSpan={9} className="px-5 py-12 text-center">
                  <div className="flex flex-col items-center gap-2 text-slate-600">
                    <span className="text-3xl">📭</span>
                    <span className="text-sm">No trades yet</span>
                  </div>
                </td>
              </tr>
            ) : (
              trades.map((t) => {
                const pnl = t.pnl != null ? parseFloat(t.pnl) : null
                const pnlPositive = pnl !== null && pnl > 0
                const pnlColor =
                  pnl === null
                    ? 'text-slate-500'
                    : pnl > 0
                    ? 'text-emerald-400 font-semibold'
                    : pnl < 0
                    ? 'text-red-400 font-semibold'
                    : 'text-slate-400'

                const isOpen = t.status === 'Open'

                const statusBadge = isOpen
                  ? 'bg-amber-500/10 text-amber-300 border border-amber-500/20'
                  : 'bg-slate-800 text-slate-500 border border-slate-700'

                const modeBadge =
                  t.mode === 'Live'
                    ? 'bg-red-500/10 text-red-400 border border-red-500/20'
                    : 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'

                const date = t.created_at
                  ? new Date(t.created_at).toLocaleDateString()
                  : '—'

                return (
                  <tr
                    key={t.id}
                    className="hover:bg-white/[0.02] transition-colors group"
                  >
                    <td className="px-5 py-3 text-slate-600 text-xs font-mono">
                      #{t.id}
                    </td>
                    <td className="px-5 py-3">
                      <span className="font-semibold text-white text-sm">
                        {t.symbol.replace('.NS', '')}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-right text-slate-400 text-sm">
                      {t.quantity}
                    </td>
                    <td className="px-5 py-3 text-right text-slate-300 text-sm font-mono">
                      {t.buy_price ? `₹${parseFloat(t.buy_price).toFixed(2)}` : '—'}
                    </td>
                    <td className="px-5 py-3 text-right text-slate-300 text-sm font-mono">
                      {t.sell_price ? `₹${parseFloat(t.sell_price).toFixed(2)}` : '—'}
                    </td>
                    <td className={`px-5 py-3 text-right text-sm font-mono ${pnlColor}`}>
                      <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs ${
                        pnl !== null
                          ? pnlPositive
                            ? 'bg-emerald-500/10'
                            : pnl < 0
                            ? 'bg-red-500/10'
                            : ''
                          : ''
                      }`}>
                        {pnl !== null
                          ? `${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}`
                          : '—'}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-center">
                      <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${statusBadge}`}>
                        {isOpen ? (
                          <span className="flex items-center gap-1">
                            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 animate-pulse inline-block" />
                            {t.status}
                          </span>
                        ) : t.status}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-center">
                      <span className={`px-2 py-0.5 rounded-full text-[11px] font-medium ${modeBadge}`}>
                        {t.mode}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-right text-slate-600 text-xs">{date}</td>
                  </tr>
                )
              })
            )}
          </tbody>
        </table>
      </div>
    </div>
  )
}
