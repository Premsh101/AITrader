import { Trade } from '../lib/api'

interface TradeTableProps {
  trades: Trade[]
}

export default function TradeTable({ trades }: TradeTableProps) {
  return (
    <div className="bg-[#161b27] border border-slate-700 rounded-xl overflow-hidden shadow-sm">
      <div className="px-5 py-3 border-b border-slate-700 flex items-center justify-between">
        <h2 className="font-semibold text-slate-200">📋 Trade History</h2>
        <span className="text-xs text-slate-500">{trades.length} records</span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-slate-500 uppercase tracking-wider border-b border-slate-700">
              <th className="px-5 py-3 text-left">ID</th>
              <th className="px-5 py-3 text-left">Symbol</th>
              <th className="px-5 py-3 text-right">Qty</th>
              <th className="px-5 py-3 text-right">Buy ₹</th>
              <th className="px-5 py-3 text-right">Sell ₹</th>
              <th className="px-5 py-3 text-right">P&amp;L ₹</th>
              <th className="px-5 py-3 text-center">Status</th>
              <th className="px-5 py-3 text-center">Mode</th>
              <th className="px-5 py-3 text-right">Date</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {trades.length === 0 ? (
              <tr>
                <td colSpan={9} className="px-5 py-10 text-center text-slate-500">
                  No trades yet
                </td>
              </tr>
            ) : (
              trades.map((t) => {
                const pnl = t.pnl != null ? parseFloat(t.pnl) : null
                const pnlColor =
                  pnl === null
                    ? 'text-slate-400'
                    : pnl > 0
                    ? 'text-emerald-400 font-semibold'
                    : pnl < 0
                    ? 'text-red-400 font-semibold'
                    : 'text-slate-400'

                const statusBadge =
                  t.status === 'Open'
                    ? 'bg-amber-500/20 text-amber-400 border border-amber-500/40'
                    : 'bg-slate-700 text-slate-400 border border-slate-600'

                const modeBadge =
                  t.mode === 'Live'
                    ? 'bg-red-500/20 text-red-400'
                    : 'bg-emerald-500/20 text-emerald-400'

                const date = t.created_at
                  ? new Date(t.created_at).toLocaleDateString()
                  : '—'

                return (
                  <tr key={t.id} className="hover:bg-slate-800/40 transition-colors">
                    <td className="px-5 py-3 text-slate-500 text-xs">#{t.id}</td>
                    <td className="px-5 py-3 font-medium text-white">{t.symbol.replace('.NS', '')}</td>
                    <td className="px-5 py-3 text-right text-slate-300">{t.quantity}</td>
                    <td className="px-5 py-3 text-right text-slate-300">
                      {t.buy_price ? `₹${parseFloat(t.buy_price).toFixed(2)}` : '—'}
                    </td>
                    <td className="px-5 py-3 text-right text-slate-300">
                      {t.sell_price ? `₹${parseFloat(t.sell_price).toFixed(2)}` : '—'}
                    </td>
                    <td className={`px-5 py-3 text-right ${pnlColor}`}>
                      {pnl !== null
                        ? `${pnl >= 0 ? '+' : ''}₹${pnl.toFixed(2)}`
                        : '—'}
                    </td>
                    <td className="px-5 py-3 text-center">
                      <span className={`px-2 py-0.5 rounded-full text-xs ${statusBadge}`}>
                        {t.status}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-center">
                      <span className={`px-2 py-0.5 rounded-full text-xs ${modeBadge}`}>
                        {t.mode}
                      </span>
                    </td>
                    <td className="px-5 py-3 text-right text-slate-500 text-xs">{date}</td>
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
