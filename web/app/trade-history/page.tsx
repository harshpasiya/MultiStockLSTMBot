'use client';

export default function TradeHistoryPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Trade History</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Review all executed trades and their performance metrics.
        </p>
      </div>

      <div className="card-glow p-6 rounded-lg bg-card border border-card-border overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-card-border">
              <th className="text-left py-3 px-4 text-muted-foreground font-semibold">Date</th>
              <th className="text-left py-3 px-4 text-muted-foreground font-semibold">Symbol</th>
              <th className="text-left py-3 px-4 text-muted-foreground font-semibold">Type</th>
              <th className="text-left py-3 px-4 text-muted-foreground font-semibold">Entry</th>
              <th className="text-left py-3 px-4 text-muted-foreground font-semibold">Exit</th>
              <th className="text-left py-3 px-4 text-muted-foreground font-semibold">P&L</th>
            </tr>
          </thead>
          <tbody>
            {[
              { date: '2026-06-27', symbol: 'BTC/USD', type: 'LONG', entry: '45,120', exit: '45,890', pnl: '+$770' },
              { date: '2026-06-27', symbol: 'ETH/USD', type: 'LONG', entry: '2,810', exit: '2,895', pnl: '+$85' },
              { date: '2026-06-26', symbol: 'SOL/USD', type: 'SHORT', entry: '112', exit: '108', pnl: '+$60' },
              { date: '2026-06-26', symbol: 'ADA/USD', type: 'LONG', entry: '0.65', exit: '0.62', pnl: '-$45' },
            ].map((trade, idx) => (
              <tr key={idx} className="border-b border-card-border/50 hover:bg-secondary/20">
                <td className="py-3 px-4 text-muted-foreground">{trade.date}</td>
                <td className="py-3 px-4 font-mono font-semibold text-foreground">{trade.symbol}</td>
                <td className="py-3 px-4">
                  <span className={`px-2 py-1 rounded text-xs font-semibold ${
                    trade.type === 'LONG' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                  }`}>
                    {trade.type}
                  </span>
                </td>
                <td className="py-3 px-4 font-mono text-sm">{trade.entry}</td>
                <td className="py-3 px-4 font-mono text-sm">{trade.exit}</td>
                <td className={`py-3 px-4 font-mono font-bold ${
                  trade.pnl.startsWith('+') ? 'text-green-400' : 'text-red-400'
                }`}>
                  {trade.pnl}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
