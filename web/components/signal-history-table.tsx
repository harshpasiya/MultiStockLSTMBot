import { format } from 'date-fns';

interface HistorySignal {
  symbol: string;
  side: string;
  confidence_score: number;
  entry_time: string;
  outcome?: 'Win' | 'Loss' | 'Open';
  pnl?: number;
}

interface SignalHistoryTableProps {
  signals: HistorySignal[];
  loading: boolean;
}

export function SignalHistoryTable({ signals, loading }: SignalHistoryTableProps) {
  if (loading) {
    return (
      <div className="card-hover rounded-lg border border-border bg-card overflow-hidden">
        <div className="p-6">
          <div className="space-y-3">
            {Array.from({ length: 5 }).map((_, i) => (
              <div key={i} className="h-12 animate-pulse bg-accent-subtle rounded" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="card-hover rounded-lg border border-border bg-card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              <th className="px-6 py-4 text-left text-xs uppercase font-semibold text-muted-foreground">Date</th>
              <th className="px-6 py-4 text-left text-xs uppercase font-semibold text-muted-foreground">Symbol</th>
              <th className="px-6 py-4 text-left text-xs uppercase font-semibold text-muted-foreground">Action</th>
              <th className="px-6 py-4 text-left text-xs uppercase font-semibold text-muted-foreground">Confidence</th>
              <th className="px-6 py-4 text-left text-xs uppercase font-semibold text-muted-foreground">Outcome</th>
              <th className="px-6 py-4 text-right text-xs uppercase font-semibold text-muted-foreground">P&L</th>
            </tr>
          </thead>
          <tbody>
            {signals.map((signal, idx) => (
              <tr key={idx} className="border-b border-border/50 hover:bg-accent-hover transition-colors">
                <td className="px-6 py-4 text-muted-foreground">
                  {format(new Date(signal.entry_time), 'MMM dd, HH:mm')}
                </td>
                <td className="px-6 py-4 font-mono font-semibold text-foreground">{signal.symbol}</td>
                <td className="px-6 py-4">
                  <span
                    className={`inline-block px-3 py-1 rounded text-xs font-semibold ${
                      signal.side === 'BUY' || signal.side === 'STRONG_BUY'
                        ? 'bg-positive/20 text-positive'
                        : signal.side === 'SELL' || signal.side === 'STRONG_SELL'
                          ? 'bg-negative/20 text-negative'
                          : 'bg-accent-subtle text-muted-foreground'
                    }`}
                  >
                    {signal.side}
                  </span>
                </td>
                <td className="px-6 py-4">
                  <span
                    className={`inline-block px-3 py-1 rounded text-xs font-semibold ${
                      signal.confidence_score >= 75
                        ? 'bg-positive/20 text-positive'
                        : signal.confidence_score >= 60
                          ? 'bg-yellow-500/20 text-yellow-400'
                          : 'bg-negative/20 text-negative'
                    }`}
                  >
                    {signal.confidence_score}%
                  </span>
                </td>
                <td className="px-6 py-4">
                  {signal.outcome ? (
                    <span
                      className={`text-xs font-semibold ${
                        signal.outcome === 'Win'
                          ? 'text-positive'
                          : signal.outcome === 'Loss'
                            ? 'text-negative'
                            : 'text-muted-foreground'
                      }`}
                    >
                      {signal.outcome}
                    </span>
                  ) : (
                    <span className="text-xs text-muted-foreground">—</span>
                  )}
                </td>
                <td className="px-6 py-4 text-right font-mono font-semibold">
                  {signal.pnl !== undefined ? (
                    <span className={signal.pnl >= 0 ? 'text-positive' : 'text-negative'}>
                      {signal.pnl >= 0 ? '+' : ''}${signal.pnl.toFixed(2)}
                    </span>
                  ) : (
                    <span className="text-muted-foreground">—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
