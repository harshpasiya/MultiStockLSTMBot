interface Position {
  symbol: string;
  entry_price: number;
  quantity: number;
  position_value: number;
  tp_price: number;
  sl_price: number;
  tp_pct: number;
  sl_pct: number;
  hold_days: number;
  confidence_score: number;
  mode: 'SWING' | 'INTRADAY';
  side: 'LONG' | 'SHORT';
}

interface PositionsTableProps {
  positions: Position[];
  loading: boolean;
}

export function PositionsTable({ positions, loading }: PositionsTableProps) {
  if (loading) {
    return (
      <div className="space-y-0.5 px-6 py-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-10 bg-accent-subtle rounded animate-pulse" />
        ))}
      </div>
    );
  }

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 75) return 'text-positive';
    if (confidence >= 60) return 'text-muted';
    return 'text-negative';
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="border-b border-border bg-background sticky top-0">
          <tr className="text-muted-foreground text-xs font-medium uppercase tracking-wide">
            <th className="px-6 py-3 text-left">Symbol</th>
            <th className="px-6 py-3 text-right">Qty</th>
            <th className="px-6 py-3 text-right">Avg Price</th>
            <th className="px-6 py-3 text-right">Current P&L</th>
            <th className="px-6 py-3 text-right">TP</th>
            <th className="px-6 py-3 text-right">SL</th>
            <th className="px-6 py-3 text-right">Day</th>
            <th className="px-6 py-3 text-right">Conf</th>
          </tr>
        </thead>
        <tbody>
          {positions.map((pos, idx) => (
            <tr
              key={pos.symbol}
              className={`h-12 border-b border-border hover:bg-accent-hover transition-colors ${
                idx % 2 === 0 ? 'bg-background' : 'bg-accent-subtle/30'
              }`}
            >
              {/* Symbol + Mode */}
              <td className="px-6 py-3">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-foreground">{pos.symbol}</span>
                  <span className="inline-block px-1.5 py-0.5 rounded text-xs font-medium bg-accent-subtle text-muted-foreground">
                    {pos.mode === 'SWING' ? 'S' : 'I'}
                  </span>
                </div>
              </td>

              {/* Quantity */}
              <td className="px-6 py-3 text-right font-mono text-foreground">{pos.quantity}</td>

              {/* Avg Price */}
              <td className="px-6 py-3 text-right font-mono text-muted-foreground">
                ₹{pos.entry_price.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
              </td>

              {/* Current P&L */}
              <td className="px-6 py-3 text-right font-mono">
                <div className={pos.current_pnl >= 0 ? 'text-positive' : 'text-negative'}>
                  ₹{Math.abs(pos.current_pnl).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                  <span className="text-xs ml-1">({pos.current_pnl_pct > 0 ? '+' : ''}{pos.current_pnl_pct.toFixed(2)}%)</span>
                </div>
              </td>

              {/* TP Price + % */}
              <td className="px-6 py-3 text-right font-mono">
                <div className="text-positive">
                  ₹{pos.tp_price.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                  <span className="text-xs"> (+{pos.tp_pct.toFixed(2)}%)</span>
                </div>
              </td>

              {/* SL Price + % */}
              <td className="px-6 py-3 text-right font-mono">
                <div className="text-negative">
                  ₹{pos.sl_price.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
                  <span className="text-xs"> ({pos.sl_pct.toFixed(2)}%)</span>
                </div>
              </td>

              {/* Days Held */}
              <td className="px-6 py-3 text-right font-mono text-muted-foreground text-sm">
                {pos.hold_days}d
              </td>

              {/* Confidence */}
              <td className={`px-6 py-3 text-right font-mono text-sm font-medium ${getConfidenceColor(pos.confidence_score)}`}>
                {pos.confidence_score}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
