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

interface PositionsMobileCardsProps {
  positions: Position[];
  loading: boolean;
}

export function PositionsMobileCards({ positions, loading }: PositionsMobileCardsProps) {
  if (loading) {
    return (
      <div className="space-y-3 px-4 py-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-16 bg-accent-subtle rounded animate-pulse" />
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
    <div className="space-y-2 px-4 py-4">
      {positions.map((pos) => (
        <div
          key={pos.symbol}
          className="border border-border rounded bg-background p-3 hover:bg-accent-hover transition-colors"
        >
          {/* Line 1: Symbol + Mode + Qty */}
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <span className="font-semibold text-foreground">{pos.symbol}</span>
              <span className="inline-block px-1.5 py-0.5 rounded text-xs font-medium bg-accent-subtle text-muted-foreground">
                {pos.mode === 'SWING' ? 'S' : 'I'}
              </span>
            </div>
            <span className="font-mono font-semibold text-foreground">{pos.quantity} qty</span>
          </div>

          {/* Line 2: TP / SL / Conf / Days */}
          <div className="flex items-center justify-between gap-2 text-xs font-mono">
            <div className="text-positive">
              TP ₹{pos.tp_price.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
              <span> (+{pos.tp_pct.toFixed(1)}%)</span>
            </div>
            <div className="h-3 w-px bg-border" />
            <div className="text-negative">
              SL ₹{pos.sl_price.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
              <span> ({pos.sl_pct.toFixed(1)}%)</span>
            </div>
            <div className="h-3 w-px bg-border" />
            <div className={`font-semibold ${getConfidenceColor(pos.confidence_score)}`}>
              {pos.confidence_score}%
            </div>
            <div className="h-3 w-px bg-border" />
            <div className="text-muted-foreground">{pos.hold_days}d</div>
          </div>
        </div>
      ))}
    </div>
  );
}
