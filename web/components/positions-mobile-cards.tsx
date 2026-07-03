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
      <div className="space-y-1.5 px-2 py-2">
        {Array.from({ length: 5 }).map((_, i) => (
          <div key={i} className="h-12 bg-accent-subtle rounded animate-pulse" />
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
    <div className="space-y-1 px-2 py-2">
      {positions.map((pos) => (
        <div
          key={pos.symbol}
          className="border border-border rounded bg-background p-2 hover:bg-accent-hover transition-colors"
        >
          {/* Line 1: Symbol + Mode + Qty + Current P&L */}
          <div className="flex items-center justify-between mb-1">
            <div className="flex items-center gap-1.5">
              <span className="font-semibold text-foreground text-sm sm:text-base">{pos.symbol}</span>
              <span className="inline-block px-1 py-0 rounded text-[11px] sm:text-[13px] font-semibold bg-accent-subtle text-muted-foreground">
                {pos.mode === 'SWING' ? 'S' : 'I'}
              </span>
            </div>
            <div className="flex flex-col items-end gap-0.5">
              <span className="font-mono font-semibold text-foreground text-[12px] sm:text-[14px]">{pos.quantity} qty</span>
              <span className={`font-mono font-semibold text-[11px] sm:text-[13px] ${pos.current_pnl >= 0 ? 'text-positive' : 'text-negative'}`}>
                ₹{Math.abs(pos.current_pnl).toFixed(0)} ({pos.current_pnl_pct > 0 ? '+' : ''}{pos.current_pnl_pct.toFixed(2)}%)
              </span>
            </div>
          </div>

          {/* Line 2: TP / SL / Conf / Days */}
          <div className="flex items-center justify-between gap-1 text-[11px] sm:text-[13px] font-mono">
            <div className="text-positive min-w-0">
              TP {pos.tp_pct > 0 ? '+' : ''}{pos.tp_pct.toFixed(1)}%
            </div>
            <div className="h-2 w-px bg-border flex-shrink-0" />
            <div className="text-negative min-w-0">
              SL {pos.sl_pct.toFixed(1)}%
            </div>
            <div className="h-2 w-px bg-border flex-shrink-0" />
            <div className={`font-semibold flex-shrink-0 ${getConfidenceColor(pos.confidence_score)}`}>
              {pos.confidence_score}%
            </div>
            <div className="h-2 w-px bg-border flex-shrink-0" />
            <div className="text-muted-foreground flex-shrink-0">{pos.hold_days}d</div>
          </div>
        </div>
      ))}
    </div>
  );
}
