interface PositionsSummaryStripProps {
  count: number;
  totalInvested: number;
  totalUnrealisedPnl: number;
}

export function PositionsSummaryStrip({
  count,
  totalInvested,
  totalUnrealisedPnl,
}: PositionsSummaryStripProps) {
  const pnlColor = totalUnrealisedPnl >= 0 ? 'text-positive' : 'text-negative';

  return (
    <div className="border-b border-border px-6 py-3">
      <div className="flex items-center justify-start gap-8 text-sm">
        <div className="flex items-baseline gap-2">
          <span className="text-muted-foreground">{count}</span>
          <span className="text-foreground font-medium">Open Position{count !== 1 ? 's' : ''}</span>
        </div>

        <div className="h-4 w-px bg-border" />

        <div className="flex items-baseline gap-2">
          <span className="text-muted-foreground">Total Invested</span>
          <span className="font-mono font-semibold text-foreground">
            ₹{totalInvested.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
          </span>
        </div>

        <div className="h-4 w-px bg-border" />

        <div className="flex items-baseline gap-2">
          <span className="text-muted-foreground">Unrealised P&L</span>
          <span className={`font-mono font-semibold ${pnlColor}`}>
            ₹{totalUnrealisedPnl.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
          </span>
        </div>
      </div>
    </div>
  );
}
