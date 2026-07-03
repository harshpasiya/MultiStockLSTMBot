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
    <div className="border-b border-border px-2 sm:px-4 md:px-6 py-2 sm:py-3">
      <div className="flex flex-wrap items-center justify-start gap-2 sm:gap-4 md:gap-8 text-[11px] sm:text-sm">
        <div className="flex items-baseline gap-1 sm:gap-2">
          <span className="text-muted-foreground">{count}</span>
          <span className="text-foreground font-medium">Open</span>
        </div>

        <div className="h-3 w-px bg-border" />

        <div className="flex items-baseline gap-1 sm:gap-2">
          <span className="text-muted-foreground text-[10px] sm:text-xs">Invested</span>
          <span className="font-mono font-semibold text-foreground text-xs sm:text-sm">
            ₹{totalInvested.toLocaleString('en-IN', { maximumFractionDigits: 0 })}
          </span>
        </div>

        <div className="h-3 w-px bg-border" />

        <div className="flex items-baseline gap-1 sm:gap-2">
          <span className="text-muted-foreground text-[10px] sm:text-xs">P&L</span>
          <span className={`font-mono font-semibold text-xs sm:text-sm ${pnlColor}`}>
            ₹{totalUnrealisedPnl.toLocaleString('en-IN', { maximumFractionDigits: 2 })}
          </span>
        </div>
      </div>
    </div>
  );
}
