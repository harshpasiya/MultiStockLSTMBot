'use client';

import { TrendingUp, TrendingDown } from 'lucide-react';

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
  const pnlBg = totalUnrealisedPnl >= 0 ? 'bg-positive/10' : 'bg-negative/10';
  const isPositive = totalUnrealisedPnl >= 0;

  return (
    <div className="flex flex-wrap items-center justify-start gap-3 sm:gap-4 md:gap-5">
      {/* Open Positions */}
      <div className="flex items-baseline gap-1">
        <span className="text-[10px] sm:text-xs md:text-sm font-semibold uppercase text-muted-foreground">Open</span>
        <span className="text-base sm:text-lg md:text-xl font-bold text-foreground">{count}</span>
      </div>

      <div className="w-px h-4 bg-border/50" />

      {/* Total Invested */}
      <div className="flex items-baseline gap-1">
        <span className="text-[10px] sm:text-xs md:text-sm font-semibold uppercase text-muted-foreground">Invested</span>
        <span className="font-mono text-base sm:text-lg md:text-xl font-bold text-foreground">
          ₹{(totalInvested / 100000).toFixed(2)}L
        </span>
      </div>

      <div className="w-px h-4 bg-border/50" />

      {/* Current P&L */}
      <div className="flex items-baseline gap-1">
        <div className="flex items-center gap-1">
          <span className="text-[10px] sm:text-xs md:text-sm font-semibold uppercase text-muted-foreground">P&L</span>
          {isPositive ? (
            <TrendingUp className="h-3 w-3 sm:h-3.5 sm:w-3.5 text-positive" />
          ) : (
            <TrendingDown className="h-3 w-3 sm:h-3.5 sm:w-3.5 text-negative" />
          )}
        </div>
        <span className={`font-mono text-base sm:text-lg md:text-xl font-bold ${pnlColor}`}>
          {isPositive ? '+' : ''}₹{Math.abs(totalUnrealisedPnl).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
        </span>
        <span className={`text-[9px] sm:text-[10px] md:text-xs ${pnlColor}`}>
          ({isPositive ? '+' : ''}{(totalUnrealisedPnl > 0 ? 2.06 : -1.5).toFixed(1)}%)
        </span>
      </div>
    </div>
  );
}
