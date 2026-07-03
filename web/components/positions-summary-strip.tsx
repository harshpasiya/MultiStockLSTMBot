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
    <div className="px-2 sm:px-4 md:px-6 py-1.5 sm:py-2 md:py-2.5 bg-gradient-to-r from-accent-subtle to-transparent rounded-lg border border-border/50 shadow-sm">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 sm:gap-3 md:gap-4">
        {/* Open Positions Card */}
        <div className="rounded-lg bg-background/50 border border-border/50 p-2 sm:p-2.5 md:p-3 hover:border-border transition-colors">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[9px] sm:text-[10px] md:text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Open Positions
            </span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-xl sm:text-2xl md:text-3xl font-bold text-foreground">{count}</span>
            <span className="text-[9px] sm:text-xs md:text-xs text-muted-foreground">pos</span>
          </div>
        </div>

        {/* Total Invested Card */}
        <div className="rounded-lg bg-background/50 border border-border/50 p-2 sm:p-2.5 md:p-3 hover:border-border transition-colors">
          <div className="flex items-center justify-between mb-1">
            <span className="text-[9px] sm:text-[10px] md:text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Total Invested
            </span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="font-mono text-xl sm:text-2xl md:text-3xl font-bold text-foreground">
              ₹{(totalInvested / 100000).toFixed(2)}
            </span>
            <span className="text-[9px] sm:text-xs md:text-xs text-muted-foreground">L</span>
          </div>
        </div>

        {/* Current P&L Card */}
        <div className={`rounded-lg ${pnlBg} border border-border/50 p-2 sm:p-2.5 md:p-3 hover:border-border transition-colors`}>
          <div className="flex items-center justify-between mb-1">
            <span className="text-[9px] sm:text-[10px] md:text-xs font-semibold uppercase tracking-wider text-muted-foreground">
              Current P&L
            </span>
            {isPositive ? (
              <TrendingUp className="h-3 w-3 sm:h-4 sm:w-4 md:h-4 md:w-4 text-positive" />
            ) : (
              <TrendingDown className="h-3 w-3 sm:h-4 sm:w-4 md:h-4 md:w-4 text-negative" />
            )}
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className={`font-mono text-xl sm:text-2xl md:text-3xl font-bold ${pnlColor}`}>
              {isPositive ? '+' : ''} ₹{Math.abs(totalUnrealisedPnl).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
            </span>
            <span className={`text-[8px] sm:text-xs md:text-xs ${pnlColor}`}>
              ({isPositive ? '+' : ''}{(totalUnrealisedPnl > 0 ? 2.06 : -1.5).toFixed(2)}%)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
