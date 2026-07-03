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
    <div className="px-2 sm:px-4 md:px-6 py-3 sm:py-4 md:py-6 bg-gradient-to-r from-accent-subtle to-transparent rounded-lg border border-border/50 shadow-sm">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 md:gap-6">
        {/* Open Positions Card */}
        <div className="rounded-lg bg-background/50 backdrop-blur-sm border border-border/50 p-3 sm:p-4 md:p-5 hover:border-border transition-colors">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] sm:text-xs md:text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              Open Positions
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-2xl sm:text-3xl md:text-4xl font-bold text-foreground">{count}</span>
            <span className="text-xs sm:text-sm text-muted-foreground">positions</span>
          </div>
        </div>

        {/* Total Invested Card */}
        <div className="rounded-lg bg-background/50 backdrop-blur-sm border border-border/50 p-3 sm:p-4 md:p-5 hover:border-border transition-colors">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] sm:text-xs md:text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              Total Invested
            </span>
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-2xl sm:text-3xl md:text-4xl font-bold text-foreground">
              ₹{(totalInvested / 100000).toFixed(2)}
            </span>
            <span className="text-xs sm:text-sm text-muted-foreground">L</span>
          </div>
        </div>

        {/* Current P&L Card */}
        <div className={`rounded-lg ${pnlBg} backdrop-blur-sm border border-border/50 p-3 sm:p-4 md:p-5 hover:border-border transition-colors`}>
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] sm:text-xs md:text-sm font-semibold uppercase tracking-wider text-muted-foreground">
              Current P&L
            </span>
            {isPositive ? (
              <TrendingUp className="h-4 w-4 sm:h-5 sm:w-5 text-positive" />
            ) : (
              <TrendingDown className="h-4 w-4 sm:h-5 sm:w-5 text-negative" />
            )}
          </div>
          <div className="flex items-baseline gap-2">
            <span className={`font-mono text-2xl sm:text-3xl md:text-4xl font-bold ${pnlColor}`}>
              {isPositive ? '+' : ''} ₹{Math.abs(totalUnrealisedPnl).toLocaleString('en-IN', { maximumFractionDigits: 2 })}
            </span>
            <span className={`text-xs sm:text-sm ${pnlColor}`}>
              ({isPositive ? '+' : ''}{(totalUnrealisedPnl > 0 ? 2.06 : -1.5).toFixed(2)}%)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
