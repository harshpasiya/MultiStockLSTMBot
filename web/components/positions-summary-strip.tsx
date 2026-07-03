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
    <div className="px-0 py-0 bg-transparent">
      <div className="grid grid-cols-3 gap-1 sm:gap-2 md:gap-3">
        {/* Open Positions Card */}
        <div className="rounded-md bg-background/30 border border-border/30 p-1.5 sm:p-2 md:p-2.5">
          <div className="flex items-center justify-between gap-1 mb-0.5">
            <span className="text-[8px] sm:text-[9px] md:text-[10px] font-semibold uppercase text-muted-foreground truncate">
              Open
            </span>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="text-lg sm:text-xl md:text-2xl font-bold text-foreground">{count}</span>
          </div>
        </div>

        {/* Total Invested Card */}
        <div className="rounded-md bg-background/30 border border-border/30 p-1.5 sm:p-2 md:p-2.5">
          <div className="flex items-center justify-between gap-1 mb-0.5">
            <span className="text-[8px] sm:text-[9px] md:text-[10px] font-semibold uppercase text-muted-foreground truncate">
              Invested
            </span>
          </div>
          <div className="flex items-baseline gap-1">
            <span className="font-mono text-lg sm:text-xl md:text-2xl font-bold text-foreground truncate">
              ₹{(totalInvested / 100000).toFixed(2)}L
            </span>
          </div>
        </div>

        {/* Current P&L Card */}
        <div className={`rounded-md border border-border/30 p-1.5 sm:p-2 md:p-2.5 ${pnlBg.replace('/10', '/20')}`}>
          <div className="flex items-center justify-between gap-1 mb-0.5">
            <span className="text-[8px] sm:text-[9px] md:text-[10px] font-semibold uppercase text-muted-foreground truncate">
              P&L
            </span>
            {isPositive ? (
              <TrendingUp className="h-2.5 w-2.5 sm:h-3 sm:w-3 text-positive flex-shrink-0" />
            ) : (
              <TrendingDown className="h-2.5 w-2.5 sm:h-3 sm:w-3 text-negative flex-shrink-0" />
            )}
          </div>
          <div className="flex items-baseline gap-0.5">
            <span className={`font-mono text-lg sm:text-xl md:text-2xl font-bold ${pnlColor} truncate`}>
              {isPositive ? '+' : ''}₹{Math.abs(totalUnrealisedPnl).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
            </span>
            <span className={`text-[7px] sm:text-[8px] md:text-[9px] ${pnlColor} flex-shrink-0`}>
              {(totalUnrealisedPnl > 0 ? 2.06 : -1.5).toFixed(1)}%
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
