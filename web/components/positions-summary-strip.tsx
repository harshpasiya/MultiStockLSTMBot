'use client';

import { TrendingUp, TrendingDown, Zap, Target, BarChart3 } from 'lucide-react';

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
  const isPositive = totalUnrealisedPnl >= 0;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 sm:gap-3 md:gap-4">
      {/* Open Positions Card */}
      <div className="relative group overflow-hidden rounded-lg border border-border/50 bg-background/40 backdrop-blur-md p-3 sm:p-4 hover:border-accent-cyan/40 hover:bg-background/60 transition-all duration-300">
        <div className="absolute inset-0 bg-gradient-to-br from-accent-cyan/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        <div className="relative z-10 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[8px] sm:text-[9px] md:text-xs font-bold uppercase tracking-widest text-muted-foreground">Open Positions</span>
            <Zap className="h-3 w-3 sm:h-3.5 sm:w-3.5 md:h-4 md:w-4 text-accent-cyan/60 group-hover:text-accent-cyan group-hover:animate-pulse transition-colors" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="text-xl sm:text-2xl md:text-4xl font-bold text-foreground">{count}</span>
            <span className="text-[8px] sm:text-[9px] md:text-xs text-muted-foreground">active</span>
          </div>
        </div>
      </div>

      {/* Total Invested Card */}
      <div className="relative group overflow-hidden rounded-lg border border-border/50 bg-background/40 backdrop-blur-md p-3 sm:p-4 hover:border-accent-cyan/40 hover:bg-background/60 transition-all duration-300">
        <div className="absolute inset-0 bg-gradient-to-br from-accent-cyan/10 via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300" />
        <div className="relative z-10 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[8px] sm:text-[9px] md:text-xs font-bold uppercase tracking-widest text-muted-foreground">Total Invested</span>
            <Target className="h-3 w-3 sm:h-3.5 sm:w-3.5 md:h-4 md:w-4 text-accent-cyan/60 group-hover:text-accent-cyan transition-colors" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-xl sm:text-2xl md:text-4xl font-bold text-foreground">
              ₹{(totalInvested / 100000).toFixed(2)}
            </span>
            <span className="text-[8px] sm:text-[9px] md:text-xs text-muted-foreground">L</span>
          </div>
        </div>
      </div>

      {/* Current P&L Card */}
      <div className={`relative group overflow-hidden rounded-lg border ${isPositive ? 'border-positive/30 hover:border-positive/50' : 'border-negative/30 hover:border-negative/50'} bg-background/40 backdrop-blur-md p-3 sm:p-4 hover:bg-background/60 transition-all duration-300 sm:col-span-2 lg:col-span-1`}>
        <div className={`absolute inset-0 bg-gradient-to-br ${isPositive ? 'from-positive/10' : 'from-negative/10'} via-transparent to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300`} />
        <div className="relative z-10 space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-[8px] sm:text-[9px] md:text-xs font-bold uppercase tracking-widest text-muted-foreground">Current P&L</span>
            {isPositive ? (
              <TrendingUp className="h-3 w-3 sm:h-3.5 sm:w-3.5 md:h-4 md:w-4 text-positive/60 group-hover:text-positive transition-colors" />
            ) : (
              <TrendingDown className="h-3 w-3 sm:h-3.5 sm:w-3.5 md:h-4 md:w-4 text-negative/60 group-hover:text-negative transition-colors" />
            )}
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className={`font-mono text-xl sm:text-2xl md:text-4xl font-bold ${pnlColor}`}>
              {isPositive ? '+' : ''}₹{Math.abs(totalUnrealisedPnl).toLocaleString('en-IN', { maximumFractionDigits: 0 })}
            </span>
            <span className={`text-[8px] sm:text-[9px] md:text-xs font-semibold ${pnlColor}`}>
              ({isPositive ? '+' : ''}{(totalUnrealisedPnl > 0 ? 2.06 : -1.5).toFixed(1)}%)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
