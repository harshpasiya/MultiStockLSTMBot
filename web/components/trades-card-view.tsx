'use client';

import { Skeleton } from './skeleton';

interface Trade {
  symbol: string;
  mode: string;
  side: string;
  entry_price: number;
  exit_price: number;
  quantity: number;
  realised_pnl: number;
  exit_reason: string;
  hold_days: number;
  confidence_score: number;
  entry_time: string;
  exit_time: string;
}

interface TradesCardViewProps {
  trades: Trade[];
  loading: boolean;
}

export function TradesCardView({ trades, loading }: TradesCardViewProps) {
  if (loading) {
    return (
      <div className="space-y-3">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-40 w-full rounded-lg" />
        ))}
      </div>
    );
  }

  if (trades.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-6 text-center">
        <p className="text-muted-foreground">No trades found. Try adjusting your filters.</p>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {trades.map((trade, idx) => (
        <div
          key={idx}
          className="rounded-lg border border-border bg-card p-4 space-y-3 transition-colors hover:bg-accent-hover"
        >
          {/* Header: Symbol and P&L */}
          <div className="flex items-center justify-between">
            <div>
              <p className="text-lg font-mono font-bold text-foreground">{trade.symbol}</p>
              <p className="text-xs text-muted-foreground">
                {new Date(trade.entry_time).toLocaleDateString()}
              </p>
            </div>
            <span
              className={`px-2 py-1 rounded font-mono font-bold text-sm ${
                trade.realised_pnl >= 0
                  ? 'bg-positive/20 text-positive'
                  : 'bg-negative/20 text-negative'
              }`}
            >
              {trade.realised_pnl >= 0 ? '+' : ''}${trade.realised_pnl.toFixed(2)}
            </span>
          </div>

          {/* Mode and Entry/Exit */}
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-xs text-muted-foreground">Mode</p>
              <p className="text-sm font-semibold text-foreground">{trade.mode}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Entry → Exit</p>
              <p className="text-sm font-mono text-foreground">
                ${trade.entry_price.toFixed(2)} → ${trade.exit_price.toFixed(2)}
              </p>
            </div>
          </div>

          {/* Details Row */}
          <div className="grid grid-cols-3 gap-2 border-t border-border pt-3">
            <div>
              <p className="text-xs text-muted-foreground">Days</p>
              <p className="text-sm font-mono font-semibold text-foreground">{trade.hold_days}</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Confidence</p>
              <p className="text-sm font-mono font-semibold text-foreground">
                {trade.confidence_score}%
              </p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Reason</p>
              <p className="text-xs font-semibold text-foreground truncate">{trade.exit_reason}</p>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
