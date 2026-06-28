'use client';

import { TrendingUp, TrendingDown } from 'lucide-react';
import { formatCurrency, formatPercent } from '@/lib/utils';

interface TradeData {
  trade_id?: string;
  pnl: number;
  return_pct: number;
  entry_price: number;
  exit_price: number;
  quantity: number;
}

interface TradeSummaryCardsProps {
  bestTrade: TradeData | null;
  worstTrade: TradeData | null;
}

export function TradeSummaryCards({
  bestTrade,
  worstTrade,
}: TradeSummaryCardsProps) {
  return (
    <section className="grid grid-cols-1 gap-4 sm:grid-cols-2">
      {/* Best Trade */}
      <div className="card-hover rounded-xl border border-border bg-card p-5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Best Trade
          </span>
          <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-positive bg-positive/10">
            <TrendingUp className="h-4 w-4 text-positive" />
          </span>
        </div>

        {bestTrade ? (
          <>
            <p className="font-mono text-2xl font-semibold tracking-tight mt-4 text-positive">
              {formatCurrency(bestTrade.pnl)}
            </p>
            <div className="mt-3 text-xs text-muted-foreground space-y-1">
              <p>
                <span className="text-muted-foreground">Entry:</span>{' '}
                <span className="font-mono">{formatCurrency(bestTrade.entry_price)}</span>
              </p>
              <p>
                <span className="text-muted-foreground">Exit:</span>{' '}
                <span className="font-mono">{formatCurrency(bestTrade.exit_price)}</span>
              </p>
              <p>
                <span className="text-muted-foreground">Return:</span>{' '}
                <span className="font-mono text-positive">
                  {formatPercent(bestTrade.return_pct)}
                </span>
              </p>
            </div>
          </>
        ) : (
          <p className="mt-4 text-sm text-muted-foreground">No trades yet</p>
        )}
      </div>

      {/* Worst Trade */}
      <div className="card-hover rounded-xl border border-border bg-card p-5">
        <div className="flex items-center justify-between">
          <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
            Worst Trade
          </span>
          <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-negative bg-negative/10">
            <TrendingDown className="h-4 w-4 text-negative" />
          </span>
        </div>

        {worstTrade ? (
          <>
            <p className="font-mono text-2xl font-semibold tracking-tight mt-4 text-negative">
              {formatCurrency(worstTrade.pnl)}
            </p>
            <div className="mt-3 text-xs text-muted-foreground space-y-1">
              <p>
                <span className="text-muted-foreground">Entry:</span>{' '}
                <span className="font-mono">{formatCurrency(worstTrade.entry_price)}</span>
              </p>
              <p>
                <span className="text-muted-foreground">Exit:</span>{' '}
                <span className="font-mono">{formatCurrency(worstTrade.exit_price)}</span>
              </p>
              <p>
                <span className="text-muted-foreground">Return:</span>{' '}
                <span className="font-mono text-negative">
                  {formatPercent(worstTrade.return_pct)}
                </span>
              </p>
            </div>
          </>
        ) : (
          <p className="mt-4 text-sm text-muted-foreground">No trades yet</p>
        )}
      </div>
    </section>
  );
}
