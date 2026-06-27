'use client';

import { ChevronUp, ChevronDown } from 'lucide-react';
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

interface TradesTableProps {
  trades: Trade[];
  loading: boolean;
  sortColumn: string;
  sortDirection: 'asc' | 'desc';
  onSort: (column: string) => void;
}

export function TradesTable({
  trades,
  loading,
  sortColumn,
  sortDirection,
  onSort,
}: TradesTableProps) {
  if (loading) {
    return (
      <div className="space-y-2 rounded-lg border border-border bg-card p-4">
        {Array.from({ length: 10 }).map((_, i) => (
          <Skeleton key={i} className="h-12 w-full rounded" />
        ))}
      </div>
    );
  }

  if (trades.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-8 text-center">
        <p className="text-muted-foreground">No trades found. Try adjusting your filters.</p>
      </div>
    );
  }

  const SortHeader = ({ column, label }: { column: string; label: string }) => (
    <button
      onClick={() => onSort(column)}
      className="flex items-center gap-1 hover:text-foreground transition-colors"
    >
      {label}
      {sortColumn === column && (
        sortDirection === 'asc' ? (
          <ChevronUp className="h-3 w-3" />
        ) : (
          <ChevronDown className="h-3 w-3" />
        )
      )}
    </button>
  );

  return (
    <div className="overflow-x-auto rounded-lg border border-border bg-card">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
              <SortHeader column="entry_time" label="Date" />
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
              <SortHeader column="symbol" label="Symbol" />
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Mode</th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
              <SortHeader column="entry_price" label="Entry" />
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
              <SortHeader column="exit_price" label="Exit" />
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
              <SortHeader column="realised_pnl" label="P&L" />
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Reason</th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">
              <SortHeader column="hold_days" label="Days" />
            </th>
            <th className="px-4 py-3 text-left text-xs font-semibold text-muted-foreground">Confidence</th>
          </tr>
        </thead>
        <tbody>
          {trades.map((trade, idx) => (
            <tr
              key={idx}
              className="border-b border-border/50 transition-colors hover:bg-accent-hover"
            >
              <td className="px-4 py-3 text-muted-foreground text-xs">
                {new Date(trade.entry_time).toLocaleDateString()}
              </td>
              <td className="px-4 py-3 font-mono font-semibold text-foreground">{trade.symbol}</td>
              <td className="px-4 py-3">
                <span
                  className={`inline-block px-2 py-1 rounded text-xs font-semibold ${
                    trade.mode === 'SWING'
                      ? 'bg-accent-subtle text-foreground'
                      : 'bg-accent-subtle text-foreground'
                  }`}
                >
                  {trade.mode}
                </span>
              </td>
              <td className="px-4 py-3 font-mono text-sm text-foreground">
                ${trade.entry_price.toFixed(2)}
              </td>
              <td className="px-4 py-3 font-mono text-sm text-foreground">
                ${trade.exit_price.toFixed(2)}
              </td>
              <td className="px-4 py-3">
                <span
                  className={`inline-block px-2 py-1 rounded font-mono font-semibold text-xs ${
                    trade.realised_pnl >= 0
                      ? 'bg-positive/20 text-positive'
                      : 'bg-negative/20 text-negative'
                  }`}
                >
                  {trade.realised_pnl >= 0 ? '+' : ''}
                  ${trade.realised_pnl.toFixed(2)}
                </span>
              </td>
              <td className="px-4 py-3 text-muted-foreground text-sm truncate max-w-xs">
                {trade.exit_reason}
              </td>
              <td className="px-4 py-3 font-mono text-sm text-foreground">{trade.hold_days}</td>
              <td className="px-4 py-3">
                <div className="flex items-center gap-2">
                  <div className="text-xs font-mono text-foreground w-8 text-right">
                    {trade.confidence_score}%
                  </div>
                  <div className="w-16 h-1.5 rounded-full bg-border overflow-hidden">
                    <div
                      className={`h-full rounded-full transition-all ${
                        trade.confidence_score >= 70
                          ? 'bg-positive'
                          : trade.confidence_score >= 50
                            ? 'bg-accent-subtle'
                            : 'bg-negative'
                      }`}
                      style={{ width: `${trade.confidence_score}%` }}
                    />
                  </div>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
