'use client';

import { Search, Calendar } from 'lucide-react';
import { useCallback } from 'react';

interface TradesFilterBarProps {
  onSymbolChange: (symbol: string) => void;
  onStatusChange: (status: 'all' | 'open' | 'closed') => void;
  onDateRangeChange: (from: string, to: string) => void;
  onExport: () => void;
  selectedStatus: 'all' | 'open' | 'closed';
  selectedSymbol: string;
  isExporting: boolean;
}

export function TradesFilterBar({
  onSymbolChange,
  onStatusChange,
  onDateRangeChange,
  onExport,
  selectedStatus,
  selectedSymbol,
  isExporting,
}: TradesFilterBarProps) {
  const handleDateFromChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const to = (document.getElementById('date-to') as HTMLInputElement)?.value || '';
      onDateRangeChange(e.target.value, to);
    },
    [onDateRangeChange]
  );

  const handleDateToChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const from = (document.getElementById('date-from') as HTMLInputElement)?.value || '';
      onDateRangeChange(from, e.target.value);
    },
    [onDateRangeChange]
  );

  return (
    <div className="space-y-4 rounded-lg border border-border bg-card p-4">
      <div className="flex flex-col gap-3 lg:flex-row lg:items-end lg:gap-4">
        {/* Symbol Search */}
        <div className="flex-1">
          <label className="block text-xs font-semibold text-muted-foreground mb-2">Symbol</label>
          <div className="relative">
            <Search className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
            <input
              type="text"
              placeholder="e.g. AAPL, BTC"
              value={selectedSymbol}
              onChange={(e) => onSymbolChange(e.target.value.toUpperCase())}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 pl-9 text-sm text-foreground placeholder-muted-foreground focus:border-accent-subtle focus:outline-none"
            />
          </div>
        </div>

        {/* Date Range */}
        <div className="flex gap-2">
          <div className="flex-1 sm:flex-initial">
            <label className="block text-xs font-semibold text-muted-foreground mb-2">From</label>
            <div className="relative">
              <Calendar className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                id="date-from"
                type="date"
                onChange={handleDateFromChange}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 pl-9 text-sm text-foreground focus:border-accent-subtle focus:outline-none sm:w-auto"
              />
            </div>
          </div>
          <div className="flex-1 sm:flex-initial">
            <label className="block text-xs font-semibold text-muted-foreground mb-2">To</label>
            <div className="relative">
              <Calendar className="absolute left-3 top-2.5 h-4 w-4 text-muted-foreground" />
              <input
                id="date-to"
                type="date"
                onChange={handleDateToChange}
                className="w-full rounded-lg border border-border bg-background px-3 py-2 pl-9 text-sm text-foreground focus:border-accent-subtle focus:outline-none sm:w-auto"
              />
            </div>
          </div>
        </div>

        {/* Status Dropdown */}
        <div>
          <label className="block text-xs font-semibold text-muted-foreground mb-2">Status</label>
          <select
            value={selectedStatus}
            onChange={(e) => onStatusChange(e.target.value as 'all' | 'open' | 'closed')}
            className="rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground focus:border-accent-subtle focus:outline-none"
          >
            <option value="all">All Trades</option>
            <option value="open">Open</option>
            <option value="closed">Closed</option>
          </select>
        </div>

        {/* Export Button */}
        <button
          onClick={onExport}
          disabled={isExporting}
          className="rounded-lg border border-accent-subtle bg-accent-subtle px-3 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent-hover disabled:opacity-50"
        >
          {isExporting ? 'Exporting...' : 'Export CSV'}
        </button>
      </div>
    </div>
  );
}
