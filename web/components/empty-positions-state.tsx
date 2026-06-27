'use client';

import { TrendingUp } from 'lucide-react';

export function EmptyPositionsState() {
  return (
    <div className="flex min-h-[600px] flex-col items-center justify-center rounded-lg border border-border bg-card/50 px-6 py-12">
      {/* Icon container */}
      <div className="mb-4 flex h-16 w-16 items-center justify-center rounded-full border-2 border-accent-subtle bg-accent-subtle/50">
        <TrendingUp className="h-8 w-8 text-muted-foreground" />
      </div>

      {/* Heading */}
      <h3 className="mb-2 text-lg font-semibold text-foreground">No Open Positions</h3>

      {/* Description */}
      <p className="max-w-sm text-center text-sm text-muted-foreground">
        You don&apos;t have any open positions at the moment. Start trading or wait for your trading system to identify new opportunities.
      </p>

      {/* Subtle action hint */}
      <div className="mt-6 text-xs text-muted-foreground/60">
        Positions will appear here when they are opened
      </div>
    </div>
  );
}
