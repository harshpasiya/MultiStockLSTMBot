'use client';

import { useEffect, useState } from 'react';
import { PositionCard } from '@/components/position-card';
import { EmptyPositionsState } from '@/components/empty-positions-state';
import { Skeleton } from '@/components/skeleton';

interface Position {
  symbol: string;
  entry_price: number;
  quantity: number;
  tp_price: number;
  sl_price: number;
  tp_pct: number;
  sl_pct: number;
  hold_days: number;
  confidence_score: number;
  mode: 'SWING' | 'INTRADAY';
  entry_time?: string;
}

interface PositionsResponse {
  count: number;
  positions: Position[];
}

// Mock data for development/testing
const MOCK_POSITIONS: PositionsResponse = {
  count: 6,
  positions: [
    {
      symbol: 'AAPL',
      entry_price: 158.45,
      quantity: 25,
      tp_price: 175.2,
      sl_price: 145.3,
      tp_pct: 10.59,
      sl_pct: -8.27,
      hold_days: 3,
      confidence_score: 85,
      mode: 'SWING',
      entry_time: '2024-06-24T10:30:00Z',
    },
    {
      symbol: 'MSFT',
      entry_price: 420.75,
      quantity: 15,
      tp_price: 445.0,
      sl_price: 408.5,
      tp_pct: 5.77,
      sl_pct: -2.90,
      hold_days: 2,
      confidence_score: 72,
      mode: 'SWING',
      entry_time: '2024-06-25T09:15:00Z',
    },
    {
      symbol: 'NVDA',
      entry_price: 132.1,
      quantity: 40,
      tp_price: 145.8,
      sl_price: 122.5,
      tp_pct: 10.40,
      sl_pct: -7.27,
      hold_days: 5,
      confidence_score: 92,
      mode: 'SWING',
      entry_time: '2024-06-20T14:45:00Z',
    },
    {
      symbol: 'TSLA',
      entry_price: 245.3,
      quantity: 20,
      tp_price: 258.5,
      sl_price: 235.0,
      tp_pct: 5.36,
      sl_pct: -4.22,
      hold_days: 1,
      confidence_score: 68,
      mode: 'INTRADAY',
      entry_time: '2024-06-26T10:00:00Z',
    },
    {
      symbol: 'GOOGL',
      entry_price: 197.8,
      quantity: 30,
      tp_price: 215.2,
      sl_price: 185.5,
      tp_pct: 8.78,
      sl_pct: -6.19,
      hold_days: 4,
      confidence_score: 78,
      mode: 'SWING',
      entry_time: '2024-06-21T11:20:00Z',
    },
    {
      symbol: 'META',
      entry_price: 512.4,
      quantity: 10,
      tp_price: 560.0,
      sl_price: 490.0,
      tp_pct: 9.28,
      sl_pct: -4.36,
      hold_days: 2,
      confidence_score: 81,
      mode: 'SWING',
      entry_time: '2024-06-25T13:30:00Z',
    },
  ],
};

export default function PositionsPage() {
  const [data, setData] = useState<PositionsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchPositions() {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL;
        if (!apiUrl) {
          console.log('[v0] No API URL configured, using mock data');
          setData(MOCK_POSITIONS);
          setLoading(false);
          return;
        }

        const res = await fetch(`${apiUrl}/api/positions`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) {
          throw new Error(`API error: ${res.status}`);
        }

        const positionsData: PositionsResponse = await res.json();
        setData(positionsData);
      } catch (err) {
        console.error('[v0] Failed to fetch positions:', err);
        // Fallback to mock data on error
        setData(MOCK_POSITIONS);
      } finally {
        setLoading(false);
      }
    }

    fetchPositions();
  }, []);

  // Loading state
  if (loading) {
    return (
      <div className="space-y-6 p-6">
        <div>
          <h1 className="mb-2 text-3xl font-bold text-foreground">Positions</h1>
          <p className="text-sm text-muted-foreground">Track all your active trading positions</p>
        </div>

        {/* Loading skeleton grid */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Skeleton key={i} className="h-72 w-full rounded-lg" />
          ))}
        </div>
      </div>
    );
  }

  // Error state (though we fallback to mock data)
  if (error) {
    return (
      <div className="space-y-6 p-6">
        <div>
          <h1 className="mb-2 text-3xl font-bold text-foreground">Positions</h1>
          <p className="text-sm text-muted-foreground">Track all your active trading positions</p>
        </div>
        <div className="rounded-lg border border-negative/30 bg-negative/10 p-4">
          <p className="text-sm text-negative">Error loading positions. Showing mock data instead.</p>
        </div>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {MOCK_POSITIONS.positions.map((position) => (
            <PositionCard
              key={position.symbol}
              symbol={position.symbol}
              entryPrice={position.entry_price}
              quantity={position.quantity}
              tpPrice={position.tp_price}
              slPrice={position.sl_price}
              tpPct={position.tp_pct}
              slPct={position.sl_pct}
              holdDays={position.hold_days}
              confidenceScore={position.confidence_score}
              mode={position.mode}
              entryTime={position.entry_time}
            />
          ))}
        </div>
      </div>
    );
  }

  // Empty state
  if (!data || data.count === 0) {
    return (
      <div className="space-y-6 p-6">
        <div>
          <h1 className="mb-2 text-3xl font-bold text-foreground">Positions</h1>
          <p className="text-sm text-muted-foreground">Track all your active trading positions</p>
        </div>
        <EmptyPositionsState />
      </div>
    );
  }

  // Positions grid
  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="mb-2 text-3xl font-bold text-foreground">Positions</h1>
        <p className="text-sm text-muted-foreground">
          {data.count} active position{data.count !== 1 ? 's' : ''}
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {data.positions.map((position) => (
          <PositionCard
            key={position.symbol}
            symbol={position.symbol}
            entryPrice={position.entry_price}
            quantity={position.quantity}
            tpPrice={position.tp_price}
            slPrice={position.sl_price}
            tpPct={position.tp_pct}
            slPct={position.sl_pct}
            holdDays={position.hold_days}
            confidenceScore={position.confidence_score}
            mode={position.mode}
            entryTime={position.entry_time}
          />
        ))}
      </div>
    </div>
  );
}
