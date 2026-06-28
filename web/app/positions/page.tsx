'use client';

import { useEffect, useState } from 'react';
import { PositionsSummaryStrip } from '@/components/positions-summary-strip';
import { PositionsTable } from '@/components/positions-table';
import { PositionsMobileCards } from '@/components/positions-mobile-cards';

interface Position {
  symbol: string;
  entry_price: number;
  quantity: number;
  position_value: number;
  tp_price: number;
  sl_price: number;
  tp_pct: number;
  sl_pct: number;
  hold_days: number;
  confidence_score: number;
  mode: 'SWING' | 'INTRADAY';
  side: 'LONG' | 'SHORT';
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
      position_value: 3961.25,
      tp_price: 175.2,
      sl_price: 145.3,
      tp_pct: 10.59,
      sl_pct: -8.27,
      hold_days: 3,
      confidence_score: 85,
      mode: 'SWING',
      side: 'LONG',
      entry_time: '2024-06-24T10:30:00Z',
    },
    {
      symbol: 'MSFT',
      entry_price: 420.75,
      quantity: 15,
      position_value: 6311.25,
      tp_price: 445.0,
      sl_price: 408.5,
      tp_pct: 5.77,
      sl_pct: -2.90,
      hold_days: 2,
      confidence_score: 72,
      mode: 'SWING',
      side: 'LONG',
      entry_time: '2024-06-25T09:15:00Z',
    },
    {
      symbol: 'NVDA',
      entry_price: 132.1,
      quantity: 40,
      position_value: 5284.0,
      tp_price: 145.8,
      sl_price: 122.5,
      tp_pct: 10.40,
      sl_pct: -7.27,
      hold_days: 5,
      confidence_score: 92,
      mode: 'SWING',
      side: 'LONG',
      entry_time: '2024-06-20T14:45:00Z',
    },
    {
      symbol: 'TSLA',
      entry_price: 245.3,
      quantity: 20,
      position_value: 4906.0,
      tp_price: 258.5,
      sl_price: 235.0,
      tp_pct: 5.36,
      sl_pct: -4.22,
      hold_days: 1,
      confidence_score: 68,
      mode: 'INTRADAY',
      side: 'LONG',
      entry_time: '2024-06-26T10:00:00Z',
    },
    {
      symbol: 'GOOGL',
      entry_price: 197.8,
      quantity: 30,
      position_value: 5934.0,
      tp_price: 215.2,
      sl_price: 185.5,
      tp_pct: 8.78,
      sl_pct: -6.19,
      hold_days: 4,
      confidence_score: 78,
      mode: 'SWING',
      side: 'LONG',
      entry_time: '2024-06-21T11:20:00Z',
    },
    {
      symbol: 'META',
      entry_price: 512.4,
      quantity: 10,
      position_value: 5124.0,
      tp_price: 560.0,
      sl_price: 490.0,
      tp_pct: 9.28,
      sl_pct: -4.36,
      hold_days: 2,
      confidence_score: 81,
      mode: 'SWING',
      side: 'LONG',
      entry_time: '2024-06-25T13:30:00Z',
    },
  ],
};

export default function PositionsPage() {
  const [data, setData] = useState<PositionsResponse | null>(null);
  const [loading, setLoading] = useState(true);

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
        setData(MOCK_POSITIONS);
      } finally {
        setLoading(false);
      }
    }

    fetchPositions();
  }, []);

  if (loading) {
    return (
      <div className="flex h-screen flex-col bg-background">
        <div className="h-16 border-b border-border bg-background animate-pulse" />
        <div className="flex-1 space-y-0.5 overflow-y-auto">
          {Array.from({ length: 8 }).map((_, i) => (
            <div key={i} className="h-12 border-b border-border bg-accent-subtle/20" />
          ))}
        </div>
      </div>
    );
  }

  if (!data || data.count === 0) {
    return (
      <div className="flex h-screen flex-col items-center justify-center bg-background">
        <div className="text-center">
          <div className="mb-2 text-4xl opacity-20">◯</div>
          <p className="text-sm text-muted-foreground">No open positions</p>
        </div>
      </div>
    );
  }

  const totalInvested = data.positions.reduce((sum, p) => sum + p.position_value, 0);
  const totalUnrealisedPnl = data.positions.reduce((sum, p) => {
    const currentValue = p.position_value * (1 + (p.tp_pct / 100) * 0.5); // Placeholder midpoint
    return sum + (currentValue - p.position_value);
  }, 0);

  return (
    <div className="flex flex-col bg-background min-h-screen">
      {/* Header */}
      <div className="border-b border-border">
        <div className="px-6 py-4">
          <h1 className="text-2xl font-bold text-foreground">Positions</h1>
        </div>
        <PositionsSummaryStrip
          count={data.count}
          totalInvested={totalInvested}
          totalUnrealisedPnl={totalUnrealisedPnl}
        />
      </div>

      {/* Desktop Table View */}
      <div className="hidden lg:block flex-1 overflow-y-auto">
        <PositionsTable positions={data.positions} loading={false} />
      </div>

      {/* Mobile Card View */}
      <div className="lg:hidden flex-1 overflow-y-auto">
        <PositionsMobileCards positions={data.positions} loading={false} />
      </div>
    </div>
  );
}
