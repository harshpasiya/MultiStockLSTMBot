'use client';

import { useEffect, useState } from 'react';
import { PositionsSummaryStrip } from '@/components/positions-summary-strip';
import { PositionsTable } from '@/components/positions-table';
import { PositionsMobileCards } from '@/components/positions-mobile-cards';

interface Position {
  symbol: string;
  entry_price: number;
  current_price: number;
  quantity: number;
  position_value: number;
  current_pnl: number;
  current_pnl_pct: number;
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
      current_price: 165.20,
      quantity: 25,
      position_value: 3961.25,
      current_pnl: 168.75,
      current_pnl_pct: 4.26,
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
      current_price: 432.40,
      quantity: 15,
      position_value: 6311.25,
      current_pnl: 174.75,
      current_pnl_pct: 2.77,
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
      current_price: 140.85,
      quantity: 40,
      position_value: 5284.0,
      current_pnl: 350.0,
      current_pnl_pct: 6.62,
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
      current_price: 242.15,
      quantity: 20,
      position_value: 4906.0,
      current_pnl: -63.0,
      current_pnl_pct: -1.29,
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
      current_price: 208.15,
      quantity: 30,
      position_value: 5934.0,
      current_pnl: 310.5,
      current_pnl_pct: 5.23,
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
      current_price: 525.80,
      quantity: 10,
      position_value: 5124.0,
      current_pnl: 134.0,
      current_pnl_pct: 2.61,
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
    <div className="space-y-2 sm:space-y-3 md:space-y-4">
      <div>
        <h1 className="mb-0.5 text-lg font-bold sm:mb-1 sm:text-2xl md:text-3xl text-foreground">Positions</h1>
        <p className="text-[12px] sm:text-sm md:text-base text-muted-foreground">
          Open positions and targets
        </p>
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
      {/* Header Section */}
      <div className="space-y-3 sm:space-y-4 md:space-y-5 px-2 sm:px-4 md:px-6 py-3 sm:py-4 md:py-6">
        {/* Title Area */}
        <div>
          <h1 className="mb-1 text-xl sm:text-2xl md:text-4xl font-bold text-foreground tracking-tight">
            Positions
          </h1>
          <p className="text-[11px] sm:text-xs md:text-sm text-muted-foreground">
            Monitor and manage your active trading positions in real-time
          </p>
        </div>

        {/* Summary Cards */}
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
