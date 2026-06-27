'use client';

import { useEffect, useState, useCallback } from 'react';
import { TradesFilterBar } from '@/components/trades-filter-bar';
import { TradesTable } from '@/components/trades-table';
import { TradesCardView } from '@/components/trades-card-view';
import { PaginationControls } from '@/components/pagination-controls';
import { exportTradesToCSV } from '@/lib/csv-export';

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

interface TradesResponse {
  total: number;
  trades: Trade[];
}

// Mock data for development/testing
const MOCK_TRADES: Trade[] = [
  {
    symbol: 'AAPL',
    mode: 'SWING',
    side: 'LONG',
    entry_price: 158.45,
    exit_price: 165.23,
    quantity: 25,
    realised_pnl: 169.50,
    exit_reason: 'Take Profit Hit',
    hold_days: 3,
    confidence_score: 85,
    entry_time: '2024-06-24T10:30:00Z',
    exit_time: '2024-06-27T14:15:00Z',
  },
  {
    symbol: 'MSFT',
    mode: 'SWING',
    side: 'LONG',
    entry_price: 420.75,
    exit_price: 428.90,
    quantity: 15,
    realised_pnl: 122.25,
    exit_reason: 'Stop Loss Hit',
    hold_days: 2,
    confidence_score: 72,
    entry_time: '2024-06-25T09:15:00Z',
    exit_time: '2024-06-27T11:45:00Z',
  },
  {
    symbol: 'NVDA',
    mode: 'SWING',
    side: 'LONG',
    entry_price: 132.10,
    exit_price: 145.80,
    quantity: 40,
    realised_pnl: 548.00,
    exit_reason: 'Take Profit Hit',
    hold_days: 5,
    confidence_score: 92,
    entry_time: '2024-06-20T14:45:00Z',
    exit_time: '2024-06-27T10:30:00Z',
  },
  {
    symbol: 'TSLA',
    mode: 'INTRADAY',
    side: 'SHORT',
    entry_price: 245.30,
    exit_price: 242.15,
    quantity: 20,
    realised_pnl: 63.00,
    exit_reason: 'Manual Exit',
    hold_days: 1,
    confidence_score: 68,
    entry_time: '2024-06-26T10:00:00Z',
    exit_time: '2024-06-26T15:30:00Z',
  },
  {
    symbol: 'GOOGL',
    mode: 'SWING',
    side: 'LONG',
    entry_price: 197.80,
    exit_price: 215.20,
    quantity: 30,
    realised_pnl: 522.00,
    exit_reason: 'Take Profit Hit',
    hold_days: 4,
    confidence_score: 78,
    entry_time: '2024-06-21T11:20:00Z',
    exit_time: '2024-06-27T09:45:00Z',
  },
  {
    symbol: 'META',
    mode: 'SWING',
    side: 'LONG',
    entry_price: 512.40,
    exit_price: 495.60,
    quantity: 10,
    realised_pnl: -168.00,
    exit_reason: 'Stop Loss Hit',
    hold_days: 2,
    confidence_score: 81,
    entry_time: '2024-06-25T13:30:00Z',
    exit_time: '2024-06-27T16:20:00Z',
  },
  {
    symbol: 'AMZN',
    mode: 'INTRADAY',
    side: 'LONG',
    entry_price: 187.25,
    exit_price: 192.50,
    quantity: 35,
    realised_pnl: 183.75,
    exit_reason: 'Take Profit Hit',
    hold_days: 1,
    confidence_score: 75,
    entry_time: '2024-06-27T09:00:00Z',
    exit_time: '2024-06-27T13:00:00Z',
  },
  {
    symbol: 'NFLX',
    mode: 'SWING',
    side: 'SHORT',
    entry_price: 625.80,
    exit_price: 610.40,
    quantity: 8,
    realised_pnl: 122.00,
    exit_reason: 'Take Profit Hit',
    hold_days: 3,
    confidence_score: 82,
    entry_time: '2024-06-24T15:45:00Z',
    exit_time: '2024-06-27T12:15:00Z',
  },
];

export default function TradeHistoryPage() {
  const [data, setData] = useState<TradesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [exporting, setExporting] = useState(false);

  // Filter state
  const [symbol, setSymbol] = useState('');
  const [status, setStatus] = useState<'all' | 'open' | 'closed'>('all');
  const [dateFrom, setDateFrom] = useState('');
  const [dateTo, setDateTo] = useState('');

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [pageSize, setPageSize] = useState(25);

  // Sorting state
  const [sortColumn, setSortColumn] = useState('entry_time');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // Fetch trades
  useEffect(() => {
    async function fetchTrades() {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL;
        if (!apiUrl) {
          console.log('[v0] No API URL configured, using mock data');
          setData({
            total: MOCK_TRADES.length,
            trades: MOCK_TRADES,
          });
          setLoading(false);
          return;
        }

        const params = new URLSearchParams({
          from_date: dateFrom || '2024-01-01',
          to_date: dateTo || new Date().toISOString().split('T')[0],
          status: status,
          limit: pageSize.toString(),
          offset: ((currentPage - 1) * pageSize).toString(),
        });

        const res = await fetch(`${apiUrl}/api/trades?${params}`, {
          method: 'GET',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!res.ok) {
          throw new Error(`API error: ${res.status}`);
        }

        const tradesData: TradesResponse = await res.json();
        setData(tradesData);
      } catch (err) {
        console.error('[v0] Failed to fetch trades:', err);
        // Fallback to mock data
        setData({
          total: MOCK_TRADES.length,
          trades: MOCK_TRADES,
        });
      } finally {
        setLoading(false);
      }
    }

    fetchTrades();
  }, [dateFrom, dateTo, status, currentPage, pageSize]);

  // Filter trades
  const filteredTrades = data?.trades.filter((trade) => {
    if (symbol && !trade.symbol.toUpperCase().includes(symbol.toUpperCase())) {
      return false;
    }
    return true;
  }) || [];

  // Sort trades
  const sortedTrades = [...filteredTrades].sort((a, b) => {
    let aVal: any = (a as any)[sortColumn];
    let bVal: any = (b as any)[sortColumn];

    if (typeof aVal === 'string') {
      aVal = aVal.toLowerCase();
      bVal = bVal.toLowerCase();
    }

    if (sortDirection === 'asc') {
      return aVal > bVal ? 1 : -1;
    } else {
      return aVal < bVal ? 1 : -1;
    }
  });

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('desc');
    }
  };

  const handleExport = async () => {
    setExporting(true);
    try {
      await new Promise((resolve) => setTimeout(resolve, 500));
      exportTradesToCSV(sortedTrades, 'zodiac-godseye-trades');
    } finally {
      setExporting(false);
    }
  };

  const totalPages = Math.ceil((data?.total || 0) / pageSize);
  const isSmallScreen = typeof window !== 'undefined' && window.innerWidth < 1024;

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="mb-2 text-3xl font-bold text-foreground">Trade History</h1>
        <p className="text-sm text-muted-foreground">
          Review all executed trades and their performance metrics
        </p>
      </div>

      {/* Filter Bar */}
      <TradesFilterBar
        selectedSymbol={symbol}
        onSymbolChange={setSymbol}
        selectedStatus={status}
        onStatusChange={setStatus}
        onDateRangeChange={(from, to) => {
          setDateFrom(from);
          setDateTo(to);
          setCurrentPage(1);
        }}
        onExport={handleExport}
        isExporting={exporting}
      />

      {/* Table or Card View */}
      <div className="hidden lg:block">
        <TradesTable
          trades={sortedTrades}
          loading={loading}
          sortColumn={sortColumn}
          sortDirection={sortDirection}
          onSort={handleSort}
        />
      </div>

      <div className="lg:hidden">
        <TradesCardView trades={sortedTrades} loading={loading} />
      </div>

      {/* Pagination */}
      {!loading && filteredTrades.length > 0 && (
        <PaginationControls
          currentPage={currentPage}
          totalPages={totalPages}
          totalCount={data?.total || 0}
          pageSize={pageSize}
          onPageChange={setCurrentPage}
          onPageSizeChange={(size) => {
            setPageSize(size);
            setCurrentPage(1);
          }}
        />
      )}
    </div>
  );
}
