'use client';

import { useEffect, useState } from 'react';
import { PerformanceSymbolChart } from '@/components/performance-symbol-chart';
import { PerformanceExitReasonsChart } from '@/components/performance-exit-reasons-chart';
import { PerformanceMonthlyChart } from '@/components/performance-monthly-chart';

interface SymbolData {
  symbol: string;
  trades: number;
  wins: number;
  total_pnl: number;
}

interface ExitReasonData {
  exit_reason: string;
  count: number;
  total_pnl: number;
}

interface MonthlyData {
  month: string;
  trades: number;
  wins: number;
  pnl: number;
}

// Mock data for development/testing
const MOCK_SYMBOLS: SymbolData[] = [
  { symbol: 'AAPL', trades: 8, wins: 6, total_pnl: 1350.45 },
  { symbol: 'MSFT', trades: 6, wins: 4, total_pnl: 745.20 },
  { symbol: 'NVDA', trades: 12, wins: 11, total_pnl: 2890.75 },
  { symbol: 'TSLA', trades: 5, wins: 3, total_pnl: -125.30 },
  { symbol: 'GOOGL', trades: 7, wins: 5, total_pnl: 1560.00 },
  { symbol: 'META', trades: 4, wins: 2, total_pnl: -450.15 },
];

const MOCK_EXIT_REASONS: ExitReasonData[] = [
  { exit_reason: 'Take Profit', count: 32, total_pnl: 4200.50 },
  { exit_reason: 'Stop Loss', count: 18, total_pnl: -890.25 },
  { exit_reason: 'Manual Exit', count: 12, total_pnl: 680.30 },
];

const MOCK_MONTHLY: MonthlyData[] = [
  { month: 'Jan', trades: 18, wins: 12, pnl: 2150.25 },
  { month: 'Feb', trades: 22, wins: 15, pnl: 1875.60 },
  { month: 'Mar', trades: 16, wins: 10, pnl: -350.40 },
  { month: 'Apr', trades: 20, wins: 14, pnl: 2340.75 },
  { month: 'May', trades: 14, wins: 9, pnl: 945.30 },
  { month: 'Jun', trades: 17, wins: 11, pnl: 1680.45 },
];

export default function PerformancePage() {
  const [symbolData, setSymbolData] = useState<SymbolData[]>([]);
  const [exitReasonData, setExitReasonData] = useState<ExitReasonData[]>([]);
  const [monthlyData, setMonthlyData] = useState<MonthlyData[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchPerformanceData() {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL;
        if (!apiUrl) {
          console.log('[v0] No API URL configured, using mock data');
          setSymbolData(MOCK_SYMBOLS);
          setExitReasonData(MOCK_EXIT_REASONS);
          setMonthlyData(MOCK_MONTHLY);
          setLoading(false);
          return;
        }

        const [symbolRes, exitRes, monthlyRes] = await Promise.all([
          fetch(`${apiUrl}/api/performance/by-symbol`, {
            headers: { 'Content-Type': 'application/json' },
          }),
          fetch(`${apiUrl}/api/performance/exit-reasons`, {
            headers: { 'Content-Type': 'application/json' },
          }),
          fetch(`${apiUrl}/api/performance/monthly?months=6`, {
            headers: { 'Content-Type': 'application/json' },
          }),
        ]);

        if (symbolRes.ok) {
          const data = await symbolRes.json();
          setSymbolData(data.symbols || MOCK_SYMBOLS);
        } else {
          setSymbolData(MOCK_SYMBOLS);
        }

        if (exitRes.ok) {
          const data = await exitRes.json();
          setExitReasonData(data.reasons || MOCK_EXIT_REASONS);
        } else {
          setExitReasonData(MOCK_EXIT_REASONS);
        }

        if (monthlyRes.ok) {
          const data = await monthlyRes.json();
          setMonthlyData(data.months || MOCK_MONTHLY);
        } else {
          setMonthlyData(MOCK_MONTHLY);
        }
      } catch (err) {
        console.error('[v0] Failed to fetch performance data:', err);
        setSymbolData(MOCK_SYMBOLS);
        setExitReasonData(MOCK_EXIT_REASONS);
        setMonthlyData(MOCK_MONTHLY);
      } finally {
        setLoading(false);
      }
    }

    fetchPerformanceData();
  }, []);

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="mb-2 text-3xl font-bold text-foreground">Performance</h1>
        <p className="text-sm text-muted-foreground">
          Analyze performance metrics and trading statistics across symbols, exit reasons, and time periods
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <PerformanceSymbolChart data={symbolData} loading={loading} />
        <PerformanceExitReasonsChart data={exitReasonData} loading={loading} />
      </div>

      <PerformanceMonthlyChart data={monthlyData} loading={loading} />
    </div>
  );
}
