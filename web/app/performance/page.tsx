'use client';

import { useEffect, useState } from 'react';
import { BarChart3, Percent, Trophy, Zap } from 'lucide-react';
import { PerformanceMetricCard } from '@/components/performance-metrics-card';
import { EquityCurveChart } from '@/components/equity-curve-chart';
import { PerformanceExitReasonsChart } from '@/components/performance-exit-reasons-chart';
import { PerformanceMonthlyChart } from '@/components/performance-monthly-chart';
import { Skeleton } from '@/components/skeleton';

interface EquityPoint {
  date: string;
  value: number;
  pnl: number;
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

interface PerformanceMetrics {
  total_trades: number;
  total_wins: number;
  total_pnl: number;
  win_rate: number;
  profit_factor: number;
  avg_win: number;
  avg_loss: number;
  best_trade: number;
  worst_trade: number;
}

// Mock data for development/testing
const MOCK_EQUITY_CURVE: EquityPoint[] = [
  { date: 'Jun 01', value: 100000, pnl: 0 },
  { date: 'Jun 05', value: 102150, pnl: 2150.25 },
  { date: 'Jun 10', value: 104025, pnl: 4025.60 },
  { date: 'Jun 15', value: 103675, pnl: 3675.20 },
  { date: 'Jun 20', value: 106015, pnl: 6015.95 },
  { date: 'Jun 25', value: 107696, pnl: 7696.40 },
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

const MOCK_METRICS: PerformanceMetrics = {
  total_trades: 127,
  total_wins: 79,
  total_pnl: 7696.40,
  win_rate: 62.2,
  profit_factor: 2.14,
  avg_win: 97.42,
  avg_loss: 45.63,
  best_trade: 1260.50,
  worst_trade: -890.25,
};

export default function PerformancePage() {
  const [equityData, setEquityData] = useState<EquityPoint[]>([]);
  const [exitReasonData, setExitReasonData] = useState<ExitReasonData[]>([]);
  const [monthlyData, setMonthlyData] = useState<MonthlyData[]>([]);
  const [metrics, setMetrics] = useState<PerformanceMetrics>(MOCK_METRICS);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchPerformanceData() {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL;
        if (!apiUrl) {
          console.log('[v0] No API URL configured, using mock data');
          setEquityData(MOCK_EQUITY_CURVE);
          setExitReasonData(MOCK_EXIT_REASONS);
          setMonthlyData(MOCK_MONTHLY);
          setMetrics(MOCK_METRICS);
          setLoading(false);
          return;
        }

        const [equityRes, exitRes, monthlyRes] = await Promise.all([
          fetch(`${apiUrl}/api/equity-curve?days=90`, {
            headers: { 'Content-Type': 'application/json' },
          }),
          fetch(`${apiUrl}/api/performance/exit-reasons`, {
            headers: { 'Content-Type': 'application/json' },
          }),
          fetch(`${apiUrl}/api/performance/monthly?months=6`, {
            headers: { 'Content-Type': 'application/json' },
          }),
        ]);

        if (equityRes.ok) {
          const data = await equityRes.json();
          setEquityData(data.points || MOCK_EQUITY_CURVE);
        } else {
          setEquityData(MOCK_EQUITY_CURVE);
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
        setEquityData(MOCK_EQUITY_CURVE);
        setExitReasonData(MOCK_EXIT_REASONS);
        setMonthlyData(MOCK_MONTHLY);
      } finally {
        setLoading(false);
      }
    }

    fetchPerformanceData();
  }, []);

  const isPositiveWinRate = metrics.win_rate >= 50;
  const isPositivePnL = metrics.total_pnl >= 0;
  const isPositiveProfitFactor = metrics.profit_factor >= 1.5;

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="mb-2 text-3xl font-bold text-foreground">Performance</h1>
        <p className="text-sm text-muted-foreground">
          Comprehensive analytics of your trading performance and key metrics
        </p>
      </div>

      {/* Metrics Grid */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {loading ? (
          <>
            {Array.from({ length: 4 }).map((_, i) => (
              <Skeleton key={i} className="h-32 w-full rounded-lg" />
            ))}
          </>
        ) : (
          <>
            <PerformanceMetricCard
              label="Total P&L"
              value={`$${metrics.total_pnl.toFixed(2)}`}
              subtext={`${metrics.total_trades} trades`}
              isPositive={isPositivePnL}
              icon={<BarChart3 className="h-5 w-5 text-muted" />}
            />
            <PerformanceMetricCard
              label="Win Rate"
              value={`${metrics.win_rate.toFixed(1)}%`}
              subtext={`${metrics.total_wins} wins`}
              isPositive={isPositiveWinRate}
              icon={<Trophy className="h-5 w-5 text-muted" />}
            />
            <PerformanceMetricCard
              label="Profit Factor"
              value={metrics.profit_factor.toFixed(2)}
              subtext="Gross profit / Gross loss"
              isPositive={isPositiveProfitFactor}
              icon={<Percent className="h-5 w-5 text-muted" />}
            />
            <PerformanceMetricCard
              label="Avg Win / Loss"
              value={`$${metrics.avg_win.toFixed(0)} / $${Math.abs(
                metrics.avg_loss
              ).toFixed(0)}`}
              subtext="Risk/reward ratio"
              icon={<Zap className="h-5 w-5 text-muted" />}
            />
          </>
        )}
      </div>

      {/* Equity Curve Chart */}
      {loading ? (
        <Skeleton className="h-96 w-full rounded-lg" />
      ) : (
        <EquityCurveChart data={equityData} />
      )}

      {/* Charts Grid */}
      <div className="grid gap-6 lg:grid-cols-2">
        {loading ? (
          <>
            <Skeleton className="h-96 w-full rounded-lg" />
            <Skeleton className="h-96 w-full rounded-lg" />
          </>
        ) : (
          <>
            <PerformanceExitReasonsChart data={exitReasonData} loading={loading} />
            <PerformanceMonthlyChart data={monthlyData} loading={loading} />
          </>
        )}
      </div>
    </div>
  );
}
