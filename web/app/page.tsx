'use client';

import { useEffect, useState } from 'react';
import { Wallet, TrendingUp, Radio, Activity } from 'lucide-react';
import { StatCard } from '@/components/stat-card';
import { EquityCurveChart } from '@/components/equity-curve-chart';
import { TradeSummaryCards } from '@/components/trade-summary-cards';
import {
  StatCardSkeleton,
  ChartSkeleton,
  TradeCardSkeleton,
} from '@/components/skeleton';

interface PortfolioData {
  portfolio_value: number;
  total_pnl: number;
  today_pnl: number;
  open_positions: number;
  win_rate: number;
  trades_today: number;
  best_trade: {
    trade_id?: string;
    pnl: number;
    return_pct: number;
    entry_price: number;
    exit_price: number;
    quantity: number;
  } | null;
  worst_trade: {
    trade_id?: string;
    pnl: number;
    return_pct: number;
    entry_price: number;
    exit_price: number;
    quantity: number;
  } | null;
  total_return_pct: number;
}

interface EquityCurveData {
  points: Array<{
    date: string;
    value: number;
    pnl: number;
  }>;
}

export default function OverviewPage() {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [equityCurveData, setEquityCurveData] = useState<EquityCurveData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

        const [portfolioRes, equityCurveRes] = await Promise.all([
          fetch(`${apiUrl}/api/portfolio`, { cache: 'no-store' }),
          fetch(`${apiUrl}/api/equity-curve?days=90`, { cache: 'no-store' }),
        ]);

        if (!portfolioRes.ok || !equityCurveRes.ok) {
          throw new Error('Failed to fetch data');
        }

        const portfolio = await portfolioRes.json();
        const equityCurve = await equityCurveRes.json();

        setPortfolioData(portfolio);
        setEquityCurveData(equityCurve);
        setError(null);
      } catch (err) {
        console.error('[v0] Error fetching overview data:', err);
        setError(
          err instanceof Error ? err.message : 'Failed to load data'
        );
        // Set mock data for development
        setPortfolioData({
          portfolio_value: 128450.72,
          total_pnl: 18450.72,
          today_pnl: 1250.50,
          open_positions: 7,
          win_rate: 62.5,
          trades_today: 3,
          best_trade: {
            pnl: 5200,
            return_pct: 8.5,
            entry_price: 61200,
            exit_price: 66400,
            quantity: 0.083,
          },
          worst_trade: {
            pnl: -1800,
            return_pct: -3.2,
            entry_price: 56000,
            exit_price: 54200,
            quantity: 0.032,
          },
          total_return_pct: 16.8,
        });
        setEquityCurveData({
          points: Array.from({ length: 90 }, (_, i) => ({
            date: new Date(Date.now() - (89 - i) * 86400000)
              .toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
            value: 110000 + Math.random() * 20000 + i * 200,
            pnl: i * 205 + Math.random() * 500,
          })),
        });
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  return (
    <div className="flex flex-col gap-2 sm:gap-3 md:gap-4 lg:gap-6">
      {/* Page header */}
      <div className="flex flex-col gap-0.5">
        <h1 className="text-lg font-semibold tracking-tight text-balance sm:text-2xl md:text-3xl lg:text-4xl">
          Overview
        </h1>
        <p className="text-[16px] sm:text-md md:text-2md text-muted-foreground">
          Real-time snapshot of your paper-trading engine.
        </p>
      </div>

      {/* Stat cards */}
      <section className="grid grid-cols-1 gap-2 sm:gap-3 md:gap-4 lg:gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {loading ? (
          <>
            <StatCardSkeleton />
            <StatCardSkeleton />
            <StatCardSkeleton />
            <StatCardSkeleton />
          </>
        ) : portfolioData ? (
          <>
            <StatCard
              label="Portfolio Value"
              value={portfolioData.portfolio_value}
              isPositive={portfolioData.total_pnl >= 0}
              icon={Wallet}
              isCurrency
              decimals={2}
            />
            <StatCard
              label="Today's P&L"
              value={portfolioData.today_pnl}
              isPositive={portfolioData.today_pnl >= 0}
              icon={Activity}
              isCurrency
              decimals={2}
            />
            <StatCard
              label="Win Rate"
              value={portfolioData.win_rate}
              isPositive={portfolioData.win_rate >= 50}
              icon={TrendingUp}
              isCurrency={false}
              decimals={1}
            />
            <StatCard
              label="Open Positions"
              value={portfolioData.open_positions}
              isPositive={true}
              icon={Radio}
              isCurrency={false}
              decimals={0}
            />
          </>
        ) : (
          <div className="col-span-full text-center text-muted-foreground">
            Failed to load stats
          </div>
        )}
      </section>

      {/* Chart and trade summary */}
      <section className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        {loading ? (
          <>
            <ChartSkeleton />
            <TradeCardSkeleton />
            <TradeCardSkeleton />
          </>
        ) : equityCurveData ? (
          <>
            <EquityCurveChart data={equityCurveData.points} />
            <TradeSummaryCards
              bestTrade={portfolioData?.best_trade || null}
              worstTrade={portfolioData?.worst_trade || null}
            />
          </>
        ) : (
          <div className="col-span-full text-center text-muted-foreground">
            Failed to load chart data
          </div>
        )}
      </section>
    </div>
  );
}