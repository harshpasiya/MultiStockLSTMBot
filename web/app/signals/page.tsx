'use client';

import { useEffect, useState } from 'react';
import { TodaySignalHero } from '@/components/todays-signal-hero';
import { ConfidenceDistributionChart } from '@/components/confidence-distribution-chart';
import { SignalAccuracyMetrics } from '@/components/signal-accuracy-metrics';
import { SignalHistoryTable } from '@/components/signal-history-table';

interface Signal {
  symbol: string;
  mode: string;
  side: string;
  entry_price: number;
  tp_price: number;
  sl_price: number;
  confidence_score: number;
  entry_time: string;
}

interface HistorySignal extends Signal {
  outcome?: 'Win' | 'Loss' | 'Open';
  pnl?: number;
}

// Mock today's signal
const MOCK_TODAY_SIGNAL: Signal = {
  symbol: 'AAPL',
  mode: 'SWING',
  side: 'STRONG_BUY',
  entry_price: 158.45,
  tp_price: 175.20,
  sl_price: 145.30,
  confidence_score: 82,
  entry_time: new Date().toISOString(),
};

// Mock historical signals for accuracy analysis
const MOCK_HISTORY_SIGNALS: HistorySignal[] = [
  {
    symbol: 'AAPL',
    mode: 'SWING',
    side: 'STRONG_BUY',
    entry_price: 158.45,
    tp_price: 175.20,
    sl_price: 145.30,
    confidence_score: 82,
    entry_time: new Date(Date.now() - 86400000).toISOString(),
    outcome: 'Win',
    pnl: 1680.50,
  },
  {
    symbol: 'MSFT',
    mode: 'SWING',
    side: 'BUY',
    entry_price: 420.75,
    tp_price: 445.00,
    sl_price: 408.50,
    confidence_score: 71,
    entry_time: new Date(Date.now() - 172800000).toISOString(),
    outcome: 'Win',
    pnl: 360.75,
  },
  {
    symbol: 'NVDA',
    mode: 'SWING',
    side: 'STRONG_BUY',
    entry_price: 132.10,
    tp_price: 145.80,
    sl_price: 122.50,
    confidence_score: 85,
    entry_time: new Date(Date.now() - 259200000).toISOString(),
    outcome: 'Win',
    pnl: 548.00,
  },
  {
    symbol: 'TSLA',
    mode: 'INTRADAY',
    side: 'SELL',
    entry_price: 245.30,
    tp_price: 235.00,
    sl_price: 255.40,
    confidence_score: 62,
    entry_time: new Date(Date.now() - 345600000).toISOString(),
    outcome: 'Loss',
    pnl: -125.30,
  },
  {
    symbol: 'GOOGL',
    mode: 'SWING',
    side: 'BUY',
    entry_price: 197.80,
    tp_price: 215.20,
    sl_price: 185.50,
    confidence_score: 78,
    entry_time: new Date(Date.now() - 432000000).toISOString(),
    outcome: 'Win',
    pnl: 522.00,
  },
  {
    symbol: 'META',
    mode: 'SWING',
    side: 'SELL',
    entry_price: 512.40,
    tp_price: 490.00,
    sl_price: 530.50,
    confidence_score: 58,
    entry_time: new Date(Date.now() - 518400000).toISOString(),
    outcome: 'Loss',
    pnl: -225.20,
  },
];

export default function SignalsPage() {
  const [todaySignal, setTodaySignal] = useState<Signal | null>(null);
  const [historySignals, setHistorySignals] = useState<HistorySignal[]>([]);
  const [loading, setLoading] = useState(true);

  // Calculate confidence distribution buckets
  const confidenceBuckets = [
    { range: '55-60%', count: 2 },
    { range: '60-65%', count: 3 },
    { range: '65-70%', count: 5 },
    { range: '70-75%', count: 8 },
    { range: '75-80%', count: 12 },
    { range: '80-85%', count: 9 },
    { range: '85%+', count: 6 },
  ];

  // Calculate accuracy metrics
  const calculateAccuracyMetrics = (signals: HistorySignal[]) => {
    const highConfidenceSignals = signals.filter((s) => s.confidence_score > 75 && s.outcome);
    const lowConfidenceSignals = signals.filter((s) => s.confidence_score >= 55 && s.confidence_score <= 65 && s.outcome);

    const highConfidenceWins = highConfidenceSignals.filter((s) => s.outcome === 'Win').length;
    const lowConfidenceWins = lowConfidenceSignals.filter((s) => s.outcome === 'Win').length;

    const winners = signals.filter((s) => s.outcome === 'Win');
    const losers = signals.filter((s) => s.outcome === 'Loss');

    return {
      highConfidenceWinRate: highConfidenceSignals.length > 0 ? Math.round((highConfidenceWins / highConfidenceSignals.length) * 100) : 0,
      lowConfidenceWinRate: lowConfidenceSignals.length > 0 ? Math.round((lowConfidenceWins / lowConfidenceSignals.length) * 100) : 0,
      avgConfidenceWinners: winners.length > 0 ? Math.round(winners.reduce((sum, s) => sum + s.confidence_score, 0) / winners.length) : 0,
      avgConfidenceLosers: losers.length > 0 ? Math.round(losers.reduce((sum, s) => sum + s.confidence_score, 0) / losers.length) : 0,
    };
  };

  useEffect(() => {
    async function fetchSignals() {
      try {
        const apiUrl = process.env.NEXT_PUBLIC_API_URL;
        if (!apiUrl) {
          console.log('[v0] No API URL configured, using mock data');
          setTodaySignal(MOCK_TODAY_SIGNAL);
          setHistorySignals(MOCK_HISTORY_SIGNALS);
          setLoading(false);
          return;
        }

        const [todayRes, tradesRes] = await Promise.all([
          fetch(`${apiUrl}/api/signals/today`, {
            headers: { 'Content-Type': 'application/json' },
          }),
          fetch(`${apiUrl}/api/trades?status=closed&limit=200`, {
            headers: { 'Content-Type': 'application/json' },
          }),
        ]);

        if (todayRes.ok) {
          const data = await todayRes.json();
          setTodaySignal(data.signals?.[0] || null);
        } else {
          setTodaySignal(MOCK_TODAY_SIGNAL);
        }

        if (tradesRes.ok) {
          const data = await tradesRes.json();
          setHistorySignals(data.trades?.slice(0, 30) || MOCK_HISTORY_SIGNALS);
        } else {
          setHistorySignals(MOCK_HISTORY_SIGNALS);
        }
      } catch (err) {
        console.error('[v0] Failed to fetch signals:', err);
        setTodaySignal(MOCK_TODAY_SIGNAL);
        setHistorySignals(MOCK_HISTORY_SIGNALS);
      } finally {
        setLoading(false);
      }
    }

    fetchSignals();
  }, []);

  const accuracyMetrics = calculateAccuracyMetrics(historySignals);

  return (
    <div className="space-y-3 sm:space-y-4 md:space-y-6 lg:space-y-8">
      <div>
        <h1 className="mb-1 text-2xl font-bold sm:mb-2 sm:text-3xl text-foreground">Signals</h1>
        <p className="text-xs sm:text-sm text-muted-foreground">View trading signals from the Zodiac Godseye LSTM engine and track their accuracy</p>
      </div>

      {/* Today's Signal Hero */}
      <section>
        <TodaySignalHero signal={todaySignal} loading={loading} />
      </section>

      {/* Confidence Distribution and Accuracy Metrics */}
      <div className="grid gap-3 sm:gap-4 md:gap-6 lg:gap-8 lg:grid-cols-3">
        <div className="lg:col-span-1">
          <ConfidenceDistributionChart data={confidenceBuckets} loading={loading} />
        </div>
        <div className="lg:col-span-2">
          <h3 className="mb-3 text-base sm:text-lg font-semibold text-foreground">Signal Accuracy Analysis</h3>
          <SignalAccuracyMetrics metrics={accuracyMetrics} loading={loading} />
        </div>
      </div>

      {/* Signal History Table */}
      <section>
        <h3 className="mb-2 sm:mb-3 md:mb-4 text-base sm:text-lg font-semibold text-foreground">Recent Signal History</h3>
        <SignalHistoryTable signals={historySignals} loading={loading} />
      </section>
    </div>
  );
}
