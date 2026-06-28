'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

interface SymbolData {
  symbol: string;
  trades: number;
  wins: number;
  total_pnl: number;
}

interface SymbolChartProps {
  data: SymbolData[];
  loading?: boolean;
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload?.[0]) {
    const data = payload[0].payload;
    return (
      <div className="rounded-lg border border-border bg-card p-3 shadow-lg">
        <p className="font-mono text-sm font-bold text-foreground">{data.symbol}</p>
        <p className="text-xs text-muted-foreground">P&L: ${data.total_pnl.toFixed(2)}</p>
        <p className="text-xs text-muted-foreground">
          Trades: {data.trades} ({data.wins} wins)
        </p>
      </div>
    );
  }
  return null;
};

export function PerformanceSymbolChart({ data, loading = false }: SymbolChartProps) {
  if (loading) {
    return (
      <div className="h-96 w-full animate-pulse rounded-lg bg-accent-subtle" />
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex h-96 w-full items-center justify-center rounded-lg border border-border bg-card">
        <p className="text-sm text-muted-foreground">No performance data available</p>
      </div>
    );
  }

  // Sort by total_pnl descending
  const sortedData = [...data].sort((a, b) => b.total_pnl - a.total_pnl);

  return (
    <div className="rounded-lg border border-border bg-card p-6">
      <h2 className="mb-4 text-lg font-semibold text-foreground">P&L by Symbol</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={sortedData}
          layout="vertical"
          margin={{ top: 5, right: 30, left: 80, bottom: 5 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#3a3a3a" />
          <XAxis type="number" stroke="#a0a0a0" />
          <YAxis dataKey="symbol" type="category" stroke="#a0a0a0" width={70} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="total_pnl" fill="#4ade80" radius={[0, 8, 8, 0]}>
            {sortedData.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.total_pnl >= 0 ? '#4ade80' : '#f87171'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
