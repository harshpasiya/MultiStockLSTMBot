'use client';

import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';

interface MonthlyData {
  month: string;
  trades: number;
  wins: number;
  pnl: number;
}

interface MonthlyChartProps {
  data: MonthlyData[];
  loading?: boolean;
}

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload?.[0]) {
    const data = payload[0].payload;
    return (
      <div className="rounded-lg border border-border bg-card p-3 shadow-lg">
        <p className="font-mono text-sm font-bold text-foreground">{data.month}</p>
        <p className="text-xs text-muted-foreground">P&L: ${data.pnl.toFixed(2)}</p>
        <p className="text-xs text-muted-foreground">
          Trades: {data.trades} ({data.wins} wins)
        </p>
      </div>
    );
  }
  return null;
};

export function PerformanceMonthlyChart({ data, loading = false }: MonthlyChartProps) {
  if (loading) {
    return (
      <div className="h-80 w-full animate-pulse rounded-lg bg-accent-subtle" />
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex h-80 w-full items-center justify-center rounded-lg border border-border bg-card">
        <p className="text-sm text-muted-foreground">No monthly data available</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-6">
      <h2 className="mb-4 text-lg font-semibold text-foreground">Monthly P&L</h2>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart
          data={data}
          margin={{ top: 20, right: 30, left: 0, bottom: 60 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#3a3a3a" />
          <XAxis 
            dataKey="month" 
            stroke="#a0a0a0" 
            angle={-45}
            textAnchor="end"
            height={80}
          />
          <YAxis stroke="#a0a0a0" />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="pnl" fill="#4ade80" radius={[8, 8, 0, 0]}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={entry.pnl >= 0 ? '#4ade80' : '#f87171'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
