'use client';

import { PieChart, Pie, Cell, Legend, Tooltip, ResponsiveContainer } from 'recharts';

interface ExitReasonData {
  exit_reason: string;
  count: number;
  total_pnl: number;
}

interface ExitReasonsChartProps {
  data: ExitReasonData[];
  loading?: boolean;
}

// Color palette for exit reasons
const COLORS = [
  '#4ade80', // green
  '#f87171', // red
  '#facc15', // yellow
  '#60a5fa', // blue
  '#a78bfa', // purple
  '#34d399', // teal
];

const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload?.[0]) {
    const data = payload[0].payload;
    return (
      <div className="rounded-lg border border-border bg-card p-3 shadow-lg">
        <p className="font-mono text-sm font-bold text-foreground">{data.exit_reason}</p>
        <p className="text-xs text-muted-foreground">Count: {data.count}</p>
        <p className="text-xs text-muted-foreground">P&L: ${data.total_pnl.toFixed(2)}</p>
      </div>
    );
  }
  return null;
};

export function PerformanceExitReasonsChart({ data, loading = false }: ExitReasonsChartProps) {
  if (loading) {
    return (
      <div className="h-80 w-full animate-pulse rounded-lg bg-accent-subtle" />
    );
  }

  if (!data || data.length === 0) {
    return (
      <div className="flex h-80 w-full items-center justify-center rounded-lg border border-border bg-card">
        <p className="text-sm text-muted-foreground">No exit reason data available</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-6">
      <h2 className="mb-4 text-lg font-semibold text-foreground">Exit Reasons</h2>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={data}
            dataKey="count"
            nameKey="exit_reason"
            cx="50%"
            cy="50%"
            outerRadius={100}
            label={({ exit_reason }) => exit_reason}
            labelLine={false}
            animationBegin={0}
            animationDuration={800}
          >
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
            ))}
          </Pie>
          <Tooltip content={<CustomTooltip />} />
          <Legend verticalAlign="bottom" height={36} />
        </PieChart>
      </ResponsiveContainer>
    </div>
  );
}
