'use client';

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  defs,
  linearGradient,
  stop,
} from 'recharts';
import { formatCurrency } from '@/lib/utils';

interface EquityPoint {
  date: string;
  value: number;
  pnl: number;
}

interface EquityCurveChartProps {
  data: EquityPoint[];
}

export function EquityCurveChart({ data }: EquityCurveChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="card-hover lg:col-span-2 flex min-h-[380px] flex-col rounded-xl border border-border bg-card p-6">
        <h2 className="text-lg font-semibold tracking-tight">Equity Curve</h2>
        <div className="mt-6 flex flex-1 items-center justify-center rounded-lg border border-dashed border-border text-sm text-muted-foreground">
          No data available
        </div>
      </div>
    );
  }

  const minValue = Math.min(...data.map((d) => d.value));
  const maxValue = Math.max(...data.map((d) => d.value));
  const range = maxValue - minValue;
  const padding = range * 0.1;

  const CustomTooltip = ({
    active,
    payload,
  }: {
    active?: boolean;
    payload?: Array<{ value: number; payload: EquityPoint }>;
  }) => {
    if (!active || !payload || payload.length === 0) return null;

    const { date, value, pnl } = payload[0].payload;
    const isPositive = pnl >= 0;

    return (
      <div className="rounded-lg border border-border bg-card px-3 py-2 shadow-lg">
        <p className="text-xs text-muted-foreground">{date}</p>
        <p className="font-mono text-sm font-semibold">{formatCurrency(value)}</p>
        <p
          className={`text-xs font-medium ${
            isPositive ? 'text-positive' : 'text-negative'
          }`}
        >
          P&L: {isPositive ? '+' : ''}{formatCurrency(pnl)}
        </p>
      </div>
    );
  };

  return (
    <div className="card-hover lg:col-span-2 flex min-h-[380px] flex-col rounded-xl border border-border bg-card p-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold tracking-tight">Equity Curve</h2>
        <span className="text-xs text-muted-foreground">Last 90 days</span>
      </div>

      <div className="mt-6 flex-1">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#a0a0a0" stopOpacity={0.25} />
                <stop offset="100%" stopColor="#a0a0a0" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#3a3a3a" vertical={false} />
            <XAxis
              dataKey="date"
              stroke="#767676"
              style={{ fontSize: '12px' }}
              tick={{ fill: '#767676' }}
            />
            <YAxis
              domain={[minValue - padding, maxValue + padding]}
              stroke="#767676"
              style={{ fontSize: '12px' }}
              tickFormatter={(value) => formatCurrency(value)}
              tick={{ fill: '#767676' }}
            />
            <Tooltip content={<CustomTooltip />} />
            <Line
              type="monotone"
              dataKey="value"
              stroke="#e5e5e5"
              strokeWidth={2}
              dot={false}
              isAnimationActive={true}
              animationDuration={1000}
              fill="url(#equityGradient)"
              filterUnits="userSpaceOnUse"
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
