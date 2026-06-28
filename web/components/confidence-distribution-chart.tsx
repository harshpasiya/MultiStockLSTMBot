import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface ConfidenceData {
  range: string;
  count: number;
}

interface ConfidenceDistributionChartProps {
  data: ConfidenceData[];
  loading: boolean;
}

export function ConfidenceDistributionChart({ data, loading }: ConfidenceDistributionChartProps) {
  if (loading) {
    return (
      <div className="card-hover rounded-lg border border-border bg-card p-8">
        <div className="h-80 animate-pulse bg-accent-subtle rounded-lg" />
      </div>
    );
  }

  return (
    <div className="card-hover rounded-lg border border-border bg-card p-8">
      <div>
        <h3 className="text-lg font-semibold text-foreground mb-2">Confidence Distribution</h3>
        <p className="text-sm text-muted-foreground mb-6">Historical spread of model confidence scores across all signals</p>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data}>
          <defs>
            <linearGradient id="confidenceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#7c3aed" stopOpacity={0.8} />
              <stop offset="100%" stopColor="#7c3aed" stopOpacity={0.3} />
            </linearGradient>
          </defs>
          <CartesianGrid strokeDasharray="3 3" stroke="#3a3a3a" />
          <XAxis dataKey="range" stroke="#a0a0a0" style={{ fontSize: '12px' }} />
          <YAxis stroke="#a0a0a0" style={{ fontSize: '12px' }} />
          <Tooltip
            contentStyle={{
              backgroundColor: '#252525',
              border: '1px solid #3a3a3a',
              borderRadius: '6px',
              padding: '8px 12px',
            }}
            labelStyle={{ color: '#e5e5e5', fontSize: '12px' }}
            formatter={(value) => [value, 'Count']}
          />
          <Bar dataKey="count" fill="url(#confidenceGradient)" radius={[8, 8, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
