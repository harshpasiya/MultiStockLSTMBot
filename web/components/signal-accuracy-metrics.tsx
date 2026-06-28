import { TrendingUp, TrendingDown } from 'lucide-react';

interface AccuracyMetrics {
  highConfidenceWinRate: number;
  lowConfidenceWinRate: number;
  avgConfidenceWinners: number;
  avgConfidenceLosers: number;
}

interface SignalAccuracyMetricsProps {
  metrics: AccuracyMetrics;
  loading: boolean;
}

export function SignalAccuracyMetrics({ metrics, loading }: SignalAccuracyMetricsProps) {
  if (loading) {
    return (
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {Array.from({ length: 4 }).map((_, i) => (
          <div key={i} className="card-hover rounded-lg border border-border bg-card p-6">
            <div className="h-24 animate-pulse bg-accent-subtle rounded-lg" />
          </div>
        ))}
      </div>
    );
  }

  const highConfidenceBetter = metrics.highConfidenceWinRate >= metrics.lowConfidenceWinRate;
  const winnersMoreConfident = metrics.avgConfidenceWinners >= metrics.avgConfidenceLosers;

  return (
    <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <div className="card-hover rounded-lg border border-border bg-card p-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs uppercase text-muted-foreground font-semibold mb-3">High Confidence Win Rate</p>
            <p className="text-3xl font-mono font-bold text-foreground">{metrics.highConfidenceWinRate}%</p>
            <p className="text-xs text-muted-foreground mt-2">Confidence &gt; 75%</p>
          </div>
          <TrendingUp className={`h-5 w-5 ${highConfidenceBetter ? 'text-positive' : 'text-negative'}`} />
        </div>
      </div>

      <div className="card-hover rounded-lg border border-border bg-card p-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs uppercase text-muted-foreground font-semibold mb-3">Low Confidence Win Rate</p>
            <p className="text-3xl font-mono font-bold text-foreground">{metrics.lowConfidenceWinRate}%</p>
            <p className="text-xs text-muted-foreground mt-2">Confidence 55-65%</p>
          </div>
          <TrendingDown className={`h-5 w-5 ${!highConfidenceBetter ? 'text-positive' : 'text-negative'}`} />
        </div>
      </div>

      <div className="card-hover rounded-lg border border-border bg-card p-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs uppercase text-muted-foreground font-semibold mb-3">Avg Conf. Winners</p>
            <p className="text-3xl font-mono font-bold text-positive">{metrics.avgConfidenceWinners}%</p>
            <p className="text-xs text-muted-foreground mt-2">Winning signals</p>
          </div>
          <TrendingUp className="h-5 w-5 text-positive" />
        </div>
      </div>

      <div className="card-hover rounded-lg border border-border bg-card p-6">
        <div className="flex items-start justify-between">
          <div>
            <p className="text-xs uppercase text-muted-foreground font-semibold mb-3">Avg Conf. Losers</p>
            <p className="text-3xl font-mono font-bold text-negative">{metrics.avgConfidenceLosers}%</p>
            <p className="text-xs text-muted-foreground mt-2">Losing signals</p>
          </div>
          <TrendingDown className="h-5 w-5 text-negative" />
        </div>
      </div>
    </div>
  );
}
