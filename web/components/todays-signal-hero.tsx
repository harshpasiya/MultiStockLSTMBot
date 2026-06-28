import { Clock, TrendingUp, TrendingDown } from 'lucide-react';

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

interface TodaySignalHeroProps {
  signal: Signal | null;
  loading: boolean;
}

export function TodaySignalHero({ signal, loading }: TodaySignalHeroProps) {
  if (loading) {
    return (
      <div className="card-hover rounded-lg border border-border bg-card p-12">
        <div className="h-32 animate-pulse bg-accent-subtle rounded-lg" />
      </div>
    );
  }

  if (!signal) {
    return (
      <div className="card-hover rounded-lg border border-border bg-card p-12">
        <div className="flex flex-col items-center justify-center py-16 gap-4">
          <div className="relative">
            <Clock className="h-16 w-16 text-muted-foreground animate-pulse" />
            <div className="absolute inset-0 rounded-full border-2 border-muted-foreground/30 animate-pulse" />
          </div>
          <div className="text-center">
            <h3 className="text-xl font-semibold text-foreground mb-2">No signal generated yet</h3>
            <p className="text-sm text-muted-foreground">Next decision at 6:45 PM</p>
          </div>
        </div>
      </div>
    );
  }

  const isLong = signal.side === 'BUY' || signal.side === 'STRONG_BUY';
  const isShort = signal.side === 'SELL' || signal.side === 'STRONG_SELL';
  const isStrong = signal.side.includes('STRONG');

  const getConfidenceColor = (score: number) => {
    if (score >= 70) return 'text-green-400';
    if (score >= 55) return 'text-yellow-400';
    return 'text-negative';
  };

  const getConfidenceFill = (score: number) => {
    if (score >= 70) return '#4ade80';
    if (score >= 55) return '#facc15';
    return '#f87171';
  };

  return (
    <div className="card-hover rounded-lg border border-border bg-card p-8 md:p-12">
      <div className="flex flex-col items-center justify-center gap-8 md:flex-row md:justify-between">
        <div className="flex flex-col items-center gap-6">
          <div className="text-center md:text-left">
            <p className="text-sm font-semibold text-muted-foreground uppercase tracking-wide mb-3">Today's Signal</p>
            <h2 className="text-6xl md:text-7xl font-bold text-foreground font-mono">{signal.symbol}</h2>
          </div>

          <div className="flex gap-4">
            <span
              className={`px-6 py-2 rounded-lg font-bold uppercase tracking-wider text-sm md:text-base ${
                isLong
                  ? 'bg-positive/20 text-positive'
                  : isShort
                    ? 'bg-negative/20 text-negative'
                    : 'bg-accent-subtle text-muted-foreground'
              }`}
            >
              {signal.side}
            </span>
            {isStrong && (
              <span className="px-3 py-2 rounded-lg border border-border bg-accent-subtle text-xs md:text-sm text-muted-foreground flex items-center gap-2">
                Strong Signal
              </span>
            )}
          </div>
        </div>

        <div className="flex flex-col items-center gap-4">
          <div className="relative h-40 w-40 flex items-center justify-center">
            <svg viewBox="0 0 100 100" className="h-full w-full -rotate-90">
              <circle cx="50" cy="50" r="45" fill="none" stroke="currentColor" strokeWidth="2" className="text-accent-subtle" />
              <circle
                cx="50"
                cy="50"
                r="45"
                fill="none"
                stroke={getConfidenceFill(signal.confidence_score)}
                strokeWidth="2"
                strokeDasharray={`${(signal.confidence_score / 100) * 282.7} 282.7`}
                className="transition-all duration-1000"
              />
            </svg>
            <div className="absolute flex flex-col items-center">
              <span className={`text-4xl font-bold font-mono ${getConfidenceColor(signal.confidence_score)}`}>
                {signal.confidence_score}%
              </span>
              <span className="text-xs text-muted-foreground uppercase tracking-wide">Confidence</span>
            </div>
          </div>
        </div>
      </div>

      <div className="mt-10 grid grid-cols-3 gap-6 border-t border-border pt-8">
        <div>
          <p className="text-xs uppercase text-muted-foreground font-semibold mb-2">Entry Price</p>
          <p className="text-lg font-mono font-bold text-foreground">${signal.entry_price.toFixed(2)}</p>
        </div>
        <div>
          <p className="text-xs uppercase text-muted-foreground font-semibold mb-2">Take Profit</p>
          <p className="flex items-center gap-2">
            <span className="text-lg font-mono font-bold text-positive">${signal.tp_price.toFixed(2)}</span>
            <TrendingUp className="h-4 w-4 text-positive" />
          </p>
        </div>
        <div>
          <p className="text-xs uppercase text-muted-foreground font-semibold mb-2">Stop Loss</p>
          <p className="flex items-center gap-2">
            <span className="text-lg font-mono font-bold text-negative">${signal.sl_price.toFixed(2)}</span>
            <TrendingDown className="h-4 w-4 text-negative" />
          </p>
        </div>
      </div>
    </div>
  );
}
