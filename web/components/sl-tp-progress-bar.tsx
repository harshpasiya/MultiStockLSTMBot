'use client';

interface SLTPProgressBarProps {
  entryPrice: number;
  currentPrice?: number; // if omitted, shows entry at midpoint
  slPrice: number;
  tpPrice: number;
}

export function SLTPProgressBar({
  entryPrice,
  currentPrice,
  slPrice,
  tpPrice,
}: SLTPProgressBarProps) {
  // Calculate position: entry is at 50%, SL is 0%, TP is 100%
  const priceToPercent = (price: number) => {
    const range = tpPrice - slPrice;
    if (range === 0) return 50;
    return ((price - slPrice) / range) * 100;
  };

  const entryPercent = priceToPercent(entryPrice);
  const currentPercent = currentPrice ? priceToPercent(currentPrice) : entryPercent;

  // Clamp to 0-100 for visual display
  const displayPercent = Math.max(0, Math.min(100, currentPercent));

  return (
    <div className="space-y-2">
      {/* Progress bar with gradient */}
      <div className="relative h-3 w-full overflow-hidden rounded-full bg-accent-subtle">
        {/* Gradient background: red → yellow → green */}
        <div
          className="absolute inset-0 bg-gradient-to-r"
          style={{
            background:
              'linear-gradient(90deg, rgb(248, 113, 113) 0%, rgb(251, 191, 36) 50%, rgb(74, 222, 128) 100%)',
          }}
        />
        {/* Overlay the filled progress */}
        <div
          className="h-full bg-gradient-to-r transition-all duration-300"
          style={{
            width: `${displayPercent}%`,
            background:
              displayPercent < 50
                ? `linear-gradient(90deg, rgb(248, 113, 113), rgb(251, 191, 36))`
                : `linear-gradient(90deg, rgb(251, 191, 36), rgb(74, 222, 128))`,
          }}
        />
        {/* Entry marker */}
        <div
          className="absolute top-1/2 h-4 w-1 -translate-y-1/2 -translate-x-1/2 rounded-full bg-white shadow-md"
          style={{
            left: `${entryPercent}%`,
          }}
        />
      </div>

      {/* Labels: SL and TP */}
      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
        <span>SL</span>
        <span>TP</span>
      </div>
    </div>
  );
}
