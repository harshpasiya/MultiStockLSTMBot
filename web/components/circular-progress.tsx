'use client';

interface CircularProgressProps {
  value: number; // 0-100
  size?: number; // diameter in pixels
  label?: string;
}

export function CircularProgress({ value, size = 48, label }: CircularProgressProps) {
  const radius = (size - 4) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (value / 100) * circumference;

  // Color based on confidence: red < 40, yellow 40-70, green > 70
  let color = 'rgb(168, 85, 247)'; // purple
  if (value < 40) color = 'rgb(248, 113, 113)'; // red
  else if (value < 70) color = 'rgb(251, 191, 36)'; // yellow
  else color = 'rgb(74, 222, 128)'; // green

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size} className="drop-shadow-sm">
        {/* Background circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--accent-subtle)"
          strokeWidth="2"
        />
        {/* Progress circle */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="2.5"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{
            transform: `rotate(-90deg)`,
            transformOrigin: `${size / 2}px ${size / 2}px`,
            transition: 'stroke-dashoffset 600ms ease',
          }}
        />
        {/* Center text */}
        <text
          x={size / 2}
          y={size / 2}
          textAnchor="middle"
          dy="0.3em"
          className="font-mono text-[10px] font-semibold fill-muted-foreground"
        >
          {value}%
        </text>
      </svg>
      {label && <span className="text-[11px] text-muted-foreground">{label}</span>}
    </div>
  );
}
