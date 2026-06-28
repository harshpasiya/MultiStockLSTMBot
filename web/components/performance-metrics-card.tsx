'use client';

import { TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  isPositive?: boolean;
  icon?: React.ReactNode;
  className?: string;
}

export function PerformanceMetricCard({
  label,
  value,
  subtext,
  isPositive,
  icon,
  className,
}: MetricCardProps) {
  return (
    <div
      className={cn(
        'card-hover rounded-lg border border-border bg-card p-5 transition-all',
        className
      )}
    >
      <div className="flex items-start justify-between">
        <div className="flex-1">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
            {label}
          </p>
          <p className="mt-2 flex items-baseline gap-2 font-mono text-2xl font-bold text-foreground">
            {value}
            {isPositive !== undefined && (
              <span
                className={`text-sm font-semibold ${
                  isPositive ? 'text-positive' : 'text-negative'
                }`}
              >
                {isPositive ? (
                  <TrendingUp className="h-4 w-4" />
                ) : (
                  <TrendingDown className="h-4 w-4" />
                )}
              </span>
            )}
          </p>
          {subtext && (
            <p className="mt-1 text-xs text-muted-foreground">{subtext}</p>
          )}
        </div>
        {icon && (
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-accent-subtle">
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}
