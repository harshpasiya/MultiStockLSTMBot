'use client';

import { TrendingUp, TrendingDown } from 'lucide-react';
import { cn } from '@/lib/utils';

interface MetricCardProps {
  label: string;
  value: string | number;
  subtext?: string;
  isPositive?: boolean;
  icon?: React.ComponentType<{ className?: string }> | React.ReactNode;
  trend?: 'up' | 'down';
  change?: string;
  className?: string;
}

export function PerformanceMetricCard({
  label,
  value,
  subtext,
  isPositive,
  icon,
  trend,
  change,
  className,
}: MetricCardProps) {
  const IconComponent = typeof icon === 'function' ? icon : null;
  return (
    <div
      className={cn(
        'card-hover rounded-lg border border-border bg-card p-3 sm:p-4 md:p-5 transition-all',
        className
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex-1 min-w-0">
          <p className="text-[10px] sm:text-xs font-medium uppercase tracking-wide text-muted-foreground truncate">
            {label}
          </p>
          <p className="mt-1 sm:mt-2 flex items-baseline gap-1 sm:gap-2 font-mono text-xl sm:text-2xl font-bold text-foreground truncate">
            {value}
            {isPositive !== undefined && (
              <span
                className={`text-xs sm:text-sm font-semibold flex-shrink-0 ${
                  isPositive ? 'text-positive' : 'text-negative'
                }`}
              >
                {isPositive ? (
                  <TrendingUp className="h-3 w-3 sm:h-4 sm:w-4" />
                ) : (
                  <TrendingDown className="h-3 w-3 sm:h-4 sm:w-4" />
                )}
              </span>
            )}
          </p>
          {subtext && (
            <p className="mt-0.5 sm:mt-1 text-[9px] sm:text-xs text-muted-foreground truncate">{subtext}</p>
          )}
        </div>
        {IconComponent && (
          <div className="flex h-8 w-8 sm:h-10 sm:w-10 flex-shrink-0 items-center justify-center rounded-lg bg-accent-subtle">
            <IconComponent className="h-4 w-4 sm:h-5 sm:w-5" />
          </div>
        )}
      </div>
    </div>
  );
}
