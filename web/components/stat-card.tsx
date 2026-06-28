'use client';

import { useEffect, useState } from 'react';
import { ArrowUpRight, ArrowDownRight, LucideIcon } from 'lucide-react';

interface StatCardProps {
  label: string;
  value: number;
  isPositive: boolean;
  icon: LucideIcon;
  isCurrency?: boolean;
  decimals?: number;
}

export function StatCard({
  label,
  value,
  isPositive,
  icon: Icon,
  isCurrency = false,
  decimals = 2,
}: StatCardProps) {
  const [displayValue, setDisplayValue] = useState(0);

  useEffect(() => {
    const duration = 800;
    const steps = 60;
    const stepDuration = duration / steps;
    let currentStep = 0;

    const interval = setInterval(() => {
      currentStep++;
      const progress = Math.min(currentStep / steps, 1);
      setDisplayValue(value * progress);

      if (progress === 1) {
        clearInterval(interval);
      }
    }, stepDuration);

    return () => clearInterval(interval);
  }, [value]);

  const formattedValue = isCurrency
    ? `$${displayValue.toFixed(decimals).replace(/\B(?=(\d{3})+(?!\d))/g, ',')}`
    : Math.floor(displayValue).toString();

  return (
    <div className="card-hover rounded-xl border border-border bg-card p-5">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </span>
        <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-accent-subtle bg-accent-subtle">
          <Icon className="h-4 w-4 text-muted" />
        </span>
      </div>
      <p className="font-mono text-2xl font-semibold tracking-tight mt-4">
        {formattedValue}
      </p>
      <div
        className={`mt-2 inline-flex items-center gap-1 text-xs font-medium ${
          isPositive ? 'text-positive' : 'text-negative'
        }`}
      >
        {isPositive ? (
          <ArrowUpRight className="h-3.5 w-3.5" />
        ) : (
          <ArrowDownRight className="h-3.5 w-3.5" />
        )}
        {isPositive ? '+' : '-'}
        {Math.abs(value).toFixed(decimals)}%
      </div>
    </div>
  );
}
