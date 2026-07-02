import type { LucideIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface StatCardProps {
  label: string;
  value: number;
  isPositive: boolean;
  icon: LucideIcon;
  isCurrency?: boolean;
  decimals?: number;
}

function formatValue(value: number, isCurrency: boolean, decimals: number) {
  if (isCurrency) {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: decimals,
      minimumFractionDigits: decimals,
    }).format(value);
  }
  return value.toFixed(decimals);
}

export function StatCard({
  label,
  value,
  isPositive,
  icon: Icon,
  isCurrency = false,
  decimals = 2,
}: StatCardProps) {
  return (
    <div className="flex flex-col gap-1 rounded-md border border-border bg-card px-2.5 py-2 sm:px-3 sm:py-2.5">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-medium uppercase tracking-wide text-muted-foreground sm:text-[11px]">
          {label}
        </span>
        <Icon className="h-3 w-3 text-muted-foreground/70 sm:h-3.5 sm:w-3.5" strokeWidth={1.75} />
      </div>
      <span
        className={cn(
          'text-sm font-semibold tabular-nums leading-none sm:text-base',
          isPositive ? 'text-emerald-600' : 'text-red-600'
        )}
      >
        {formatValue(value, isCurrency, decimals)}
      </span>
    </div>
  );
}