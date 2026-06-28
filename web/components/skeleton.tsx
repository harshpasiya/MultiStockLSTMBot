'use client';

import { cn } from '@/lib/utils';

interface SkeletonProps {
  className?: string;
}

export function Skeleton({ className }: SkeletonProps) {
  return (
    <div
      className={cn(
        'animate-pulse rounded-lg bg-accent-subtle',
        className
      )}
    />
  );
}

export function StatCardSkeleton() {
  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="flex items-center justify-between">
        <Skeleton className="h-4 w-24" />
        <Skeleton className="h-8 w-8 rounded-lg" />
      </div>
      <Skeleton className="mt-4 h-8 w-32" />
      <Skeleton className="mt-2 h-4 w-20" />
    </div>
  );
}

export function ChartSkeleton() {
  return (
    <div className="card-hover lg:col-span-2 flex min-h-[380px] flex-col rounded-xl border border-border bg-card p-6">
      <Skeleton className="h-6 w-32" />
      <div className="mt-6 flex flex-1 flex-col gap-4">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-12 w-full" />
        ))}
      </div>
    </div>
  );
}

export function TradeCardSkeleton() {
  return (
    <div className="card-hover rounded-xl border border-border bg-card p-5">
      <Skeleton className="h-4 w-24" />
      <Skeleton className="mt-4 h-8 w-28" />
      <Skeleton className="mt-2 h-4 w-20" />
    </div>
  );
}
