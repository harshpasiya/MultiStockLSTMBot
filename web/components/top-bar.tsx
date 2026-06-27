"use client";

import { Eye } from "lucide-react";

export function TopBar() {
  const portfolioValue = 128_450.72;

  return (
    <header className="sticky top-0 z-20 flex h-16 items-center justify-between gap-3 border-b border-border bg-background/80 px-4 backdrop-blur-md md:px-6">
      {/* Left: mobile logo + paper mode badge */}
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-2 md:hidden">
          <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent-subtle">
            <Eye className="h-4 w-4 text-foreground" />
          </div>
          <span className="font-semibold tracking-tight">Zodiac Godseye</span>
        </div>

        <span className="hidden text-lg font-semibold tracking-tight text-foreground md:inline">
          Zodiac Godseye
        </span>

        <span className="inline-flex items-center gap-1.5 rounded-full border border-muted bg-accent-subtle px-2.5 py-1 text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          <span className="h-1.5 w-1.5 rounded-full bg-muted-foreground animate-status-pulse" />
          Paper Mode
        </span>
      </div>

      {/* Right: portfolio value + system status */}
      <div className="flex items-center gap-4 md:gap-6">
        <div className="flex flex-col items-end leading-tight">
          <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
            Portfolio
          </span>
          <span className="font-mono text-sm font-semibold text-foreground sm:text-base">
            $
            {portfolioValue.toLocaleString("en-US", {
              minimumFractionDigits: 2,
              maximumFractionDigits: 2,
            })}
          </span>
        </div>

        <div className="flex items-center gap-2 rounded-full border border-border bg-card px-3 py-1.5">
          <span className="h-2 w-2 rounded-full bg-positive animate-status-pulse" />
          <span className="hidden text-xs font-medium text-muted-foreground sm:inline">
            Online
          </span>
        </div>
      </div>
    </header>
  );
}
