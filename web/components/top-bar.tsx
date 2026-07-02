"use client";

import { Eye, MoreVertical } from "lucide-react";
import { useState } from "react";
import Link from "next/link";

export function TopBar() {
  const [moreOpen, setMoreOpen] = useState(false);

  return (
    <header className="sticky top-0 z-30 flex h-12 md:h-16 items-center justify-between gap-2 md:gap-3 border-b border-border bg-background/80 px-2 md:px-4 lg:px-6 backdrop-blur-md">
      {/* Left: mobile logo + paper mode badge */}
      <div className="flex items-center gap-1.5 md:gap-3 min-w-0">
        <div className="flex items-center gap-1 md:gap-2">
          <div className="flex h-6 w-6 md:h-8 md:w-8 items-center justify-center rounded-lg bg-accent-subtle flex-shrink-0">
            <Eye className="h-3 w-3 md:h-4 md:w-4 text-foreground" />
          </div>
          <span className="hidden text-xs md:text-lg font-semibold tracking-tight text-foreground md:inline truncate">
            Zodiac Godseye
          </span>
        </div>

        <span className="inline-flex items-center gap-1 rounded-full border border-muted bg-accent-subtle px-1.5 md:px-2.5 py-0.5 md:py-1 text-[8px] md:text-[11px] font-semibold uppercase tracking-wider text-muted-foreground flex-shrink-0">
          <span className="h-1 w-1 md:h-1.5 md:w-1.5 rounded-full bg-muted-foreground animate-status-pulse" />
          <span className="hidden sm:inline">Paper</span>
        </span>
      </div>

      {/* Right: system status + more menu */}
      <div className="flex items-center gap-1 md:gap-3 ml-auto">
        <div className="flex items-center gap-1 md:gap-2 rounded-full border border-border bg-card px-2 md:px-3 py-1">
          <span className="h-1.5 w-1.5 rounded-full bg-positive animate-status-pulse" />
          <span className="hidden text-[9px] md:text-xs font-medium text-muted-foreground sm:inline">
            Online
          </span>
        </div>

        <div className="relative md:hidden">
          <button
            onClick={() => setMoreOpen(!moreOpen)}
            className="p-1 rounded-lg hover:bg-accent-subtle text-muted-foreground hover:text-foreground transition-colors"
            aria-label="More options"
          >
            <MoreVertical className="h-4 w-4" />
          </button>
          
          {moreOpen && (
            <div className="absolute right-0 top-full mt-1 w-40 rounded-lg border border-border bg-card shadow-lg z-40">
              <Link href="/settings" className="block px-3 py-2 text-xs text-muted-foreground hover:text-foreground hover:bg-accent-subtle">Settings</Link>
              <Link href="/profile" className="block px-3 py-2 text-xs text-muted-foreground hover:text-foreground hover:bg-accent-subtle border-t border-border">Profile</Link>
              <Link href="/subscription" className="block px-3 py-2 text-xs text-muted-foreground hover:text-foreground hover:bg-accent-subtle border-t border-border">Subscription</Link>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}
