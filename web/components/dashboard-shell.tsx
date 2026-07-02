"use client";

import { useState } from "react";
import { Sidebar } from "@/components/sidebar";
import { TopBar } from "@/components/top-bar";
import { MobileNav } from "@/components/mobile-nav";

export function DashboardShell({ children }: { children: React.ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />

      <div className="flex flex-1 flex-col overflow-hidden">
        <TopBar />
        <main className="flex-1 overflow-y-auto bg-dot-grid pb-16 md:pb-0">
          <div className="mx-auto w-full max-w-7xl px-2 py-2 sm:px-3 sm:py-3 md:px-4 md:py-4 lg:px-6 lg:py-6">
            {children}
          </div>
        </main>
      </div>

      <MobileNav />
    </div>
  );
}
