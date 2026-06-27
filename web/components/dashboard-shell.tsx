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
        <main className="flex-1 overflow-y-auto bg-dot-grid pb-20 md:pb-0">
          <div className="mx-auto w-full max-w-7xl px-4 py-6 md:px-6 md:py-8">
            {children}
          </div>
        </main>
      </div>

      <MobileNav />
    </div>
  );
}
