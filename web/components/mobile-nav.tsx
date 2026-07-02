"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { navItems } from "@/lib/nav-items";

export function MobileNav() {
  const pathname = usePathname();

  // Filter out Settings, Profile, Subscription, and divider for mobile nav
  const mobileNavItems = navItems.filter(
    (item) =>
      !item.divider &&
      item.href !== "/settings" &&
      item.href !== "/profile" &&
      item.href !== "/subscription"
  );

  return (
    <nav className="fixed inset-x-0 bottom-0 z-20 flex items-stretch justify-around border-t border-border bg-card/90 backdrop-blur-md md:hidden pb-0.5">
      {mobileNavItems.map((item) => {
        const active =
          item.href === "/" ? pathname === "/" : pathname.startsWith(item.href);
        const Icon = item.icon;
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-label={item.label}
            className={`flex flex-1 flex-col items-center justify-center gap-0.5 py-1.5 text-[8px] font-medium transition-colors ${
              active
                ? "text-accent-cyan"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Icon className="h-4 w-4" />
            <span className="max-w-[48px] truncate leading-tight">{item.label}</span>
          </Link>
        );
      })}
    </nav>
  );
}
