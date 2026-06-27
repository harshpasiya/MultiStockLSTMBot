import {
  LayoutDashboard,
  Wallet,
  History,
  Radio,
  TrendingUp,
  Activity,
  type LucideIcon,
} from "lucide-react";

export type NavItem = {
  label: string;
  href: string;
  icon: LucideIcon;
};

export const navItems: NavItem[] = [
  { label: "Overview", href: "/", icon: LayoutDashboard },
  { label: "Positions", href: "/positions", icon: Wallet },
  { label: "Trade History", href: "/history", icon: History },
  { label: "Signals", href: "/signals", icon: Radio },
  { label: "Performance", href: "/performance", icon: TrendingUp },
  { label: "System Status", href: "/system", icon: Activity },
];
