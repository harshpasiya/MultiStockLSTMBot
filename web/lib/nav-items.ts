import {
  LayoutDashboard,
  Wallet,
  Radio,
  History,
  TrendingUp,
  Activity,
  Settings,
  UserCircle,
  Crown,
  type LucideIcon,
} from "lucide-react";

export type NavItem = {
  label: string;
  href: string;
  icon: LucideIcon;
  divider?: boolean;
};

export const navItems: NavItem[] = [
  { label: "Overview", href: "/", icon: LayoutDashboard },
  { label: "Positions", href: "/positions", icon: Wallet },
  { label: "Signals", href: "/signals", icon: Radio },
  { label: "Trade History", href: "/trade-history", icon: History },
  { label: "Performance", href: "/performance", icon: TrendingUp },
  { label: "System Status", href: "/system-status", icon: Activity },
  { divider: true, label: "", href: "" },
  { label: "Settings", href: "/settings", icon: Settings },
  { label: "Profile", href: "/profile", icon: UserCircle },
  { label: "Subscription", href: "/subscription", icon: Crown },
];
