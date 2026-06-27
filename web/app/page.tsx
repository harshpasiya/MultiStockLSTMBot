import { ArrowUpRight, ArrowDownRight, Wallet, Radio, TrendingUp, Activity } from "lucide-react";

type Stat = {
  label: string;
  value: string;
  delta: string;
  positive: boolean;
  icon: typeof Wallet;
};

const stats: Stat[] = [
  { label: "Portfolio Value", value: "$128,450.72", delta: "+4.21%", positive: true, icon: Wallet },
  { label: "Open Positions", value: "7", delta: "+2 today", positive: true, icon: TrendingUp },
  { label: "Active Signals", value: "12", delta: "3 pending", positive: true, icon: Radio },
  { label: "Daily P&L", value: "-$842.10", delta: "-0.64%", positive: false, icon: Activity },
];

export default function OverviewPage() {
  return (
    <div className="flex flex-col gap-8">
      {/* Page header */}
      <div className="flex flex-col gap-1">
        <h1 className="text-2xl font-semibold tracking-tight text-balance md:text-3xl">
          Overview
        </h1>
        <p className="text-sm text-muted-foreground">
          Real-time snapshot of your Zodiac Godseye paper-trading engine.
        </p>
      </div>

      {/* Stat cards */}
      <section className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div
              key={stat.label}
              className="card-hover rounded-xl border border-border bg-card p-5"
            >
              <div className="flex items-center justify-between">
                <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  {stat.label}
                </span>
                <span className="flex h-8 w-8 items-center justify-center rounded-lg border border-border bg-background/40">
                  <Icon className="h-4 w-4 text-accent-cyan" />
                </span>
              </div>
              <p className="mt-4 font-mono text-2xl font-semibold tracking-tight">
                {stat.value}
              </p>
              <div
                className={`mt-2 inline-flex items-center gap-1 text-xs font-medium ${
                  stat.positive ? "text-positive" : "text-negative"
                }`}
              >
                {stat.positive ? (
                  <ArrowUpRight className="h-3.5 w-3.5" />
                ) : (
                  <ArrowDownRight className="h-3.5 w-3.5" />
                )}
                {stat.delta}
              </div>
            </div>
          );
        })}
      </section>

      {/* Placeholder panels */}
      <section className="grid grid-cols-1 gap-4 lg:grid-cols-3">
        <div className="card-hover lg:col-span-2 flex min-h-[320px] flex-col rounded-xl border border-border bg-card p-6">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold tracking-tight">
              Equity Curve
            </h2>
            <span className="text-xs text-muted-foreground">Last 30 days</span>
          </div>
          <div className="mt-6 flex flex-1 items-center justify-center rounded-lg border border-dashed border-border text-sm text-muted-foreground">
            Chart placeholder
          </div>
        </div>

        <div className="card-hover flex min-h-[320px] flex-col rounded-xl border border-border bg-card p-6">
          <h2 className="text-lg font-semibold tracking-tight">
            Recent Signals
          </h2>
          <div className="mt-6 flex flex-1 items-center justify-center rounded-lg border border-dashed border-border text-sm text-muted-foreground">
            Signal feed placeholder
          </div>
        </div>
      </section>
    </div>
  );
}
