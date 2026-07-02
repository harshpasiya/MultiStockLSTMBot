'use client';

import { Check, X, Zap } from 'lucide-react';

interface Plan {
  id: string;
  name: string;
  price: string;
  color: string;
  features: {
    swing_signals: boolean;
    intraday_signals: boolean;
    daily_limit: number;
    monthly_limit: number;
    priority_support: boolean;
  };
}

interface SubscriptionData {
  current_plan: string;
  signals_today: number;
  signals_month: number;
  daily_limit: number;
  monthly_limit: number;
  intraday_unlocked: boolean;
}

const PLANS: Record<string, Plan> = {
  free: {
    id: 'free',
    name: 'Free',
    price: '₹0',
    color: 'bg-muted text-muted-foreground',
    features: {
      swing_signals: true,
      intraday_signals: false,
      daily_limit: 1,
      monthly_limit: 10,
      priority_support: false,
    },
  },
  basic: {
    id: 'basic',
    name: 'Basic',
    price: '₹999/mo',
    color: 'bg-blue-500/20 text-blue-400',
    features: {
      swing_signals: true,
      intraday_signals: false,
      daily_limit: 5,
      monthly_limit: 100,
      priority_support: false,
    },
  },
  pro: {
    id: 'pro',
    name: 'Pro',
    price: '₹2999/mo',
    color: 'bg-purple-500/20 text-purple-400',
    features: {
      swing_signals: true,
      intraday_signals: true,
      daily_limit: 20,
      monthly_limit: 500,
      priority_support: true,
    },
  },
  elite: {
    id: 'elite',
    name: 'Elite',
    price: '₹7999/mo',
    color: 'bg-gradient-to-r from-yellow-500/20 to-orange-500/20 text-yellow-400',
    features: {
      swing_signals: true,
      intraday_signals: true,
      daily_limit: 100,
      monthly_limit: 2000,
      priority_support: true,
    },
  },
};

const MOCK_SUBSCRIPTION: SubscriptionData = {
  current_plan: 'free',
  signals_today: 0,
  signals_month: 3,
  daily_limit: 1,
  monthly_limit: 10,
  intraday_unlocked: false,
};

function PlanBadge({ color }: { color: string }) {
  return (
    <span className={`inline-block rounded-full px-3 py-1 text-xs font-bold ${color}`}>
      {PLANS[MOCK_SUBSCRIPTION.current_plan].name}
    </span>
  );
}

function ProgressBar({
  used,
  limit,
  label,
}: {
  used: number;
  limit: number;
  label: string;
}) {
  const percentage = Math.min((used / limit) * 100, 100);
  const remaining = Math.max(limit - used, 0);

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <span className="text-sm font-medium text-foreground">{label}</span>
        <span className="text-xs text-muted-foreground">
          {used} / {limit} ({Math.round(percentage)}%)
        </span>
      </div>
      <div className="h-2 w-full rounded-full bg-accent-subtle">
        <div
          className={`h-full rounded-full transition-all ${
            percentage > 80 ? 'bg-negative' : percentage > 50 ? 'bg-yellow-500' : 'bg-positive'
          }`}
          style={{ width: `${percentage}%` }}
        />
      </div>
      <p className="text-xs text-muted-foreground">{remaining} signals remaining</p>
    </div>
  );
}

function FeatureRow({
  feature,
  included,
}: {
  feature: string;
  included: boolean;
}) {
  return (
    <div className="flex items-center gap-3 border-t border-border py-3">
      {included ? (
        <Check className="h-4 w-4 flex-shrink-0 text-positive" />
      ) : (
        <X className="h-4 w-4 flex-shrink-0 text-muted-foreground" />
      )}
      <span className={`text-sm ${included ? 'text-foreground' : 'text-muted-foreground'}`}>
        {feature}
      </span>
    </div>
  );
}

function PlanCard({
  planKey,
  isCurrent,
}: {
  planKey: string;
  isCurrent: boolean;
}) {
  const plan = PLANS[planKey];

  return (
    <div
      className={`card-hover relative rounded-lg border transition-all ${
        isCurrent
          ? 'border-purple-500/50 bg-card shadow-lg shadow-purple-500/10'
          : 'border-border bg-card'
      } p-6`}
    >
      {isCurrent && (
        <div className="absolute right-6 top-6">
          <span className="inline-flex items-center rounded-full bg-purple-500/20 px-2.5 py-1 text-xs font-bold text-purple-400">
            Current Plan
          </span>
        </div>
      )}

      <div className={`mb-2 inline-block rounded-full px-3 py-1 text-xs font-bold ${plan.color}`}>
        {plan.name}
      </div>

      <div className="mb-6 mt-4">
        <p className="text-3xl font-bold text-foreground">{plan.price}</p>
        {planKey !== 'free' && <p className="text-xs text-muted-foreground">per month</p>}
      </div>

      <div className="space-y-1 rounded-lg border border-border bg-accent-subtle p-3">
        <p className="text-xs font-semibold text-foreground">Signal Limits:</p>
        <p className="text-sm text-muted-foreground">
          {plan.features.daily_limit} signals/day
        </p>
        <p className="text-sm text-muted-foreground">
          {plan.features.monthly_limit} signals/month
        </p>
      </div>

      <div className="my-6 space-y-0">
        <FeatureRow feature="Swing Signals" included={plan.features.swing_signals} />
        <FeatureRow feature="Intraday Signals" included={plan.features.intraday_signals} />
        <FeatureRow
          feature={`Daily Limit: ${plan.features.daily_limit}`}
          included={true}
        />
        <FeatureRow
          feature={`Monthly Limit: ${plan.features.monthly_limit}`}
          included={true}
        />
        <FeatureRow feature="Priority Support" included={plan.features.priority_support} />
      </div>

      {!isCurrent && (
        <button className="w-full rounded-lg border border-border bg-accent-subtle py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent-hover">
          Upgrade
        </button>
      )}
    </div>
  );
}

export default function SubscriptionPage() {
  const subscription = MOCK_SUBSCRIPTION;
  const currentPlan = PLANS[subscription.current_plan];

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="mb-2 text-3xl font-bold text-foreground">Subscription</h1>
        <p className="text-sm text-muted-foreground">
          Manage your subscription plan and signal limits
        </p>
      </div>

      {/* Current Plan Card */}
      <div className="card-hover rounded-lg border border-border bg-card p-8">
        <div className="flex flex-col items-start gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <div className="mb-3 flex items-center gap-3">
              <h2 className="text-3xl font-bold text-foreground">{currentPlan.name}</h2>
              <span className={`inline-block rounded-full px-3 py-1 text-xs font-bold ${currentPlan.color}`}>
                Active
              </span>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-muted-foreground">
                <Zap className="mb-0.5 inline h-4 w-4" /> {subscription.daily_limit} signals/day • {subscription.monthly_limit} signals/month
              </p>
              {subscription.intraday_unlocked && (
                <p className="text-sm text-positive">
                  ✓ Intraday signals unlocked
                </p>
              )}
            </div>
          </div>
          {subscription.current_plan !== 'elite' && (
            <button className="rounded-lg border border-border bg-accent-subtle px-6 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent-hover">
              Upgrade Plan
            </button>
          )}
        </div>
      </div>

      {/* Usage This Month */}
      <div className="rounded-lg border border-border bg-card p-6">
        <h3 className="mb-4 text-lg font-semibold text-foreground">Usage This Month</h3>
        <div className="space-y-6">
          <ProgressBar
            used={subscription.signals_today}
            limit={subscription.daily_limit}
            label="Daily Signals"
          />
          <ProgressBar
            used={subscription.signals_month}
            limit={subscription.monthly_limit}
            label="Monthly Signals"
          />
        </div>
      </div>

      {/* Plans Comparison */}
      <div>
        <h3 className="mb-4 text-lg font-semibold text-foreground">Compare Plans</h3>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {Object.keys(PLANS).map((planKey) => (
            <PlanCard
              key={planKey}
              planKey={planKey}
              isCurrent={subscription.current_plan === planKey}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
