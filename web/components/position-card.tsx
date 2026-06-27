'use client';

import { CircularProgress } from './circular-progress';
import { SLTPProgressBar } from './sl-tp-progress-bar';

interface PositionCardProps {
  symbol: string;
  entryPrice: number;
  quantity: number;
  tpPrice: number;
  slPrice: number;
  tpPct: number;
  slPct: number;
  holdDays: number;
  confidenceScore: number;
  mode: 'SWING' | 'INTRADAY';
  entryTime?: string;
}

export function PositionCard({
  symbol,
  entryPrice,
  quantity,
  tpPrice,
  slPrice,
  tpPct,
  slPct,
  holdDays,
  confidenceScore,
  mode,
}: PositionCardProps) {
  return (
    <div className="group card-hover relative overflow-hidden rounded-lg border border-border bg-card p-5 transition-all duration-200">
      {/* Top section: Symbol + Mode badge */}
      <div className="mb-4 flex items-start justify-between">
        <div>
          <h3 className="text-2xl font-bold text-foreground">{symbol}</h3>
          <p className="text-xs text-muted-foreground">
            {holdDays} day{holdDays !== 1 ? 's' : ''} held
          </p>
        </div>
        <span
          className={`inline-flex rounded-full px-2 py-1 text-xs font-semibold uppercase tracking-wider ${
            mode === 'SWING'
              ? 'border border-accent-subtle bg-accent-subtle text-foreground'
              : 'border border-muted bg-muted/20 text-muted-foreground'
          }`}
        >
          {mode}
        </span>
      </div>

      {/* Entry info row */}
      <div className="mb-4 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Entry</span>
          <span className="font-mono text-sm font-semibold text-foreground">
            ${entryPrice.toFixed(2)}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-xs text-muted-foreground">Qty</span>
          <span className="font-mono text-sm font-semibold text-foreground">
            {quantity} shares
          </span>
        </div>
      </div>

      {/* SL-TP Progress bar */}
      <div className="mb-5">
        <SLTPProgressBar entryPrice={entryPrice} slPrice={slPrice} tpPrice={tpPrice} />
      </div>

      {/* TP and SL percentage badges */}
      <div className="mb-4 flex gap-2">
        <div className="flex-1 rounded-lg border border-positive/30 bg-positive/10 py-2 px-3">
          <p className="text-[10px] text-muted-foreground">TP</p>
          <p className="font-mono text-sm font-semibold text-positive">+{tpPct.toFixed(2)}%</p>
        </div>
        <div className="flex-1 rounded-lg border border-negative/30 bg-negative/10 py-2 px-3">
          <p className="text-[10px] text-muted-foreground">SL</p>
          <p className="font-mono text-sm font-semibold text-negative">{slPct.toFixed(2)}%</p>
        </div>
      </div>

      {/* Confidence score ring at bottom */}
      <div className="flex items-center justify-center">
        <CircularProgress value={confidenceScore} size={52} label="Confidence" />
      </div>
    </div>
  );
}
