interface ModelInfoCardProps {
  checkpoint: string;
  valSharpe: number;
  valCagr: number;
  symbolCount: number;
}

export function ModelInfoCard({ checkpoint, valSharpe, valCagr, symbolCount }: ModelInfoCardProps) {
  return (
    <div className="rounded-lg border border-border bg-card p-6">
      <h2 className="mb-6 text-lg font-semibold text-foreground">Model Information</h2>

      <div className="space-y-6">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Checkpoint</p>
          <p className="mt-2 font-mono text-sm text-foreground">{checkpoint}</p>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Val Sharpe</p>
            <p className="mt-2 font-mono text-2xl font-bold text-positive">{valSharpe.toFixed(3)}</p>
          </div>

          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Val CAGR</p>
            <p className="mt-2 font-mono text-2xl font-bold text-positive">{(valCagr * 100).toFixed(1)}%</p>
          </div>

          <div>
            <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Symbols</p>
            <p className="mt-2 font-mono text-2xl font-bold text-foreground">{symbolCount}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
