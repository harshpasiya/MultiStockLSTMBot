'use client';

export default function PositionsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Positions</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Monitor your open positions and exposure across all assets.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card-glow p-6 rounded-lg bg-card border border-card-border">
          <h2 className="text-lg font-semibold text-foreground mb-4">Active Positions</h2>
          <div className="space-y-3">
            {[
              { symbol: 'BTC/USD', size: '0.5', entry: '45,230', current: '46,120', pnl: '+$445' },
              { symbol: 'ETH/USD', size: '2.0', entry: '2,840', current: '2,920', pnl: '+$160' },
              { symbol: 'SOL/USD', size: '15.0', entry: '110', current: '112', pnl: '+$30' },
            ].map((pos) => (
              <div key={pos.symbol} className="p-3 rounded-lg bg-secondary/40 border border-secondary/20">
                <div className="flex justify-between items-start">
                  <div>
                    <p className="text-sm font-mono font-semibold text-foreground">{pos.symbol}</p>
                    <p className="text-xs text-muted-foreground">Size: {pos.size}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-xs text-green-400">{pos.pnl}</p>
                    <p className="text-xs text-muted-foreground">{pos.current}</p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="card-glow p-6 rounded-lg bg-card border border-card-border">
          <h2 className="text-lg font-semibold text-foreground mb-4">Position Summary</h2>
          <div className="space-y-4">
            <div>
              <p className="text-xs text-muted-foreground">Total Exposure</p>
              <p className="text-2xl font-mono font-bold text-foreground">$127,850</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Largest Position</p>
              <p className="text-2xl font-mono font-bold text-cyan-400">BTC/USD</p>
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Avg Win Rate</p>
              <p className="text-2xl font-mono font-bold text-green-400">62.5%</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
