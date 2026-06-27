'use client';

export default function PerformancePage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Performance</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Analyze performance metrics and trading statistics.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="card-glow p-6 rounded-lg bg-card border border-card-border">
          <h2 className="text-lg font-semibold text-foreground mb-4">Key Metrics</h2>
          <div className="space-y-4">
            <div className="flex justify-between items-center py-3 border-b border-card-border/50">
              <span className="text-muted-foreground">Win Rate</span>
              <span className="font-mono font-bold text-green-400">62.5%</span>
            </div>
            <div className="flex justify-between items-center py-3 border-b border-card-border/50">
              <span className="text-muted-foreground">Profit Factor</span>
              <span className="font-mono font-bold text-cyan-400">2.14</span>
            </div>
            <div className="flex justify-between items-center py-3 border-b border-card-border/50">
              <span className="text-muted-foreground">Total Trades</span>
              <span className="font-mono font-bold text-foreground">127</span>
            </div>
            <div className="flex justify-between items-center py-3">
              <span className="text-muted-foreground">Drawdown</span>
              <span className="font-mono font-bold text-red-400">-12.3%</span>
            </div>
          </div>
        </div>

        <div className="card-glow p-6 rounded-lg bg-card border border-card-border">
          <h2 className="text-lg font-semibold text-foreground mb-4">Monthly Returns</h2>
          <div className="space-y-4">
            {['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'].map((month, idx) => (
              <div key={month} className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground w-12">{month}</span>
                <div className="flex-1 mx-3 h-2 bg-secondary rounded-full overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-purple-500 to-cyan-400"
                    style={{ width: `${30 + idx * 8}%` }}
                  />
                </div>
                <span className="text-sm font-mono text-green-400 w-12 text-right">
                  +{(4 + idx * 0.5).toFixed(1)}%
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="card-glow p-6 rounded-lg bg-card border border-card-border">
        <h2 className="text-lg font-semibold text-foreground mb-4">Performance Chart</h2>
        <div className="h-64 flex items-center justify-center text-muted-foreground text-sm">
          Chart placeholder - integrate Recharts for live performance visualizations
        </div>
      </div>
    </div>
  );
}
