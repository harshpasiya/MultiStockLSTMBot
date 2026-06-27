'use client';

export default function SignalsPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">Signals</h1>
        <p className="text-sm text-muted-foreground mt-1">
          View all trading signals from the Zodiac Godseye LSTM engine.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {[
          { symbol: 'BTC/USD', signal: 'BUY', strength: 'Strong', confidence: '89%', timestamp: '2 min ago' },
          { symbol: 'ETH/USD', signal: 'BUY', strength: 'Medium', confidence: '72%', timestamp: '5 min ago' },
          { symbol: 'SOL/USD', signal: 'SELL', strength: 'Strong', confidence: '91%', timestamp: '8 min ago' },
          { symbol: 'XRP/USD', signal: 'HOLD', strength: 'Weak', confidence: '45%', timestamp: '12 min ago' },
        ].map((sig) => (
          <div key={sig.symbol} className="card-glow p-4 rounded-lg bg-card border border-card-border">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4 flex-1">
                <div>
                  <p className="text-sm font-mono font-semibold text-foreground">{sig.symbol}</p>
                  <p className="text-xs text-muted-foreground">{sig.timestamp}</p>
                </div>
              </div>
              <div className="flex items-center gap-4">
                <div className="text-right">
                  <p className={`text-sm font-bold px-3 py-1 rounded ${
                    sig.signal === 'BUY' ? 'bg-green-500/20 text-green-400' :
                    sig.signal === 'SELL' ? 'bg-red-500/20 text-red-400' :
                    'bg-gray-500/20 text-gray-400'
                  }`}>
                    {sig.signal}
                  </p>
                  <p className="text-xs text-muted-foreground mt-1">{sig.strength}</p>
                </div>
                <div className="text-right">
                  <p className="text-lg font-mono font-bold text-cyan-400">{sig.confidence}</p>
                  <p className="text-xs text-muted-foreground">Confidence</p>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
