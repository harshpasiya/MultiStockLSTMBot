'use client';

import { Activity, AlertCircle, CheckCircle2, Clock } from 'lucide-react';

export default function SystemStatusPage() {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-foreground">System Status</h1>
        <p className="text-sm text-muted-foreground mt-1">
          Monitor the health and connectivity of the Zodiac Godseye system.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[
          { name: 'LSTM Engine', status: 'online', uptime: '99.98%', color: 'text-green-400' },
          { name: 'Data Feed', status: 'online', uptime: '99.95%', color: 'text-green-400' },
          { name: 'Signal Generator', status: 'online', uptime: '99.99%', color: 'text-green-400' },
          { name: 'Execution Engine', status: 'online', uptime: '99.92%', color: 'text-green-400' },
        ].map((service) => (
          <div key={service.name} className="card-glow p-6 rounded-lg bg-card border border-card-border">
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm font-semibold text-foreground">{service.name}</p>
                <p className={`text-xs mt-1 ${service.color}`}>● {service.status.toUpperCase()}</p>
              </div>
              <div className="text-right">
                <p className="text-2xl font-mono font-bold text-cyan-400">{service.uptime}</p>
                <p className="text-xs text-muted-foreground">Uptime</p>
              </div>
            </div>
          </div>
        ))}
      </div>

      <div className="card-glow p-6 rounded-lg bg-card border border-card-border">
        <h2 className="text-lg font-semibold text-foreground mb-4">System Activity Log</h2>
        <div className="space-y-3">
          {[
            { time: '2:34 PM', event: 'New signal generated', type: 'info', icon: Activity },
            { time: '2:28 PM', event: 'Trade executed - BTC/USD', type: 'success', icon: CheckCircle2 },
            { time: '2:15 PM', event: 'Market data synchronized', type: 'success', icon: CheckCircle2 },
            { time: '1:42 PM', event: 'LSTM model retrained', type: 'warning', icon: Clock },
          ].map((log, idx) => {
            const Icon = log.icon;
            return (
              <div key={idx} className="flex items-start gap-3 py-3 border-b border-card-border/50 last:border-0">
                <Icon className={`w-4 h-4 mt-1 flex-shrink-0 ${
                  log.type === 'success' ? 'text-green-400' :
                  log.type === 'warning' ? 'text-yellow-400' :
                  'text-cyan-400'
                }`} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-foreground">{log.event}</p>
                  <p className="text-xs text-muted-foreground">{log.time}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
