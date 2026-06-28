export interface Trade {
  symbol: string;
  mode: string;
  side: string;
  entry_price: number;
  exit_price: number;
  quantity: number;
  realised_pnl: number;
  exit_reason: string;
  hold_days: number;
  confidence_score: number;
  entry_time: string;
  exit_time: string;
}

export function exportTradesToCSV(trades: Trade[], filename = 'trades.csv') {
  const headers = [
    'Symbol',
    'Mode',
    'Side',
    'Entry Price',
    'Exit Price',
    'Quantity',
    'Realized P&L',
    'Exit Reason',
    'Hold Days',
    'Confidence Score',
    'Entry Time',
    'Exit Time',
  ];

  const rows = trades.map((trade) => [
    trade.symbol,
    trade.mode,
    trade.side,
    trade.entry_price.toFixed(2),
    trade.exit_price.toFixed(2),
    trade.quantity.toString(),
    trade.realised_pnl.toFixed(2),
    trade.exit_reason,
    trade.hold_days.toString(),
    trade.confidence_score.toString(),
    new Date(trade.entry_time).toLocaleString(),
    new Date(trade.exit_time).toLocaleString(),
  ]);

  // Create CSV content
  const csvContent = [
    headers.map((h) => `"${h}"`).join(','),
    ...rows.map((row) => row.map((cell) => `"${cell}"`).join(',')),
  ].join('\n');

  // Create and download blob
  const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
  const link = document.createElement('a');
  const url = URL.createObjectURL(blob);

  link.setAttribute('href', url);
  link.setAttribute('download', `${filename}-${new Date().toISOString().split('T')[0]}.csv`);
  link.style.visibility = 'hidden';

  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
}
