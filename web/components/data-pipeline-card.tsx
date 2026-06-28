import { format } from 'date-fns';

interface DataPipelineCardProps {
  label: string;
  lastUpdateDate: string;
  staleDays: number;
}

function getStatusColor(staleDays: number) {
  if (staleDays <= 1) return { bg: 'bg-positive/20', text: 'text-positive', label: 'Fresh' };
  if (staleDays <= 3) return { bg: 'bg-yellow-500/20', text: 'text-yellow-400', label: 'Stale' };
  return { bg: 'bg-negative/20', text: 'text-negative', label: 'Very Stale' };
}

export function DataPipelineCard({ label, lastUpdateDate, staleDays }: DataPipelineCardProps) {
  const status = getStatusColor(staleDays);
  const formattedDate = new Date(lastUpdateDate).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });

  return (
    <div className="rounded-lg border border-border bg-card p-5 transition-all hover:border-muted-foreground">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm font-medium text-foreground">{label}</p>
          <p className="mt-2 text-xs text-muted-foreground">Last updated</p>
          <p className="font-mono text-sm text-muted-foreground">{formattedDate}</p>
        </div>

        <div className="text-right">
          <div className={`relative inline-flex items-center justify-center rounded-full ${status.bg} px-3 py-1.5`}>
            <span className={`text-xs font-semibold ${status.text}`}>{status.label}</span>
            {staleDays <= 1 && (
              <>
                <span className={`absolute inline-flex h-3 w-3 rounded-full ${status.text} opacity-75 animate-pulse`} />
                <span className={`absolute inline-flex h-3 w-3 rounded-full ${status.text}`} />
              </>
            )}
          </div>
          <p className="mt-2 text-xs text-muted-foreground">{staleDays}d ago</p>
        </div>
      </div>
    </div>
  );
}
