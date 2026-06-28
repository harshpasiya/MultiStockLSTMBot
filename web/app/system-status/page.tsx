'use client';

import { useEffect, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { DataPipelineCard } from '@/components/data-pipeline-card';
import { ModelInfoCard } from '@/components/model-info-card';

interface SystemData {
  ohlcv_date: string;
  features_date: string;
  embeddings_date: string;
  ohlcv_stale_days: number;
  features_stale_days: number;
  embeddings_stale_days: number;
  symbol_count: number;
  model_checkpoint: string;
  val_sharpe: number;
  val_cagr: number;
}

// Mock data for development/testing
const MOCK_SYSTEM_DATA: SystemData = {
  ohlcv_date: new Date(Date.now() - 12 * 60 * 60 * 1000).toISOString(), // 12 hours ago
  features_date: new Date(Date.now() - 18 * 60 * 60 * 1000).toISOString(), // 18 hours ago
  embeddings_date: new Date(Date.now() - 3 * 24 * 60 * 60 * 1000).toISOString(), // 3 days ago
  ohlcv_stale_days: 0,
  features_stale_days: 1,
  embeddings_stale_days: 3,
  symbol_count: 150,
  model_checkpoint: 'v2.4.1-lstm-20240627',
  val_sharpe: 1.847,
  val_cagr: 0.285,
};

export default function SystemStatusPage() {
  const [data, setData] = useState<SystemData | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchSystemData = async () => {
    try {
      setRefreshing(true);
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      
      if (!apiUrl) {
        console.log('[v0] No API URL configured, using mock data');
        setData(MOCK_SYSTEM_DATA);
        setLoading(false);
        setRefreshing(false);
        return;
      }

      const res = await fetch(`${apiUrl}/api/system`, {
        headers: { 'Content-Type': 'application/json' },
      });

      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }

      const systemData: SystemData = await res.json();
      setData(systemData);
    } catch (err) {
      console.error('[v0] Failed to fetch system data:', err);
      setData(MOCK_SYSTEM_DATA);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchSystemData();
  }, []);

  if (loading || !data) {
    return (
      <div className="space-y-6 p-6">
        <div>
          <h1 className="mb-2 text-3xl font-bold text-foreground">System Status</h1>
          <p className="text-sm text-muted-foreground">Monitor the health and data freshness of Zodiac Godseye</p>
        </div>
        <div className="text-center text-muted-foreground">Loading system status...</div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="mb-2 text-3xl font-bold text-foreground">System Status</h1>
          <p className="text-sm text-muted-foreground">Monitor the health and data freshness of Zodiac Godseye</p>
        </div>

        <button
          onClick={fetchSystemData}
          disabled={refreshing}
          className="inline-flex items-center gap-2 rounded-lg border border-border bg-card px-4 py-2 text-sm font-medium text-foreground transition-colors hover:border-muted-foreground hover:bg-accent-hover disabled:opacity-50"
        >
          <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
          {refreshing ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      {/* Data Pipeline Section */}
      <div className="space-y-4">
        <h2 className="text-lg font-semibold text-foreground">Data Pipeline</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          <DataPipelineCard
            label="OHLCV Data"
            lastUpdateDate={data.ohlcv_date}
            staleDays={data.ohlcv_stale_days}
          />
          <DataPipelineCard
            label="Features"
            lastUpdateDate={data.features_date}
            staleDays={data.features_stale_days}
          />
          <DataPipelineCard
            label="Embeddings"
            lastUpdateDate={data.embeddings_date}
            staleDays={data.embeddings_stale_days}
          />
        </div>
      </div>

      {/* Model Info Section */}
      <div className="lg:col-span-2">
        <ModelInfoCard
          checkpoint={data.model_checkpoint}
          valSharpe={data.val_sharpe}
          valCagr={data.val_cagr}
          symbolCount={data.symbol_count}
        />
      </div>
    </div>
  );
}
