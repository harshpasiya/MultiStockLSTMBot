'use client';

import { useState } from 'react';
import { AlertTriangle, Check, X, Loader2, Power } from 'lucide-react';

interface Toast {
  id: string;
  type: 'success' | 'error';
  title: string;
  message: string;
}

interface ConfirmModal {
  isOpen: boolean;
  task: string | null;
  taskLabel: string;
}

export default function SettingsPage() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [loadingTask, setLoadingTask] = useState<string | null>(null);
  const [confirmModal, setConfirmModal] = useState<ConfirmModal>({
    isOpen: false,
    task: null,
    taskLabel: '',
  });

  const [preferences, setPreferences] = useState({
    darkMode: true,
    autoRefresh: true,
    refreshInterval: 30,
    showPaperModeBadge: true,
  });

  const tasks = [
    { id: 'auth', label: 'Auth Check', description: 'Verify API authentication' },
    { id: 'morning', label: 'Morning Pipeline', description: 'Run morning data refresh' },
    { id: 'evening', label: 'Evening Pipeline', description: 'Run evening data refresh', needsConfirm: true },
    { id: 'trade', label: 'Force Trade Step', description: 'Trigger trading step', needsConfirm: true },
    { id: 'report', label: 'Generate Report', description: 'Generate daily report' },
    { id: 'status', label: 'Refresh Status', description: 'Check system status' },
  ];

  const addToast = (type: 'success' | 'error', title: string, message: string) => {
    const id = Math.random().toString(36).substr(2, 9);
    const newToast = { id, type, title, message };
    setToasts((prev) => [...prev, newToast]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 5000);
  };

  const handleTaskClick = (taskId: string, taskLabel: string, needsConfirm: boolean) => {
    if (needsConfirm) {
      setConfirmModal({ isOpen: true, task: taskId, taskLabel });
    } else {
      triggerTask(taskId);
    }
  };

  const triggerTask = async (taskId: string) => {
    setConfirmModal({ isOpen: false, task: null, taskLabel: '' });
    setLoadingTask(taskId);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL;
      if (!apiUrl) {
        // Mock response for development
        await new Promise((resolve) => setTimeout(resolve, 1500));
        addToast('success', `${taskId} Task Complete`, `Successfully executed ${taskId} task`);
        setLoadingTask(null);
        return;
      }

      const res = await fetch(`${apiUrl}/api/actions/${taskId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });

      const data = await res.json();

      if (res.ok) {
        addToast('success', 'Task Successful', data.stdout || `${taskId} task completed successfully`);
      } else {
        addToast('error', 'Task Failed', data.stderr || `Failed to execute ${taskId} task`);
      }
    } catch (err) {
      addToast('error', 'Error', `Failed to trigger ${taskId} task`);
    } finally {
      setLoadingTask(null);
    }
  };

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div>
        <h1 className="mb-2 text-3xl font-bold text-foreground">Settings</h1>
        <p className="text-sm text-muted-foreground">
          Control orchestrator tasks, configure display preferences, and manage system operations
        </p>
      </div>

      {/* Toast Notifications */}
      <div className="fixed bottom-6 right-6 space-y-2 z-50">
        {toasts.map((toast) => (
          <div
            key={toast.id}
            className={`flex items-start gap-3 rounded-lg border px-4 py-3 text-sm ${
              toast.type === 'success'
                ? 'border-positive/30 bg-positive/10 text-positive'
                : 'border-negative/30 bg-negative/10 text-negative'
            }`}
          >
            {toast.type === 'success' ? (
              <Check className="h-4 w-4 shrink-0 mt-0.5" />
            ) : (
              <X className="h-4 w-4 shrink-0 mt-0.5" />
            )}
            <div>
              <p className="font-semibold">{toast.title}</p>
              <p className="text-xs opacity-80">{toast.message}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Orchestrator Controls Section */}
      <div className="space-y-4">
        <div>
          <h2 className="text-xl font-bold text-foreground">Orchestrator Controls</h2>
          <p className="text-sm text-muted-foreground">Trigger pipeline and trading tasks remotely</p>
        </div>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {tasks.map((task) => (
            <button
              key={task.id}
              onClick={() => handleTaskClick(task.id, task.label, task.needsConfirm || false)}
              disabled={loadingTask !== null}
              className="group relative flex flex-col items-start gap-2 rounded-lg border border-border bg-card p-4 text-left transition-all hover:border-muted-foreground hover:bg-accent-hover disabled:opacity-60"
            >
              <div className="flex w-full items-center justify-between">
                <span className="font-semibold text-foreground">{task.label}</span>
                {loadingTask === task.id && <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" />}
              </div>
              <p className="text-xs text-muted-foreground">{task.description}</p>
              {task.needsConfirm && <span className="text-xs text-negative">Requires confirmation</span>}
            </button>
          ))}
        </div>
      </div>

      {/* Display Preferences Section */}
      <div className="space-y-4">
        <div>
          <h2 className="text-xl font-bold text-foreground">Display Preferences</h2>
          <p className="text-sm text-muted-foreground">Customize your dashboard experience</p>
        </div>

        <div className="space-y-3 rounded-lg border border-border bg-card p-4">
          {/* Dark Mode Toggle (Disabled) */}
          <div className="flex items-center justify-between py-3">
            <div>
              <p className="font-semibold text-foreground">Dark Mode</p>
              <p className="text-xs text-muted-foreground">Always enabled for this app</p>
            </div>
            <div className="relative inline-flex h-6 w-11 items-center rounded-full bg-positive/20">
              <span className="absolute left-1 inline-block h-4 w-4 transform rounded-full bg-positive" />
            </div>
          </div>

          {/* Auto-Refresh Toggle */}
          <div className="border-t border-border pt-3">
            <div className="flex items-center justify-between py-3">
              <div>
                <p className="font-semibold text-foreground">Auto-Refresh Dashboard</p>
                <p className="text-xs text-muted-foreground">Automatically refresh data at intervals</p>
              </div>
              <button
                onClick={() => setPreferences((p) => ({ ...p, autoRefresh: !p.autoRefresh }))}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  preferences.autoRefresh ? 'bg-positive/20' : 'bg-accent-subtle'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full transition-transform ${
                    preferences.autoRefresh ? 'translate-x-6 bg-positive' : 'translate-x-1 bg-muted-foreground'
                  }`}
                />
              </button>
            </div>

            {/* Refresh Interval Selector */}
            {preferences.autoRefresh && (
              <div className="mt-3 pl-4">
                <p className="mb-2 text-sm font-semibold text-foreground">Refresh Interval</p>
                <div className="flex gap-2">
                  {[15, 30, 60].map((interval) => (
                    <button
                      key={interval}
                      onClick={() => setPreferences((p) => ({ ...p, refreshInterval: interval }))}
                      className={`rounded px-3 py-1.5 text-sm transition-colors ${
                        preferences.refreshInterval === interval
                          ? 'bg-accent-subtle text-foreground'
                          : 'bg-background text-muted-foreground hover:bg-accent-hover'
                      }`}
                    >
                      {interval}s
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Show Paper Mode Badge Toggle */}
          <div className="border-t border-border pt-3">
            <div className="flex items-center justify-between py-3">
              <div>
                <p className="font-semibold text-foreground">Show Paper Mode Badge</p>
                <p className="text-xs text-muted-foreground">Display paper trading indicator in top bar</p>
              </div>
              <button
                onClick={() => setPreferences((p) => ({ ...p, showPaperModeBadge: !p.showPaperModeBadge }))}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  preferences.showPaperModeBadge ? 'bg-positive/20' : 'bg-accent-subtle'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full transition-transform ${
                    preferences.showPaperModeBadge ? 'translate-x-6 bg-positive' : 'translate-x-1 bg-muted-foreground'
                  }`}
                />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Danger Zone Section */}
      <div className="space-y-4">
        <div>
          <h2 className="text-xl font-bold text-foreground">Danger Zone</h2>
          <p className="text-sm text-muted-foreground">Advanced operations - use with caution</p>
        </div>

        <div className="rounded-lg border-2 border-negative/30 bg-negative/5 p-4">
          <div className="flex items-start gap-3">
            <AlertTriangle className="h-5 w-5 text-negative shrink-0 mt-0.5" />
            <div className="flex-1">
              <h3 className="font-semibold text-foreground mb-2">Emergency Stop</h3>
              <p className="text-sm text-muted-foreground mb-4">
                Immediately halt all trading operations and close all positions. This action is irreversible during the current session.
              </p>
              <button
                className="group relative flex items-center gap-2 rounded px-4 py-2 font-semibold text-negative transition-all"
                style={{
                  background: 'rgba(239, 68, 68, 0.1)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  boxShadow: '0 0 0 0 rgba(239, 68, 68, 0.4)',
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLElement).style.boxShadow =
                    '0 0 12px 4px rgba(239, 68, 68, 0.3), 0 0 24px 8px rgba(239, 68, 68, 0.15)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLElement).style.boxShadow =
                    '0 0 0 0 rgba(239, 68, 68, 0.4)';
                }}
                disabled
              >
                <Power className="h-4 w-4" />
                Emergency Stop (Disabled)
              </button>
              <p className="text-xs text-negative/70 mt-2">Coming soon - not yet implemented</p>
            </div>
          </div>
        </div>
      </div>

      {/* Confirmation Modal */}
      {confirmModal.isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="rounded-lg border border-border bg-card p-6 max-w-sm">
            <h3 className="text-lg font-bold text-foreground mb-2">Confirm Operation</h3>
            <p className="text-sm text-muted-foreground mb-4">
              This will run a real operation on the live trading system. Continue?
            </p>
            <p className="text-sm font-semibold text-foreground mb-4">
              Task: <span className="text-accent-subtle">{confirmModal.taskLabel}</span>
            </p>
            <div className="flex gap-3">
              <button
                onClick={() => setConfirmModal({ isOpen: false, task: null, taskLabel: '' })}
                className="flex-1 rounded px-4 py-2 text-sm font-semibold text-muted-foreground border border-border hover:bg-accent-hover transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={() => confirmModal.task && triggerTask(confirmModal.task)}
                className="flex-1 rounded px-4 py-2 text-sm font-semibold bg-accent-subtle text-foreground hover:bg-accent-hover transition-colors"
              >
                Continue
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
