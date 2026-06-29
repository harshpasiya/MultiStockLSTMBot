'use client';

import { useState } from 'react';
import { Copy, Edit2, Check, Lock, Zap, Shield, LogOut } from 'lucide-react';

interface ProfileData {
  name: string;
  email: string;
  role: string;
  referral_code: string;
  member_since: string;
  broker_provider: string;
  broker_connected: boolean;
  broker_account_id: string;
  paper_mode: boolean;
}

// Mock data
const MOCK_PROFILE: ProfileData = {
  name: 'Harsh Pasiya',
  email: 'harsh@zodiacgodseye.com',
  role: 'Super Admin',
  referral_code: 'ZG-HARSH-2024',
  member_since: '2024-01-15',
  broker_provider: 'Zerodha',
  broker_connected: true,
  broker_account_id: 'Z****5678',
  paper_mode: true,
};

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('[v0] Failed to copy:', err);
    }
  };

  return (
    <button
      onClick={handleCopy}
      className="inline-flex items-center gap-1.5 rounded-md border border-border bg-accent-subtle px-2.5 py-1.5 text-xs font-medium text-foreground transition-colors hover:bg-accent-hover"
    >
      {copied ? (
        <>
          <Check className="h-3.5 w-3.5" />
          Copied
        </>
      ) : (
        <>
          <Copy className="h-3.5 w-3.5" />
          Copy
        </>
      )}
    </button>
  );
}

function InitialAvatar({ name }: { name: string }) {
  const initials = name
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2);

  return (
    <div className="flex h-24 w-24 items-center justify-center rounded-full bg-gradient-to-br from-accent-subtle to-accent-hover">
      <span className="text-2xl font-bold text-foreground">{initials}</span>
    </div>
  );
}

export default function ProfilePage() {
  const [editMode, setEditMode] = useState(false);
  const [copied, setCopied] = useState(false);
  const profile = MOCK_PROFILE;

  const memberDate = new Date(profile.member_since).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'long',
    day: 'numeric',
  });

  return (
    <div className="space-y-6 p-6">
      <div>
        <h1 className="mb-2 text-3xl font-bold text-foreground">Profile</h1>
        <p className="text-sm text-muted-foreground">Manage your account information and trading preferences</p>
      </div>

      {/* Profile Header Card */}
      <div className="card-hover rounded-lg border border-border bg-card p-8">
        <div className="flex flex-col items-center gap-6 text-center sm:flex-row sm:text-left">
          <InitialAvatar name={profile.name} />
          <div className="flex-1">
            <div className="mb-2 flex items-center gap-2">
              <h2 className="text-2xl font-bold text-foreground">{profile.name}</h2>
              <button
                onClick={() => setEditMode(!editMode)}
                className="inline-flex items-center gap-1 rounded-md border border-border bg-accent-subtle px-2 py-1 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground"
              >
                <Edit2 className="h-3.5 w-3.5" />
              </button>
            </div>
            <p className="mb-1 text-sm text-muted-foreground">{profile.email}</p>
            <p className="text-xs text-muted-foreground">Member since {memberDate}</p>
          </div>
        </div>
      </div>

      {/* Account Details Section */}
      <div className="rounded-lg border border-border bg-card p-6">
        <h3 className="mb-4 text-lg font-semibold text-foreground">Account Details</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Role</span>
            <span className="inline-flex items-center rounded-full border border-border bg-accent-subtle px-3 py-1 text-xs font-semibold text-foreground">
              <Shield className="mr-1.5 h-3 w-3" />
              {profile.role}
            </span>
          </div>
          <div className="border-t border-border" />
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Referral Code</span>
            <div className="flex items-center gap-2">
              <code className="rounded bg-accent-subtle px-2 py-1 font-mono text-xs text-foreground">
                {profile.referral_code}
              </code>
              <CopyButton text={profile.referral_code} />
            </div>
          </div>
          <div className="border-t border-border" />
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Account Created</span>
            <span className="text-sm text-foreground">{memberDate}</span>
          </div>
        </div>
      </div>

      {/* Connected Broker Section */}
      <div className="rounded-lg border border-border bg-card p-6">
        <h3 className="mb-4 text-lg font-semibold text-foreground">Connected Broker</h3>
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Broker</span>
            <span className="text-sm font-medium text-foreground">{profile.broker_provider}</span>
          </div>
          <div className="border-t border-border" />
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Status</span>
            <span
              className={`inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs font-semibold ${
                profile.broker_connected
                  ? 'bg-positive/10 text-positive'
                  : 'bg-negative/10 text-negative'
              }`}
            >
              <span className={`h-2 w-2 rounded-full ${profile.broker_connected ? 'bg-positive' : 'bg-negative'}`} />
              {profile.broker_connected ? 'Connected' : 'Disconnected'}
            </span>
          </div>
          <div className="border-t border-border" />
          <div className="flex items-center justify-between">
            <span className="text-sm text-muted-foreground">Account ID</span>
            <span className="font-mono text-sm text-foreground">{profile.broker_account_id}</span>
          </div>
          {!profile.broker_connected && (
            <div className="pt-2">
              <button className="inline-flex items-center gap-2 rounded-lg border border-border bg-accent-subtle px-4 py-2 text-sm font-medium text-foreground transition-colors hover:bg-accent-hover">
                <Zap className="h-4 w-4" />
                Reconnect
              </button>
            </div>
          )}
        </div>
      </div>

      {/* Trading Mode Section */}
      <div className="rounded-lg border border-border bg-card p-6">
        <h3 className="mb-4 text-lg font-semibold text-foreground">Trading Mode</h3>
        <div className="flex flex-col gap-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <span className="text-sm font-medium text-foreground">Current Mode</span>
              <span
                className={`relative inline-flex h-8 w-16 items-center rounded-full px-1 ${
                  profile.paper_mode ? 'bg-accent-subtle' : 'bg-positive/20'
                }`}
              >
                <span
                  className={`inline-block h-6 w-7 rounded-md transition-all ${
                    profile.paper_mode
                      ? 'translate-x-0 bg-accent-hover'
                      : 'translate-x-8 bg-positive'
                  }`}
                />
              </span>
            </div>
            <span className={`text-sm font-bold ${profile.paper_mode ? 'text-muted-foreground' : 'text-positive'}`}>
              {profile.paper_mode ? 'PAPER MODE' : 'LIVE MODE'}
            </span>
          </div>
          {profile.paper_mode && (
            <div className="rounded-md border border-accent-subtle bg-accent-subtle/20 p-3">
              <p className="flex items-start gap-2 text-xs text-muted-foreground">
                <Lock className="mt-0.5 h-3.5 w-3.5 flex-shrink-0" />
                <span>
                  Paper Mode is locked for the first 6 weeks. This ensures you can test strategies safely before trading with real
                  capital. You&apos;ll be able to switch to Live Mode after 6 weeks of use.
                </span>
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Account Actions */}
      <div className="rounded-lg border border-negative/20 bg-negative/5 p-6">
        <h3 className="mb-4 flex items-center gap-2 text-lg font-semibold text-negative">
          <LogOut className="h-5 w-5" />
          Account Actions
        </h3>
        <button className="inline-flex items-center gap-2 rounded-lg border border-negative/30 bg-negative/10 px-4 py-2 text-sm font-medium text-negative transition-colors hover:bg-negative/20">
          <LogOut className="h-4 w-4" />
          Logout
        </button>
      </div>
    </div>
  );
}
