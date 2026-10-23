import type { RiskLevel } from './types';

export const LEVELS: RiskLevel[] = ['low', 'medium', 'high', 'critical'];
export const LEVEL_LABEL: Record<RiskLevel, string> = { low: 'Low', medium: 'Medium', high: 'High', critical: 'Critical' };
export const LEVEL_COLOR: Record<RiskLevel, string> = {
  low: 'var(--risk-low)',
  medium: 'var(--risk-medium)',
  high: 'var(--risk-high)',
  critical: 'var(--risk-critical)',
};
// Score thresholds: low <30, medium 30-49, high 50-74, critical 75+
export const BANDS: { level: RiskLevel; from: number; to: number }[] = [
  { level: 'low', from: 0, to: 30 },
  { level: 'medium', from: 30, to: 50 },
  { level: 'high', from: 50, to: 75 },
  { level: 'critical', from: 75, to: 100 },
];

export function scoreToLevel(score: number | null | undefined): RiskLevel | null {
  if (score == null) return null;
  return score >= 75 ? 'critical' : score >= 50 ? 'high' : score >= 30 ? 'medium' : 'low';
}
export function toLevel(level: string | null | undefined, score?: number | null): RiskLevel | null {
  const l = (level ?? '').toLowerCase();
  if (l === 'low' || l === 'medium' || l === 'high' || l === 'critical') return l;
  return scoreToLevel(score);
}
export const levelColor = (level: string | null | undefined, score?: number | null) => {
  const l = toLevel(level, score);
  return l ? LEVEL_COLOR[l] : 'var(--muted)';
};

export function mmss(sec: number | null | undefined): string {
  if (sec == null || !isFinite(sec)) return '--:--';
  const s = Math.max(0, Math.round(sec));
  return `${String(Math.floor(s / 60)).padStart(2, '0')}:${String(s % 60).padStart(2, '0')}`;
}
export const prettyAction = (a: string | null | undefined) => (a ? a.replace(/_/g, ' ') : 'unknown');

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return '—';
  const d = new Date(iso);
  return isNaN(d.getTime()) ? '—' : d.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export const ACTION_COLORS: Record<string, string> = {
  login: '#60a5fa', browsing: '#a78bfa', terminal: '#2dd4bf', admin_console: '#f472b6',
  config_change: '#fbbf24', file_transfer: '#fb7185', idle: '#94a3b8', other: '#cbd5e1',
};
