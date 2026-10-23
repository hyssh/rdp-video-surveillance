import { Component, type ReactNode, useEffect, useRef, useState } from 'react';
import { friendlyError } from '../api';
import { LEVELS, LEVEL_COLOR, LEVEL_LABEL, toLevel } from '../util';

export function RiskBadge({ level, score }: { level?: string | null; score?: number | null }) {
  const l = toLevel(level, score);
  if (!l) return <span className="badge none">Unscored</span>;
  return (
    <span className="badge" style={{ ['--c' as string]: LEVEL_COLOR[l] }} aria-label={`Risk level ${l}`}>
      {l}
    </span>
  );
}

export function ScoreBar({ score, level }: { score: number | null | undefined; level?: string | null }) {
  if (score == null) return <span className="sub">—</span>;
  const l = toLevel(level, score);
  return (
    <div className="scorebar" style={{ ['--c' as string]: l ? LEVEL_COLOR[l] : 'var(--muted)' }}>
      <div className="track" role="progressbar" aria-valuenow={score} aria-valuemin={0} aria-valuemax={100} aria-label="Risk score">
        <div className="fill" style={{ width: `${score}%` }} />
      </div>
      <b>{score}</b>
    </div>
  );
}

export const Skeleton = ({ h = 16, w = '100%' }: { h?: number; w?: number | string }) => (
  <div className="skeleton" style={{ height: h, width: w }} aria-hidden="true" />
);

export function Legend() {
  return (
    <div className="legend" aria-label="Risk color legend">
      {LEVELS.map((l) => (
        <span key={l} style={{ ['--c' as string]: LEVEL_COLOR[l] }}>
          <i /> {LEVEL_LABEL[l]}
        </span>
      ))}
    </div>
  );
}

export function ErrorMessage({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const f = friendlyError(error);
  return (
    <div className="alert" role="alert">
      <b>{f.title}</b>
      {f.detail && <div>{f.detail}</div>}
      {f.stage && <small>Stage: {f.stage}</small>}
      {onRetry && (
        <div style={{ marginTop: 8 }}>
          <button className="btn small" onClick={onRetry}>Retry</button>
        </div>
      )}
    </div>
  );
}

export function Empty({ icon = '🗂️', title, children }: { icon?: string; title: string; children?: ReactNode }) {
  return (
    <div className="empty">
      <div className="big" aria-hidden="true">{icon}</div>
      <b>{title}</b>
      {children && <div>{children}</div>}
    </div>
  );
}

export class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  render() {
    if (this.state.error) {
      return (
        <div className="page">
          <ErrorMessage error={this.state.error} onRetry={() => { this.setState({ error: null }); }} />
        </div>
      );
    }
    return this.props.children;
  }
}

export function useWidth<T extends HTMLElement>(): [React.RefObject<T>, number] {
  const ref = useRef<T>(null);
  const [w, setW] = useState(600);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setW(Math.max(260, Math.floor(e.contentRect.width))));
    ro.observe(el);
    setW(Math.max(260, el.clientWidth));
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}
