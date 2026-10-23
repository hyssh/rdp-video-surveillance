import { useQueries, useQuery } from '@tanstack/react-query';
import { useEffect, useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api } from '../api';
import { ActionBars, Heatmap, OverlayChart, RiskTimeline } from '../components/charts';
import { Empty, ErrorMessage, Legend, RiskBadge, Skeleton } from '../components/common';
import type { Timeline } from '../types';
import { mmss, prettyAction } from '../util';

function SessionSelect({ id, label, onChange, items }: { id: string; label: string; onChange: (v: number) => void; items: { id: number; filename: string | null }[]; value?: number }) {
  return (
    <label style={{ display: 'grid', gap: 4 }}>
      <span className="sub">{label}</span>
      <select id={id} onChange={(e) => onChange(Number(e.target.value))} defaultValue="">
        <option value="" disabled>Choose a session…</option>
        {items.map((v) => <option key={v.id} value={v.id}>{v.filename ?? `Session #${v.id}`}</option>)}
      </select>
    </label>
  );
}

function SummaryCard({ t, color }: { t: Timeline; color: string }) {
  const scores = t.points.map((p) => p.risk_score ?? 0);
  const peak = Math.max(0, ...scores);
  const mix = new Map<string, number>();
  t.points.forEach((p) => mix.set(p.action_type ?? 'other', (mix.get(p.action_type ?? 'other') ?? 0) + 1));
  const top = [...mix.entries()].sort((a, b) => b[1] - a[1])[0];
  return (
    <div className="card" style={{ borderTop: `4px solid ${color}` }}>
      <header><h2>{t.filename}</h2><RiskBadge level={t.security_level} score={t.risk_score} /></header>
      <div className="stat-line"><span className="sub">Overall score</span><b>{t.risk_score ?? '–'}</b></div>
      <div className="stat-line"><span className="sub">Peak frame risk</span><b>{peak}</b></div>
      <div className="stat-line"><span className="sub">High-risk events</span><b>{t.events.length}</b></div>
      <div className="stat-line"><span className="sub">Duration</span><b>{mmss(t.duration_s)}</b></div>
      <div className="stat-line"><span className="sub">Top action</span><b style={{ textTransform: 'capitalize' }}>{top ? prettyAction(top[0]) : '–'}</b></div>
      <p className="sub" style={{ marginBottom: 0 }}><Link to={`/session/${t.video_id}`}>Open review →</Link></p>
    </div>
  );
}

export default function Insights() {
  const nav = useNavigate();
  const stats = useQuery({ queryKey: ['stats'], queryFn: api.stats });
  const vids = useQuery({ queryKey: ['videos'], queryFn: api.videos });
  const items = vids.data?.items ?? [];
  const [sel, setSel] = useState<number | null>(null);
  const [a, setA] = useState<number | null>(null);
  const [b, setB] = useState<number | null>(null);
  useEffect(() => { if (sel == null && items.length) setSel(items[0].id); }, [items, sel]);
  const selQ = useQuery({ queryKey: ['timeline', sel], queryFn: () => api.timeline(sel!), enabled: sel != null });
  const cmp = useQueries({ queries: [a, b].map((v) => ({ queryKey: ['timeline', v], queryFn: () => api.timeline(v!), enabled: v != null })) });
  const [ta, tb] = [cmp[0].data, cmp[1].data];

  return (
    <div className="page">
      <div className="page-head"><div><h1>Insights</h1><p className="sub">Where and how risk shows up across sessions.</p></div><Legend /></div>
      <div className="grid cols-2e" style={{ marginBottom: 16 }}>
        <section className="card" aria-labelledby="hm-h">
          <header>
            <h2 id="hm-h">Risk heatmap</h2>
            {items.length > 0 && (
              <label className="toolbar"><span className="sub">Session</span>
                <select value={sel ?? ''} onChange={(e) => setSel(Number(e.target.value))} aria-label="Select session for heatmap">
                  {items.map((v) => <option key={v.id} value={v.id}>{v.filename ?? `Session #${v.id}`}</option>)}
                </select>
              </label>
            )}
          </header>
          <p className="sub" style={{ marginTop: 0 }}>Cell color = maximum risk in that time bucket × action type. Click a cell to jump to the frame.</p>
          {vids.isError ? <ErrorMessage error={vids.error} /> : !vids.isLoading && !items.length ? <Empty title="No sessions yet" /> : selQ.isError ? <ErrorMessage error={selQ.error} /> : selQ.data ? <Heatmap data={selQ.data} onPick={(f) => nav(`/session/${sel}?frame=${f}`)} /> : <Skeleton h={220} />}
        </section>
        <section className="card" aria-labelledby="ab-h">
          <header><h2 id="ab-h">Action type breakdown</h2><span className="sub">frames, all sessions</span></header>
          {stats.isError ? <ErrorMessage error={stats.error} /> : stats.data ? (stats.data.action_types.length ? <ActionBars data={stats.data.action_types} /> : <Empty title="No actions classified yet" />) : <Skeleton h={220} />}
        </section>
      </div>
      {selQ.data && (
        <section className="card" style={{ marginBottom: 16 }}>
          <header><h2>Selected session timeline</h2></header>
          <RiskTimeline data={selQ.data} selectedId={null} onSelect={(f) => nav(`/session/${sel}?frame=${f}`)} height={200} />
        </section>
      )}
      <section className="card" aria-labelledby="cmp-h">
        <header><h2 id="cmp-h">Compare sessions</h2></header>
        <div className="toolbar" style={{ marginBottom: 14 }}>
          <SessionSelect id="cmp-a" label="Session A" items={items} onChange={setA} />
          <SessionSelect id="cmp-b" label="Session B" items={items} onChange={setB} />
        </div>
        {a == null || b == null ? <Empty icon="⚖️" title="Pick two sessions to compare">See their risk timelines overlaid, side by side.</Empty>
          : cmp.some((c) => c.isError) ? <ErrorMessage error={cmp.find((c) => c.isError)!.error} />
          : !ta || !tb ? <Skeleton h={260} /> : (
            <>
              <OverlayChart series={[{ name: ta.filename ?? 'A', color: 'var(--accent)', data: ta }, { name: tb.filename ?? 'B', color: 'var(--risk-high)', data: tb }]} />
              <div className="grid compare-grid" style={{ marginTop: 16 }}>
                <SummaryCard t={ta} color="var(--accent)" /><SummaryCard t={tb} color="var(--risk-high)" />
              </div>
            </>
          )}
      </section>
    </div>
  );
}
