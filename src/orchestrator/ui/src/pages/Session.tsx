import { useQuery } from '@tanstack/react-query';
import { useCallback, useEffect, useMemo, useState } from 'react';
import { Link, useParams, useSearchParams } from 'react-router-dom';
import { api } from '../api';
import { RiskTimeline } from '../components/charts';
import { Empty, ErrorMessage, Legend, RiskBadge, Skeleton } from '../components/common';
import { Filmstrip } from '../components/filmstrip';
import { AskPanel, DetailsPanel, EventList, EvidencePanel, Rich } from '../components/panels';
import { FrameViewer } from '../components/viewer';
import { levelColor, mmss, toLevel, LEVEL_COLOR } from '../util';

type Tab = 'details' | 'events' | 'evidence' | 'ask';
const TABS: { id: Tab; label: string }[] = [
  { id: 'details', label: 'Details' },
  { id: 'events', label: 'Events' },
  { id: 'evidence', label: 'Evidence' },
  { id: 'ask', label: 'Ask' },
];

function Gauge({ score, level }: { score: number | null; level: string | null }) {
  const r = 34, c = 2 * Math.PI * r, v = Math.max(0, Math.min(100, score ?? 0));
  const col = levelColor(level, score);
  return (
    <svg width={84} height={84} viewBox="-42 -42 84 84" role="img" aria-label={`Overall risk score ${score ?? 'unknown'} out of 100`}>
      <circle r={r} fill="none" stroke="var(--panel-2)" strokeWidth={9} />
      <circle r={r} fill="none" stroke={col} strokeWidth={9} strokeLinecap="round" strokeDasharray={`${(v / 100) * c} ${c}`} transform="rotate(-90)" style={{ transition: 'stroke-dasharray .6s ease' }} />
      <text textAnchor="middle" y={8} fontSize={22} fontWeight={700} fill="var(--text)">{score ?? '–'}</text>
    </svg>
  );
}

export default function Session() {
  const { id } = useParams();
  const videoId = Number(id);
  const [search, setSearch] = useSearchParams();
  const tl = useQuery({ queryKey: ['timeline', videoId], queryFn: () => api.timeline(videoId), enabled: Number.isFinite(videoId) });
  const [tab, setTab] = useState<Tab>('details');
  const [overlay, setOverlay] = useState(false);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [typeFilter, setTypeFilter] = useState<string | null>(null);
  const [showSummary, setShowSummary] = useState(true);
  const [selected, setSelected] = useState<number | null>(null);

  const data = tl.data;
  const points = data?.points ?? [];
  useEffect(() => {
    if (!data || !points.length) return;
    const wanted = Number(search.get('frame'));
    if (wanted && points.some((p) => p.frame_id === wanted)) setSelected(wanted);
    else if (selected == null || !points.some((p) => p.frame_id === selected)) {
      const first = data.events[0]?.frame_id ?? points[0].frame_id;
      setSelected(first);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [data]);

  const select = useCallback((fid: number) => { setSelected(fid); setHoverIdx(null); }, []);
  const idx = points.findIndex((p) => p.frame_id === selected);
  const point = idx >= 0 ? points[idx] : undefined;

  useEffect(() => {
    if (selected != null) setSearch((s) => { const n = new URLSearchParams(s); n.set('frame', String(selected)); return n; }, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest('input, textarea, select') || e.altKey || e.ctrlKey || e.metaKey) return;
      if (e.key === 'ArrowRight' && idx < points.length - 1) { e.preventDefault(); select(points[idx + 1].frame_id); }
      if (e.key === 'ArrowLeft' && idx > 0) { e.preventDefault(); select(points[idx - 1].frame_id); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [idx, points, select]);

  const frameQ = useQuery({ queryKey: ['frame', selected], queryFn: () => api.frame(selected!), enabled: selected != null });
  const elQ = useQuery({ queryKey: ['elements', selected], queryFn: () => api.elements(selected!), enabled: selected != null, retry: 1 });
  const elements = elQ.data?.elements ?? [];

  const typeVisible = useMemo(() => (typeFilter ? (e: { type: string }) => e.type === typeFilter : undefined), [typeFilter]);

  if (tl.isError) return <div className="page"><Link to="/">← Dashboard</Link><div style={{ height: 12 }} /><ErrorMessage error={tl.error} onRetry={() => tl.refetch()} /></div>;
  if (tl.isLoading || !data) {
    return <div className="page"><Skeleton h={40} w={320} /><div style={{ height: 16 }} /><div className="review"><div className="stack"><Skeleton h={420} /><Skeleton h={230} /></div><Skeleton h={420} /></div></div>;
  }
  if (!points.length) return <div className="page"><Link to="/">← Dashboard</Link><Empty icon="🎞️" title="No frames analyzed yet">This session has no analyzed frames.</Empty></div>;

  const level = toLevel(data.security_level, data.risk_score);

  return (
    <div className="page">
      <Link to="/" className="sub">← Dashboard</Link>
      <div className="session-head" style={{ marginTop: 8 }}>
        <div style={{ flex: 1, minWidth: 280 }}>
          <div className="gauge-row">
            <Gauge score={data.risk_score} level={data.security_level} />
            <div>
              <h1>{data.filename ?? `Session #${videoId}`}</h1>
              <p className="sub" style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                <RiskBadge level={data.security_level} score={data.risk_score} />
                <span>{mmss(data.duration_s)} duration</span><span>·</span><span>{points.length} frames</span><span>·</span>
                <span style={{ color: level ? LEVEL_COLOR[level] : undefined }}>{data.events.length} high-risk events</span>
              </p>
            </div>
          </div>
        </div>
        <Legend />
      </div>

      <section className="card" style={{ marginBottom: 16 }} aria-labelledby="sum-h">
        <header>
          <h2 id="sum-h">AI summary</h2>
          <button className="btn small" aria-expanded={showSummary} onClick={() => setShowSummary((v) => !v)}>{showSummary ? 'Collapse' : 'Expand'}</button>
        </header>
        <div className={`summary${showSummary ? '' : ' clamped'}`}>{data.summary ? <Rich text={data.summary} /> : <span className="sub">No summary available.</span>}</div>
      </section>

      <div className="review">
        <div className="stack">
          <div className="card">
            <FrameViewer videoId={videoId} frame={frameQ.data} point={point} elements={elements} hoverIdx={hoverIdx} setHoverIdx={setHoverIdx}
              showBoxes={overlay || tab === 'evidence'} visible={typeVisible} />
            <div className="viewer-actions" style={{ marginTop: 8 }}>
              <label className="toggle"><input type="checkbox" checked={overlay} onChange={(e) => setOverlay(e.target.checked)} /> Element overlay</label>
              <span className="sub" style={{ marginLeft: 'auto' }}>
                <button className="btn small" disabled={idx <= 0} onClick={() => select(points[idx - 1].frame_id)} aria-label="Previous frame">←</button>{' '}
                <button className="btn small" disabled={idx >= points.length - 1} onClick={() => select(points[idx + 1].frame_id)} aria-label="Next frame">→</button>
              </span>
            </div>
          </div>
          <div className="card">
            <header><h2>Risk timeline</h2></header>
            <RiskTimeline data={data} selectedId={selected} onSelect={select} />
            <header style={{ marginTop: 14 }}><h3>Frames</h3></header>
            <Filmstrip points={points} selectedId={selected} onSelect={select} />
          </div>
        </div>

        <aside className="card" aria-label="Frame analysis panel">
          <div className="tabs" role="tablist">
            {TABS.map((t) => (
              <button key={t.id} role="tab" id={`tab-${t.id}`} aria-selected={tab === t.id} aria-controls={`panel-${t.id}`} onClick={() => setTab(t.id)}>
                {t.label}{t.id === 'events' ? ` (${data.events.length})` : ''}
              </button>
            ))}
          </div>
          <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
            {tab === 'details' && <DetailsPanel frame={frameQ.data} point={point} loading={frameQ.isLoading} />}
            {tab === 'events' && <EventList data={data} selectedId={selected} onSelect={select} />}
            {tab === 'evidence' && (
              <EvidencePanel videoId={videoId} filename={data.filename} point={point} elements={elements} loading={elQ.isLoading} error={elQ.error}
                hoverIdx={hoverIdx} setHoverIdx={setHoverIdx} typeFilter={typeFilter} setTypeFilter={setTypeFilter} />
            )}
            {tab === 'ask' && <AskPanel videoId={videoId} points={points} onSelect={select} />}
          </div>
        </aside>
      </div>
    </div>
  );
}
