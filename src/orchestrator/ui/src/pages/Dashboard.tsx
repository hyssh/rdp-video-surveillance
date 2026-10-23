import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { api, friendlyError } from '../api';
import { Donut } from '../components/charts';
import { Empty, ErrorMessage, Legend, RiskBadge, ScoreBar, Skeleton } from '../components/common';
import type { JobStatus, RiskLevel } from '../types';
import { LEVELS, LEVEL_LABEL, fmtDate, toLevel } from '../util';

const STEPS = ['queued', 'extracting', 'analyzing', 'summarizing', 'done'] as const;

function UploadCard() {
  const qc = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [fileName, setFileName] = useState('');
  const [err, setErr] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);

  const job = useQuery({
    queryKey: ['job', jobId],
    queryFn: async () => {
      const j = await api.job(jobId!);
      if (j.status === 'done') { qc.invalidateQueries({ queryKey: ['stats'] }); qc.invalidateQueries({ queryKey: ['videos'] }); }
      return j;
    },
    enabled: !!jobId,
    refetchInterval: (q) => (q.state.data && (q.state.data.status === 'done' || q.state.data.status === 'failed') ? false : 1500),
  });

  const start = async (f: File | undefined) => {
    if (!f) return;
    setErr(null);
    if (!/\.(mp4|m4v|mov|webm)$/i.test(f.name) && !f.type.startsWith('video/')) {
      setErr({ title: 'Please choose a video file (.mp4).' });
      return;
    }
    setBusy(true); setFileName(f.name); setJobId(null);
    try {
      const r = await api.ingest(f);
      setJobId(r.job_id);
    } catch (e) { setErr(e); }
    setBusy(false);
  };

  const j: JobStatus | undefined = job.data;
  const stepIdx = j ? STEPS.indexOf(j.status as (typeof STEPS)[number]) : jobId ? 0 : -1;
  const failed = j?.status === 'failed';
  const pct = j?.status === 'done' ? 100 : j && j.frames_total ? Math.min(100, Math.round((j.frames_done / j.frames_total) * 100)) : 0;

  return (
    <section className="card" aria-labelledby="up-h">
      <header><h2 id="up-h">Analyze a new recording</h2></header>
      <div
        className={`drop${over ? ' over' : ''}`} role="button" tabIndex={0}
        onClick={() => input.current?.click()} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && (e.preventDefault(), input.current?.click())}
        onDragOver={(e) => { e.preventDefault(); setOver(true); }} onDragLeave={() => setOver(false)}
        onDrop={(e) => { e.preventDefault(); setOver(false); start(e.dataTransfer.files[0]); }}
        aria-label="Upload an mp4 recording: drop a file or press Enter to browse"
      >
        <div style={{ fontSize: 28 }} aria-hidden="true">🎬</div>
        <b>{busy ? 'Uploading…' : 'Drop an RDP recording (.mp4) here'}</b>
        <div className="sub">or click to browse</div>
        <input ref={input} type="file" accept="video/*,.mp4" hidden onChange={(e) => { start(e.target.files?.[0]); e.target.value = ''; }} />
      </div>
      {err != null && <div style={{ marginTop: 12 }}>{(err as any).title ? <div className="alert" role="alert"><b>{(err as any).title}</b></div> : <ErrorMessage error={err} />}</div>}
      {jobId && (
        <div style={{ marginTop: 14 }} aria-live="polite">
          <div className="sub" style={{ display: 'flex', justifyContent: 'space-between' }}><span>{fileName}</span><span>Job {jobId.slice(0, 8)}</span></div>
          <div className="stepper">
            {STEPS.map((s, i) => (
              <div key={s} style={{ display: 'contents' }}>
                <div className={`step${failed && i === Math.max(stepIdx, 0) ? ' failed' : i < stepIdx || j?.status === 'done' ? ' done' : i === stepIdx ? ' active' : ''}`}>
                  <span className="n">{i < stepIdx || j?.status === 'done' ? '✓' : failed && i === Math.max(stepIdx, 0) ? '!' : i + 1}</span>{s}
                </div>
                {i < STEPS.length - 1 && <div className={`stepline${i < stepIdx ? ' done' : ''}`} />}
              </div>
            ))}
          </div>
          {(j?.status === 'analyzing' || j?.status === 'summarizing' || j?.status === 'done') && (
            <>
              <div className="progress" role="progressbar" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100} aria-label="Frames analyzed"><div style={{ width: `${pct}%` }} /></div>
              <div className="sub">{j.frames_done}/{j.frames_total} frames{j.frames_failed ? ` · ${j.frames_failed} failed` : ''}</div>
            </>
          )}
          {job.isError && <div style={{ marginTop: 8 }}><ErrorMessage error={job.error} /></div>}
          {failed && <div className="alert" role="alert" style={{ marginTop: 8 }}><b>Analysis failed</b>{j?.error ?? 'The pipeline reported an error.'}{j?.stage && <small> (stage: {j.stage})</small>}</div>}
          {j?.warnings?.map((w, i) => <div key={i} className="alert warn" style={{ marginTop: 8 }}><small>⚠ {w}</small></div>)}
          {j?.status === 'done' && j.video_id && (
            <div className="alert info" style={{ marginTop: 8 }}><b>Analysis complete</b><Link to={`/session/${j.video_id}`}>Open session review →</Link></div>
          )}
        </div>
      )}
    </section>
  );
}

export default function Dashboard() {
  const stats = useQuery({ queryKey: ['stats'], queryFn: api.stats });
  const vids = useQuery({ queryKey: ['videos'], queryFn: api.videos });
  const [q, setQ] = useState('');
  const [risk, setRisk] = useState<'all' | RiskLevel>('all');

  const rows = useMemo(
    () => (vids.data?.items ?? []).filter((v) => (!q || (v.filename ?? '').toLowerCase().includes(q.toLowerCase())) && (risk === 'all' || toLevel(v.security_level, v.risk_score) === risk))
      .sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? '')),
    [vids.data, q, risk],
  );
  const s = stats.data;
  const kpis: [string, number | undefined, string, string][] = [
    ['Sessions', s?.videos_total, 'recordings reviewed', 'var(--accent)'],
    ['High-risk sessions', s?.high_risk_videos, 'high or critical', 'var(--risk-critical)'],
    ['Frames analyzed', s?.frames_total, 'sampled every ~3s', 'var(--risk-medium)'],
    ['Frames described', s?.frames_described, 'AI descriptions', 'var(--risk-low)'],
  ];

  return (
    <div className="page">
      <div className="page-head">
        <div><h1>Security overview</h1><p className="sub">AI review of privileged RDP session recordings — spot risky activity in seconds.</p></div>
        <Legend />
      </div>
      {stats.isError && <div style={{ marginBottom: 16 }}><ErrorMessage error={stats.error} onRetry={() => stats.refetch()} /></div>}
      <div className="grid kpis">
        {kpis.map(([label, val, hint, c]) => (
          <div key={label} className="card kpi" style={{ ['--kpi' as string]: c }}>
            <div className="label">{label}</div>
            <div className="value">{val == null ? <Skeleton h={34} w={70} /> : val.toLocaleString()}</div>
            <div className="hint">{hint}</div>
          </div>
        ))}
      </div>
      <div className="grid cols-2">
        <section className="card" aria-labelledby="rs-h">
          <header><h2 id="rs-h">Recent sessions</h2><span className="sub">{rows.length} shown</span></header>
          <div className="toolbar" style={{ marginBottom: 10 }}>
            <label className="sr-only" htmlFor="search">Search sessions</label>
            <input id="search" type="search" placeholder="Search filename…" value={q} onChange={(e) => setQ(e.target.value)} />
            <label className="sr-only" htmlFor="rf">Filter by risk</label>
            <select id="rf" value={risk} onChange={(e) => setRisk(e.target.value as 'all' | RiskLevel)}>
              <option value="all">All risk levels</option>
              {LEVELS.map((l) => <option key={l} value={l}>{LEVEL_LABEL[l]}</option>)}
            </select>
          </div>
          {vids.isError ? <ErrorMessage error={vids.error} onRetry={() => vids.refetch()} /> : vids.isLoading ? (
            <div style={{ display: 'grid', gap: 10 }}>{[0, 1, 2, 3].map((i) => <Skeleton key={i} h={36} />)}</div>
          ) : !rows.length ? (
            <Empty icon="📼" title={vids.data?.items.length ? 'No sessions match your filters' : 'No sessions yet'}>{vids.data?.items.length ? 'Try clearing the search or risk filter.' : 'Upload an RDP recording to get started.'}</Empty>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table>
                <thead><tr><th>Session</th><th>Risk</th><th>Score</th><th>Recorded</th></tr></thead>
                <tbody>
                  {rows.map((v) => (
                    <tr key={v.id}>
                      <td><Link className="file" to={`/session/${v.id}`}>{v.filename ?? `Session #${v.id}`}</Link></td>
                      <td><RiskBadge level={v.security_level} score={v.risk_score} /></td>
                      <td><ScoreBar score={v.risk_score} level={v.security_level} /></td>
                      <td className="sub">{fmtDate(v.created_at)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
        <div className="stack">
          <section className="card" aria-labelledby="rd-h">
            <header><h2 id="rd-h">Risk distribution</h2></header>
            {s ? <Donut dist={s.risk_distribution} /> : <Skeleton h={190} />}
          </section>
          <UploadCard />
        </div>
      </div>
    </div>
  );
}
