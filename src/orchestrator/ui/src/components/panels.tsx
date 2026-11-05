import { useMutation } from '@tanstack/react-query';
import { Fragment, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { api, imageUrl } from '../api';
import type { FrameDetail, Timeline, TimelinePoint, UiElement } from '../types';
import { levelColor, mmss, prettyAction } from '../util';
import { ErrorMessage, RiskBadge, ScoreBar, Skeleton, Empty } from './common';
import { elementColor } from './viewer';

export function Rich({ text }: { text: string }) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return <>{parts.map((p, i) => (p.startsWith('**') && p.endsWith('**') && p.length > 4 ? <strong key={i}>{p.slice(2, -2)}</strong> : <Fragment key={i}>{p}</Fragment>))}</>;
}

export function DetailsPanel({ frame, point, loading }: { frame?: FrameDetail; point?: TimelinePoint; loading: boolean }) {
  if (!point) return <Empty icon="🖼️" title="Select a frame" />;
  const color = levelColor(point.risk_level, point.risk_score);
  return (
    <div>
      <dl className="kv" style={{ margin: 0 }}>
        <dt>Timestamp</dt><dd><b style={{ fontVariantNumeric: 'tabular-nums' }}>{mmss(point.t)}</b> <span className="sub">· frame #{point.frame_id}</span></dd>
        <dt>Action</dt><dd><span className="chip"><span className="dot" />{prettyAction(point.action_type)}</span></dd>
        <dt>Risk</dt><dd style={{ display: 'flex', gap: 10, alignItems: 'center' }}><RiskBadge level={point.risk_level} score={point.risk_score} /><ScoreBar score={point.risk_score} level={point.risk_level} /></dd>
      </dl>
      <div className="desc" style={{ ['--c' as string]: color }}>
        <h3 style={{ marginBottom: 6 }}>AI description</h3>
        {loading && !frame ? <><Skeleton /><div style={{ height: 6 }} /><Skeleton w="70%" /></> : <p style={{ margin: 0 }}>{frame?.image_description ?? 'No description available.'}</p>}
      </div>
    </div>
  );
}

export function EventList({ data, selectedId, onSelect }: { data: Timeline; selectedId: number | null; onSelect: (id: number) => void }) {
  if (!data.events.length) return <Empty icon="✅" title="No high-risk events">No frames scored 50 or above.</Empty>;
  return (
    <div className="scroll">
      {data.events.map((e) => (
        <button key={e.frame_id} className="event" aria-current={e.frame_id === selectedId} onClick={() => onSelect(e.frame_id)}>
          <span className="time">{mmss(e.t)}</span>
          <span style={{ flex: 1 }}>
            <span style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
              <RiskBadge level={e.risk_level} /> <span className="chip">{prettyAction(e.action_type)}</span>
            </span>
            <p>{e.description}</p>
          </span>
        </button>
      ))}
    </div>
  );
}

interface EvidenceProps {
  videoId: number;
  filename: string | null;
  point?: TimelinePoint;
  elements: UiElement[];
  loading: boolean;
  error: unknown;
  hoverIdx: number | null;
  setHoverIdx: (i: number | null) => void;
  typeFilter: string | null;
  setTypeFilter: (t: string | null) => void;
}

export function EvidencePanel({ videoId, filename, point, elements, loading, error, hoverIdx, setHoverIdx, typeFilter, setTypeFilter }: EvidenceProps) {
  const [copied, setCopied] = useState(false);
  const types = useMemo(() => Array.from(new Set(elements.map((e) => e.type))), [elements]);
  const citation = point ? `Session: ${filename ?? `#${videoId}`} (video ${videoId}) | Frame ${point.frame_id} | t=${mmss(point.t)} | risk ${point.risk_score ?? 'n/a'} (${point.risk_level ?? 'n/a'}) | action: ${prettyAction(point.action_type)}` : '';
  const copy = async () => {
    try { await navigator.clipboard.writeText(citation); } catch {
      const ta = document.createElement('textarea'); ta.value = citation; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
    }
    setCopied(true); setTimeout(() => setCopied(false), 1800);
  };
  return (
    <div>
      <div className="toolbar" style={{ justifyContent: 'space-between', marginBottom: 10 }}>
        <span className="sub">Hover boxes on the frame or items below.</span>
        <button className="btn small primary" onClick={copy} disabled={!point}>{copied ? '✓ Copied' : '📋 Copy citation'}</button>
      </div>
      {error ? <ErrorMessage error={error} /> : loading ? <Skeleton h={120} /> : !elements.length ? (
        <Empty icon="🔍" title="No UI elements parsed for this frame" />
      ) : (
        <>
          <div className="toolbar" style={{ marginBottom: 8 }} role="group" aria-label="Filter elements by type">
            <button className="chip" aria-pressed={typeFilter == null} onClick={() => setTypeFilter(null)}>All {elements.length}</button>
            {types.map((t) => (
              <button key={t} className="chip" aria-pressed={typeFilter === t} onClick={() => setTypeFilter(typeFilter === t ? null : t)}>
                <span className="dot" style={{ ['--c' as string]: elementColor(t) }} />{t.replace(/_/g, ' ')}
              </button>
            ))}
          </div>
          <div className="scroll" style={{ maxHeight: 300 }}>
            {elements.map((e, i) => (typeFilter && e.type !== typeFilter ? null : (
              <button key={i} className={`el-row${hoverIdx === i ? ' hl' : ''}`} onMouseEnter={() => setHoverIdx(i)} onMouseLeave={() => setHoverIdx(null)} onFocus={() => setHoverIdx(i)} onBlur={() => setHoverIdx(null)}>
                <span><span className="dot" style={{ display: 'inline-block', width: 8, height: 8, borderRadius: 4, background: elementColor(e.type), marginRight: 8 }} />{e.content || '(no text)'}</span>
                <span className="sub" style={{ textTransform: 'capitalize' }}>{e.type.replace(/_/g, ' ')}{e.interactivity ? ' · interactive' : ''}</span>
              </button>
            )))}
          </div>
        </>
      )}
    </div>
  );
}

const SUGGESTIONS = [
  'Was any sensitive data exported or uploaded?',
  'Were any accounts or permissions changed?',
  'Did the user disable security tools?',
  'Summarize the riskiest moments',
];

interface Msg { q: string; a?: { answer: string | null; note: string | null; matches: { frame_id: number; t: number | null; sim: number }[] }; err?: unknown }

export function AskPanel({ videoId, points, onSelect }: { videoId: number; points: TimelinePoint[]; onSelect: (id: number) => void }) {
  const [text, setText] = useState('');
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const endRef = useRef<HTMLDivElement>(null);
  const byId = useMemo(() => new Map(points.map((p) => [p.frame_id, p])), [points]);
  const nearestByT = (sec: number) => points.reduce<TimelinePoint | null>((b, p) => (!b || Math.abs(p.t - sec) < Math.abs(b.t - sec) ? p : b), null);

  const mut = useMutation({
    mutationFn: (q: string) => api.ask(videoId, q, 5),
    onSuccess: (r, q) => setMsgs((m) => [...m, { q, a: { answer: r.answer, note: r.note, matches: r.matches.map((x) => ({ frame_id: x.frame_id, t: x.frame_timestamp, sim: x.similarity })) } }]),
    onError: (e, q) => setMsgs((m) => [...m, { q, err: e }]),
  });
  useEffect(() => { endRef.current?.scrollTo({ top: endRef.current.scrollHeight, behavior: 'smooth' }); }, [msgs.length]);
  const submit = (q: string) => { const v = q.trim(); if (!v || mut.isPending) return; setText(''); mut.mutate(v); };

  const renderAnswer = (s: string): ReactNode[] => {
    const re = /(frame(?:\s*id)?\s*#?\s*(\d+)(?:\s*\(\d{1,2}:\d{2}\))?)|(\b\d{1,2}:\d{2}\b)/gi;
    const out: ReactNode[] = [];
    let last = 0, m: RegExpExecArray | null, k = 0;
    while ((m = re.exec(s))) {
      let target: TimelinePoint | undefined | null = null;
      if (m[2]) target = byId.get(+m[2]);
      else if (m[3]) { const [mi, se] = m[3].split(':').map(Number); target = nearestByT(mi * 60 + se); }
      out.push(s.slice(last, m.index));
      out.push(target ? <button key={k++} className="chip" style={{ textTransform: 'none' }} onClick={() => onSelect(target!.frame_id)} title={`Jump to ${mmss(target.t)}`}>🎞 {m[0]}</button> : m[0]);
      last = m.index + m[0].length;
    }
    out.push(s.slice(last));
    return out;
  };

  return (
    <div className="chat">
      {!msgs.length && <p className="sub" style={{ margin: 0 }}>Ask questions about this session. Answers cite frames you can jump to.</p>}
      <div className="suggest" aria-label="Suggested questions">
        {SUGGESTIONS.map((s) => <button key={s} className="chip" style={{ textTransform: 'none' }} onClick={() => submit(s)} disabled={mut.isPending}>{s}</button>)}
      </div>
      <div className="scroll" ref={endRef} style={{ display: 'grid', gap: 10, maxHeight: 420 }} aria-live="polite">
        {msgs.map((m, i) => (
          <Fragment key={i}>
            <div className="msg q">{m.q}</div>
            {m.err ? <ErrorMessage error={m.err} /> : (
              <div className="msg a">
                {m.a?.answer ? renderAnswer(m.a.answer) : <span className="sub">No generated answer.</span>}
                {m.a?.note && <div className="alert warn" style={{ marginTop: 8 }}><small>{m.a.note}</small></div>}
                {!!m.a?.matches.length && (
                  <div className="cite-grid">
                    <h3>Matching frames</h3>
                    {m.a.matches.map((x) => (
                      <button key={x.frame_id} className="cite" onClick={() => onSelect(x.frame_id)}>
                        <img src={imageUrl(x.frame_id)} alt="" loading="lazy" />
                        <span><b>{mmss(x.t)}</b> · frame #{x.frame_id}<small>{byId.get(x.frame_id)?.action_type ? prettyAction(byId.get(x.frame_id)!.action_type) : ''}</small></span>
                        <span className="sim" title="Similarity">{Math.round(x.sim * 100)}%</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </Fragment>
        ))}
        {mut.isPending && <div className="msg a typing" role="status" aria-label="Thinking"><span /><span /><span /></div>}
      </div>
      <form className="askbox" onSubmit={(e) => { e.preventDefault(); submit(text); }}>
        <label className="sr-only" htmlFor="ask-input">Question</label>
        <input id="ask-input" type="text" value={text} onChange={(e) => setText(e.target.value)} placeholder="Ask about this session…" maxLength={2000} />
        <button className="btn primary" disabled={!text.trim() || mut.isPending}>Ask</button>
      </form>
    </div>
  );
}
