import * as d3 from 'd3';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { RiskLevel, Timeline, TimelinePoint } from '../types';
import { ACTION_COLORS, BANDS, LEVELS, LEVEL_COLOR, LEVEL_LABEL, levelColor, mmss, prettyAction, scoreToLevel } from '../util';
import { useWidth } from './common';

export function Donut({ dist }: { dist: Record<RiskLevel, number> }) {
  const total = LEVELS.reduce((a, l) => a + (dist[l] ?? 0), 0);
  const size = 190, r = size / 2;
  const arcs = d3.pie<RiskLevel>().sort(null).value((l) => dist[l] ?? 0)(LEVELS);
  const arc = d3.arc<d3.PieArcDatum<RiskLevel>>().innerRadius(r * 0.62).outerRadius(r - 2).padAngle(0.03).cornerRadius(4);
  const label = LEVELS.map((l) => `${dist[l] ?? 0} ${l}`).join(', ');
  return (
    <div className="donut-wrap">
      <svg width={size} height={size} viewBox={`${-r} ${-r} ${size} ${size}`} role="img" aria-label={`Risk distribution of sessions: ${label}`}>
        {total === 0 && <circle r={r - 10} fill="none" stroke="var(--border)" strokeWidth={18} />}
        {arcs.map((a) => (
          <path key={a.data} d={arc(a) ?? ''} fill={LEVEL_COLOR[a.data]}>
            <title>{`${LEVEL_LABEL[a.data]}: ${dist[a.data] ?? 0}`}</title>
          </path>
        ))}
        <text className="donut-center" textAnchor="middle" y={6} fontSize={30}>{total}</text>
        <text textAnchor="middle" y={26} fontSize={11} fill="var(--muted)">sessions</text>
      </svg>
      <div style={{ display: 'grid', gap: 6 }}>
        {LEVELS.map((l) => (
          <div key={l} className="chip" style={{ ['--c' as string]: LEVEL_COLOR[l], textTransform: 'none' }}>
            <span className="dot" /> {LEVEL_LABEL[l]} <b>{dist[l] ?? 0}</b>
          </div>
        ))}
      </div>
    </div>
  );
}

export function ActionBars({ data }: { data: { action_type: string; count: number }[] }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const rows = [...data].sort((a, b) => b.count - a.count);
  const rowH = 30, m = { l: 110, r: 36, t: 4, b: 4 };
  const h = rows.length * rowH + m.t + m.b;
  const x = d3.scaleLinear().domain([0, d3.max(rows, (d) => d.count) || 1]).range([0, w - m.l - m.r]);
  return (
    <div className="chart" ref={ref}>
      <svg width={w} height={h} role="img" aria-label={`Action type breakdown: ${rows.map((r) => `${prettyAction(r.action_type)} ${r.count}`).join(', ')}`}>
        {rows.map((d, i) => (
          <g key={d.action_type} transform={`translate(0,${m.t + i * rowH})`}>
            <text x={m.l - 10} y={rowH / 2 + 4} textAnchor="end" style={{ fontSize: 12 }}>{prettyAction(d.action_type)}</text>
            <rect x={m.l} y={4} width={x(d.count)} height={rowH - 10} rx={5} fill={ACTION_COLORS[d.action_type] ?? ACTION_COLORS.other}>
              <title>{`${prettyAction(d.action_type)}: ${d.count} frames`}</title>
            </rect>
            <text x={m.l + x(d.count) + 6} y={rowH / 2 + 4} style={{ fill: 'var(--text)', fontWeight: 700 }}>{d.count}</text>
          </g>
        ))}
      </svg>
    </div>
  );
}

interface TimelineProps {
  data: Timeline;
  selectedId: number | null;
  onSelect: (frameId: number) => void;
  height?: number;
}

export function RiskTimeline({ data, selectedId, onSelect, height = 230 }: TimelineProps) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const m = { l: 36, r: 14, t: 10, b: 26 };
  const iw = w - m.l - m.r, ih = height - m.t - m.b;
  const pts = data.points.filter((p) => p.risk_score != null);
  const full: [number, number] = useMemo(() => {
    const e = d3.extent(data.points, (p) => p.t) as [number, number];
    return [e[0] ?? 0, Math.max(e[1] ?? 1, (e[0] ?? 0) + 1)];
  }, [data.points]);
  const [domain, setDomain] = useState<[number, number]>(full);
  useEffect(() => setDomain(full), [full]);
  const x = useMemo(() => d3.scaleLinear().domain(domain).range([0, iw]), [domain, iw]);
  const y = useMemo(() => d3.scaleLinear().domain([0, 100]).range([ih, 0]), [ih]);
  const [hover, setHover] = useState<TimelinePoint | null>(null);
  const axisX = useRef<SVGGElement>(null);
  const axisY = useRef<SVGGElement>(null);
  const brushG = useRef<SVGGElement>(null);
  const moved = useRef(false);

  useEffect(() => {
    if (axisX.current) d3.select(axisX.current).call(d3.axisBottom(x).ticks(Math.max(2, Math.floor(iw / 80))).tickFormat((v) => mmss(+v)) as any);
    if (axisY.current) d3.select(axisY.current).call(d3.axisLeft(y).tickValues([0, 25, 50, 75, 100]) as any);
  }, [x, y, iw]);

  const nearest = (clientX: number, el: Element): TimelinePoint | null => {
    const rect = el.getBoundingClientRect();
    const t = x.invert(clientX - rect.left);
    let best: TimelinePoint | null = null, bd = Infinity;
    for (const p of pts) { const d = Math.abs(p.t - t); if (d < bd) { bd = d; best = p; } }
    return best;
  };

  useEffect(() => {
    if (!brushG.current) return;
    const brush = d3.brushX().extent([[0, 0], [iw, ih]])
      .on('start', () => { moved.current = false; })
      .on('brush', () => { moved.current = true; })
      .on('end', (ev) => {
        if (!ev.selection || !ev.sourceEvent) return;
        const [a, b] = (ev.selection as [number, number]).map(x.invert);
        if (b - a >= 2) setDomain([a, b]);
        d3.select(brushG.current).call(brush.move as any, null);
      });
    const g = d3.select(brushG.current);
    g.call(brush);
    g.select('.selection').attr('fill', 'var(--accent)').attr('fill-opacity', 0.18).attr('stroke', 'var(--accent)');
  }, [x, iw, ih]);

  const area = d3.area<TimelinePoint>().x((p) => x(p.t)).y0(ih).y1((p) => y(p.risk_score!)).curve(d3.curveMonotoneX);
  const line = d3.line<TimelinePoint>().x((p) => x(p.t)).y((p) => y(p.risk_score!)).curve(d3.curveMonotoneX);
  const selected = data.points.find((p) => p.frame_id === selectedId);
  const zoomed = domain[0] !== full[0] || domain[1] !== full[1];
  const gid = `grad-${data.video_id}`;
  const clipId = `clip-${data.video_id}`;

  return (
    <div className="chart" ref={ref}>
      <div className="toolbar" style={{ justifyContent: 'space-between', marginBottom: 6 }}>
        <span className="sub">Drag to zoom · click to select a frame · ←/→ to step</span>
        {zoomed && <button className="btn small" onClick={() => setDomain(full)}>Reset zoom</button>}
      </div>
      <svg width={w} height={height} role="img" aria-label={`Risk timeline for ${data.filename ?? 'session'}. Peak risk ${d3.max(pts, (p) => p.risk_score) ?? 0}. ${data.events.length} high-risk events.`}>
        <defs>
          <linearGradient id={gid} x1="0" x2="0" y1="0" y2="1">
            <stop offset="0" stopColor="var(--risk-critical)" stopOpacity=".55" />
            <stop offset=".35" stopColor="var(--risk-high)" stopOpacity=".35" />
            <stop offset=".7" stopColor="var(--risk-medium)" stopOpacity=".22" />
            <stop offset="1" stopColor="var(--risk-low)" stopOpacity=".15" />
          </linearGradient>
          <clipPath id={clipId}><rect width={iw} height={ih + 2} y={-1} /></clipPath>
        </defs>
        <g transform={`translate(${m.l},${m.t})`}>
          {BANDS.map((b) => (
            <rect key={b.level} x={0} width={iw} y={y(b.to)} height={y(b.from) - y(b.to)} fill={LEVEL_COLOR[b.level]} opacity={0.1}>
              <title>{`${LEVEL_LABEL[b.level]} band (${b.from}–${b.to})`}</title>
            </rect>
          ))}
          <g ref={axisY} className="tick" />
          <g ref={axisX} transform={`translate(0,${ih})`} className="tick" />
          <g clipPath={`url(#${clipId})`}>
            <path d={area(pts) ?? ''} fill={`url(#${gid})`} />
            <path d={line(pts) ?? ''} fill="none" stroke="var(--text)" strokeWidth={2} strokeLinejoin="round" />
            {data.events.map((e) => {
              const p = data.points.find((q) => q.frame_id === e.frame_id);
              if (!p || p.risk_score == null) return null;
              return <circle key={e.frame_id} cx={x(p.t)} cy={y(p.risk_score)} r={4.5} fill={levelColor(e.risk_level, p.risk_score)} stroke="var(--panel)" strokeWidth={1.5} pointerEvents="none" />;
            })}
            {selected && (
              <g pointerEvents="none">
                <line x1={x(selected.t)} x2={x(selected.t)} y1={0} y2={ih} stroke="var(--accent)" strokeWidth={2} />
                <path d={`M${x(selected.t) - 6},0 h12 l-6,9 z`} fill="var(--accent)" />
              </g>
            )}
            {hover && (
              <circle cx={x(hover.t)} cy={y(hover.risk_score!)} r={6} fill="none" stroke="var(--text)" strokeWidth={2} pointerEvents="none" />
            )}
          </g>
          <g
            ref={brushG}
            onPointerMove={(e) => { if (!(e.buttons & 1)) setHover(nearest(e.clientX, e.currentTarget.querySelector('.overlay')!)); }}
            onPointerLeave={() => setHover(null)}
            onClick={(e) => { if (moved.current) return; const p = nearest(e.clientX, e.currentTarget.querySelector('.overlay')!); if (p) onSelect(p.frame_id); }}
            style={{ cursor: 'crosshair' }}
          />
        </g>
      </svg>
      {hover && (
        <div className="chart-tip" style={{ left: m.l + x(hover.t), top: m.t + y(hover.risk_score!) }} role="status">
          <b>{mmss(hover.t)}</b> · risk {hover.risk_score} ({scoreToLevel(hover.risk_score)}) · {prettyAction(hover.action_type)}
        </div>
      )}
    </div>
  );
}

export function OverlayChart({ series }: { series: { name: string; color: string; data: Timeline }[] }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const h = 240, m = { l: 36, r: 14, t: 10, b: 26 };
  const iw = w - m.l - m.r, ih = h - m.t - m.b;
  const maxT = Math.max(1, ...series.flatMap((s) => s.data.points.map((p) => p.t)));
  const x = d3.scaleLinear().domain([0, maxT]).range([0, iw]);
  const y = d3.scaleLinear().domain([0, 100]).range([ih, 0]);
  const line = d3.line<TimelinePoint>().x((p) => x(p.t)).y((p) => y(p.risk_score ?? 0)).curve(d3.curveMonotoneX);
  const ax = useRef<SVGGElement>(null), ay = useRef<SVGGElement>(null);
  useEffect(() => {
    if (ax.current) d3.select(ax.current).call(d3.axisBottom(x).ticks(Math.max(2, Math.floor(iw / 80))).tickFormat((v) => mmss(+v)) as any);
    if (ay.current) d3.select(ay.current).call(d3.axisLeft(y).tickValues([0, 25, 50, 75, 100]) as any);
  });
  return (
    <div className="chart" ref={ref}>
      <svg width={w} height={h} role="img" aria-label={`Overlaid risk timelines: ${series.map((s) => s.name).join(' vs ')}`}>
        <g transform={`translate(${m.l},${m.t})`}>
          {BANDS.map((b) => <rect key={b.level} x={0} width={iw} y={y(b.to)} height={y(b.from) - y(b.to)} fill={LEVEL_COLOR[b.level]} opacity={0.1} />)}
          <g ref={ay} className="tick" /><g ref={ax} transform={`translate(0,${ih})`} className="tick" />
          {series.map((s) => <path key={s.name} d={line(s.data.points) ?? ''} fill="none" stroke={s.color} strokeWidth={2.5} strokeLinejoin="round" />)}
        </g>
      </svg>
      <div className="legend">{series.map((s) => <span key={s.name} style={{ ['--c' as string]: s.color }}><i /> {s.name}</span>)}</div>
    </div>
  );
}

export function Heatmap({ data, onPick }: { data: Timeline; onPick?: (frameId: number) => void }) {
  const [ref, w] = useWidth<HTMLDivElement>();
  const [hov, setHov] = useState<{ a: string; b: number } | null>(null);
  const bucket = Math.max(data.interval_s ?? 3, Math.ceil((data.duration_s ?? 75) / 12 / 3) * 3);
  const maxT = Math.max(...data.points.map((p) => p.t), 0);
  const nb = Math.floor(maxT / bucket) + 1;
  const actions = Array.from(new Set(data.points.map((p) => p.action_type ?? 'other'))).sort();
  const cells = new Map<string, { risk: number; frame: number; n: number }>();
  for (const p of data.points) {
    const k = `${p.action_type ?? 'other'}|${Math.floor(p.t / bucket)}`;
    const c = cells.get(k);
    const r = p.risk_score ?? 0;
    if (!c) cells.set(k, { risk: r, frame: p.frame_id, n: 1 });
    else { c.n++; if (r > c.risk) { c.risk = r; c.frame = p.frame_id; } }
  }
  const m = { l: 110, t: 8, b: 26, r: 8 };
  const cw = (w - m.l - m.r) / nb, ch = 34;
  const h = actions.length * ch + m.t + m.b;
  const color = (r: number) => LEVEL_COLOR[scoreToLevel(r)!];
  return (
    <div className="chart heat" ref={ref}>
      <svg width={w} height={h} role="img" aria-label="Heatmap of maximum risk by time bucket and action type">
        {actions.map((a, i) => (
          <text key={a} x={m.l - 10} y={m.t + i * ch + ch / 2 + 4} textAnchor="end" style={{ fontSize: 12 }}>{prettyAction(a)}</text>
        ))}
        {Array.from({ length: nb }, (_, b) => (
          <text key={b} x={m.l + b * cw + cw / 2} y={h - 8} textAnchor="middle">{mmss(b * bucket)}</text>
        ))}
        {actions.flatMap((a, i) =>
          Array.from({ length: nb }, (_, b) => {
            const c = cells.get(`${a}|${b}`);
            const on = hov?.a === a && hov.b === b;
            return (
              <rect key={`${a}${b}`} x={m.l + b * cw + 2} y={m.t + i * ch + 2} width={Math.max(1, cw - 4)} height={ch - 4} rx={6}
                fill={c ? color(c.risk) : 'var(--panel-2)'} opacity={c ? 0.35 + (c.risk / 100) * 0.65 : 0.6}
                stroke={on ? 'var(--text)' : 'none'} strokeWidth={2}
                style={{ cursor: c ? 'pointer' : 'default' }}
                onMouseEnter={() => setHov({ a, b })} onMouseLeave={() => setHov(null)}
                onClick={() => c && onPick?.(c.frame)}>
                <title>{c ? `${prettyAction(a)} @ ${mmss(b * bucket)}–${mmss((b + 1) * bucket)}: max risk ${c.risk} (${c.n} frames)` : 'No activity'}</title>
              </rect>
            );
          }),
        )}
      </svg>
    </div>
  );
}
