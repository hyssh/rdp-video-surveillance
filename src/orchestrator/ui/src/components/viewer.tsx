import { useEffect, useRef, useState } from 'react';
import { api, friendlyError, imageUrl } from '../api';
import type { FrameDetail, TimelinePoint, UiElement } from '../types';
import { levelColor, mmss, prettyAction, toLevel } from '../util';
import { Skeleton } from './common';

export function elementColor(type: string) {
  const palette = ['#5b9dff', '#a78bfa', '#2dd4bf', '#f472b6', '#fbbf24', '#fb7185', '#34d399'];
  let h = 0;
  for (const c of type) h = (h * 31 + c.charCodeAt(0)) >>> 0;
  return palette[h % palette.length];
}

interface Props {
  videoId: number;
  frame: FrameDetail | undefined;
  point: TimelinePoint | undefined;
  elements: UiElement[];
  showBoxes: boolean;
  hoverIdx: number | null;
  setHoverIdx: (i: number | null) => void;
  visible?: (e: UiElement) => boolean;
}

export function FrameViewer({ videoId, frame, point, elements, showBoxes, hoverIdx, setHoverIdx, visible }: Props) {
  const [clip, setClip] = useState<{ state: 'idle' | 'loading' | 'ready' | 'error'; url?: string; msg?: string }>({ state: 'idle' });
  const [imgErr, setImgErr] = useState(false);
  const frameId = point?.frame_id;
  const clipToken = useRef(0);

  useEffect(() => {
    setImgErr(false);
    clipToken.current++; // invalidate any in-flight clip request for the previous frame
    setClip((c) => { if (c.url) URL.revokeObjectURL(c.url); return { state: 'idle' }; });
  }, [frameId]);

  useEffect(() => () => { clipToken.current++; }, []);

  const playClip = async () => {
    if (point == null) return;
    const token = ++clipToken.current;
    setClip({ state: 'loading' });
    try {
      const url = await api.clip(videoId, point.t);
      if (token !== clipToken.current) { URL.revokeObjectURL(url); return; }
      setClip({ state: 'ready', url });
    } catch (e) {
      if (token === clipToken.current) setClip({ state: 'error', msg: friendlyError(e).title });
    }
  };

  const color = levelColor(point?.risk_level, point?.risk_score);
  const hov = hoverIdx != null ? elements[hoverIdx] : null;

  return (
    <div>
      <div className="viewer" style={{ ['--c' as string]: color }}>
        {!point ? (
          <Skeleton h={300} />
        ) : clip.state === 'ready' ? (
          <video src={clip.url} controls autoPlay playsInline aria-label={`Clip around ${mmss(point.t)}`} />
        ) : imgErr ? (
          <div className="empty">Frame image unavailable</div>
        ) : (
          <div className="imgwrap">
            <img src={imageUrl(point.frame_id)} alt={frame?.image_description ?? `Frame at ${mmss(point.t)}`} onError={() => setImgErr(true)} />
            {showBoxes && (
              <svg className="overlay" viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="UI element overlay">
                {elements.map((e, i) => {
                  if (visible && !visible(e)) return null;
                  if (!e.bbox) return null;
                  const [x1, y1, x2, y2] = e.bbox;
                  const on = hoverIdx === i;
                  const c = elementColor(e.type);
                  return (
                    <rect key={i} x={x1} y={y1} width={x2 - x1} height={y2 - y1} fill={on ? c : 'transparent'} fillOpacity={on ? 0.28 : 0}
                      stroke={c} strokeWidth={on ? 3 : 1.5} vectorEffect="non-scaling-stroke" style={{ cursor: 'pointer', transition: 'fill-opacity .12s' }}
                      onMouseEnter={() => setHoverIdx(i)} onMouseLeave={() => setHoverIdx(null)} />
                  );
                })}
              </svg>
            )}
            {point && (
              <div className="hud">
                <span className="pill">{mmss(point.t)}</span>
                <span className="badge" style={{ ['--c' as string]: color }}>{toLevel(point.risk_level, point.risk_score) ?? '—'} · {point.risk_score ?? '—'}</span>
                <span className="pill" style={{ textTransform: 'capitalize' }}>{prettyAction(point.action_type)}</span>
              </div>
            )}
            {showBoxes && hov && (
              <div className="tip"><b style={{ textTransform: 'capitalize' }}>{hov.type.replace(/_/g, ' ')}</b>{hov.content ? ` — ${hov.content}` : ''}</div>
            )}
          </div>
        )}
      </div>
      <div className="viewer-actions">
        {clip.state === 'ready' ? (
          <button className="btn" onClick={() => setClip({ state: 'idle' })}>← Back to frame</button>
        ) : (
          <button className="btn primary" onClick={playClip} disabled={!point || clip.state === 'loading'}>
            {clip.state === 'loading' ? 'Loading clip…' : '▶ Play ±3s clip'}
          </button>
        )}
        {clip.state === 'error' && (
          <span className="alert info" role="status" style={{ padding: '6px 10px' }}>
            {clip.msg}
          </span>
        )}
      </div>
    </div>
  );
}
