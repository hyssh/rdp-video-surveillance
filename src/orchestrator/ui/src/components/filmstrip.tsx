import { useEffect, useRef } from 'react';
import { imageUrl } from '../api';
import type { TimelinePoint } from '../types';
import { levelColor, mmss } from '../util';

export function Filmstrip({ points, selectedId, onSelect }: { points: TimelinePoint[]; selectedId: number | null; onSelect: (id: number) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current?.querySelector<HTMLElement>('[aria-current="true"]');
    const box = ref.current;
    if (el && box) box.scrollTo({ left: el.offsetLeft - box.clientWidth / 2 + el.clientWidth / 2, behavior: 'smooth' });
  }, [selectedId]);
  return (
    <div className="film" ref={ref} role="listbox" aria-label="Frame filmstrip">
      {points.map((p) => (
        <button key={p.frame_id} className="thumb" role="option" aria-selected={p.frame_id === selectedId}
          aria-current={p.frame_id === selectedId} onClick={() => onSelect(p.frame_id)}
          style={{ ['--c' as string]: levelColor(p.risk_level, p.risk_score) }}
          aria-label={`Frame at ${mmss(p.t)}, risk ${p.risk_score ?? 'unknown'}`}>
          <img src={imageUrl(p.frame_id)} alt="" loading="lazy" />
          <span><b>{mmss(p.t)}</b><em style={{ fontStyle: 'normal' }}>{p.risk_score ?? '–'}</em></span>
          <div className="tbar" />
        </button>
      ))}
    </div>
  );
}
