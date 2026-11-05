import type { IncomingMessage, ServerResponse } from 'node:http';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

/* eslint-disable @typescript-eslint/no-explicit-any */
const here = dirname(fileURLToPath(import.meta.url));
const sessions: any[] = JSON.parse(readFileSync(join(here, 'sessions.json'), 'utf8'));
const jobs = new Map<string, { start: number; videoId: number; fail: boolean }>();

const COLORS: Record<string, string> = { low: '#22c55e', medium: '#f59e0b', high: '#f97316', critical: '#ef4444' };
const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]!);

function json(res: ServerResponse, status: number, body: unknown) {
  res.statusCode = status;
  res.setHeader('Content-Type', 'application/json');
  res.end(JSON.stringify(body));
}
const err = (res: ServerResponse, status: number, error: string, detail: string, stage = 'request') =>
  json(res, status, { error, stage, detail });

const findFrame = (id: number) => {
  for (const s of sessions) {
    const f = s.frames.find((x: any) => x.id === id);
    if (f) return { s, f };
  }
  return null;
};

function frameSvg(s: any, f: any): string {
  const W = 1280, H = 720;
  const c = COLORS[f.risk_level];
  const els = f.elements
    .map((e: any, i: number) => {
      const [x1, y1, x2, y2] = e.bbox;
      const x = x1 * W, y = y1 * H, w = (x2 - x1) * W, h = (y2 - y1) * H;
      const fill = e.type === 'title_bar' ? '#2b3a55' : e.type === 'button' ? '#3b6fd4' : e.type === 'icon' ? '#1b2436' : '#1c2740';
      return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="6" fill="${fill}" stroke="#44567a" stroke-width="1"/>` +
        `<text x="${x + 10}" y="${y + Math.min(h / 2 + 5, 24)}" fill="#cbd5e1" font-size="${i === 0 ? 16 : 14}" font-family="Segoe UI,Arial,sans-serif">${esc(e.content)}</text>`;
    })
    .join('');
  const desc = esc(f.description).match(/.{1,70}(\s|$)/g) || [];
  const lines = desc.slice(0, 3).map((l, i) => `<tspan x="40" dy="${i ? 24 : 0}">${l.trim()}</tspan>`).join('');
  return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0f1b33"/><stop offset="1" stop-color="#17294d"/></linearGradient></defs>
<rect width="${W}" height="${H}" fill="url(#g)"/>${els}
<rect x="0" y="${H - 36}" width="${W}" height="36" fill="#0b1220" opacity=".9"/>
<text x="14" y="${H - 12}" fill="#94a3b8" font-size="14" font-family="Segoe UI,Arial,sans-serif">${esc(s.filename)} · frame ${f.id} · ${f.action_type}</text>
<rect x="${W - 200}" y="${H - 30}" width="180" height="24" rx="12" fill="${c}"/>
<text x="${W - 110}" y="${H - 13}" fill="#0b1220" font-size="14" font-weight="700" text-anchor="middle" font-family="Segoe UI,Arial,sans-serif">${f.risk_level.toUpperCase()} · ${f.risk_score}</text>
<text x="40" y="${H * 0.78}" fill="#e2e8f0" font-size="20" font-family="Segoe UI,Arial,sans-serif" opacity=".92">${lines}</text>
</svg>`;
}

function askAnswer(s: any, q: string, topK: number) {
  const words = q.toLowerCase().split(/\W+/).filter((w) => w.length > 2);
  const scored = s.frames.map((f: any) => {
    const text = (f.description + ' ' + f.action_type).toLowerCase();
    const hits = words.filter((w) => text.includes(w)).length;
    return { f, sim: Math.min(0.97, 0.35 + hits * 0.16 + f.risk_score / 400) };
  });
  scored.sort((a: any, b: any) => b.sim - a.sim);
  const top = scored.slice(0, topK);
  const mm = (t: number) => `${String(Math.floor(t / 60)).padStart(2, '0')}:${String(t % 60).padStart(2, '0')}`;
  const cites = top.slice(0, 3).map((m: any) => `frame ${m.f.id} (${mm(m.f.t)}): ${m.f.description}`);
  return {
    video_id: s.id,
    question: q,
    answer: `Based on the session "${s.filename}", the most relevant activity is:\n\n${cites.map((c: string) => `- ${c}`).join('\n')}\n\nSee frame ${top[0].f.id} at ${mm(top[0].f.t)} for the strongest evidence.`,
    matches: top.map((m: any) => ({ frame_id: m.f.id, frame_timestamp: m.f.t, image_description: m.f.description, similarity: Number(m.sim.toFixed(3)), url: `/frames/${m.f.id}/image` })),
    note: null,
  };
}

function stats() {
  const dist: Record<string, number> = { low: 0, medium: 0, high: 0, critical: 0 };
  const types: Record<string, number> = {};
  let frames = 0;
  for (const s of sessions) {
    dist[s.security_level]++;
    frames += s.frames.length;
    for (const f of s.frames) types[f.action_type] = (types[f.action_type] || 0) + 1;
  }
  return {
    videos_total: sessions.length,
    frames_total: frames,
    frames_described: frames,
    high_risk_videos: dist.high + dist.critical,
    risk_distribution: dist,
    action_types: Object.entries(types).map(([action_type, count]) => ({ action_type, count })).sort((a, b) => b.count - a.count),
    recent_videos: sessions.map((s) => ({ id: s.id, filename: s.filename, security_level: s.security_level, risk_score: s.risk_score, created_at: s.created_at })),
  };
}

const videoItem = (s: any) => ({ id: s.id, filename: s.filename, security_level: s.security_level, risk_score: s.risk_score, created_at: s.created_at });

export function mockMiddleware() {
  return async (req: IncomingMessage, res: ServerResponse, next: () => void) => {
    const url = new URL(req.url || '/', 'http://localhost');
    const p = url.pathname;
    let m: RegExpMatchArray | null;
    const delay = () => new Promise((r) => setTimeout(r, 120));
    try {
      if (p === '/stats') { await delay(); return json(res, 200, stats()); }
      if (p === '/videos' && req.method === 'GET') { await delay(); return json(res, 200, { items: sessions.map(videoItem), total: sessions.length, limit: 50, offset: 0 }); }
      if ((m = p.match(/^\/videos\/(\d+)$/))) {
        const s = sessions.find((x) => x.id === +m![1]);
        if (!s) return err(res, 404, 'video_not_found', `Video ${m[1]} not found`);
        return json(res, 200, { ...videoItem(s), summary: s.summary, frame_count: s.frames.length, frames_described: s.frames.length });
      }
      if ((m = p.match(/^\/videos\/(\d+)\/timeline$/))) {
        const s = sessions.find((x) => x.id === +m![1]);
        if (!s) return err(res, 404, 'video_not_found', `Video ${m[1]} not found`);
        await delay();
        return json(res, 200, {
          video_id: s.id, filename: s.filename, duration_s: s.duration_s, interval_s: s.interval_s, summary: s.summary,
          security_level: s.security_level, risk_score: s.risk_score,
          points: s.frames.map((f: any) => ({ frame_id: f.id, t: f.t, risk_score: f.risk_score, risk_level: f.risk_level, action_type: f.action_type })),
          events: s.frames.filter((f: any) => f.risk_score >= 50).map((f: any) => ({ frame_id: f.id, t: f.t, risk_level: f.risk_level, action_type: f.action_type, description: f.description })),
        });
      }
      if ((m = p.match(/^\/videos\/(\d+)\/frames$/))) {
        const s = sessions.find((x) => x.id === +m![1]);
        if (!s) return err(res, 404, 'video_not_found', `Video ${m[1]} not found`);
        const items = s.frames.map((f: any) => ({ id: f.id, frame_timestamp: f.t, image_description: f.description, risk_score: f.risk_score, risk_level: f.risk_level, action_type: f.action_type, url: `/frames/${f.id}/image` }));
        return json(res, 200, { items, total: items.length, limit: 500, offset: 0 });
      }
      if ((m = p.match(/^\/videos\/(\d+)\/clip$/))) {
        return err(res, 404, 'clip_unavailable', 'Source video is not available for this demo session', 'clip');
      }
      if ((m = p.match(/^\/videos\/(\d+)\/stream$/))) {
        return err(res, 404, 'clip_unavailable', 'Source video is not available for this demo session', 'stream');
      }
      if ((m = p.match(/^\/videos\/(\d+)\/ask$/)) && req.method === 'POST') {
        const s = sessions.find((x) => x.id === +m![1]);
        if (!s) return err(res, 404, 'video_not_found', `Video ${m[1]} not found`);
        let body = '';
        for await (const chunk of req) body += chunk;
        const { question, top_k } = JSON.parse(body || '{}');
        if (!question) return err(res, 422, 'invalid_input', 'question is required', 'validating_request');
        await new Promise((r) => setTimeout(r, 700));
        return json(res, 200, askAnswer(s, String(question), Number(top_k) || 5));
      }
      if ((m = p.match(/^\/frames\/(\d+)(\/image|\/elements)?$/))) {
        const hit = findFrame(+m[1]);
        if (!hit) return err(res, 404, 'frame_not_found', `Frame ${m[1]} not found`);
        const { s, f } = hit;
        if (m[2] === '/image') {
          res.setHeader('Content-Type', 'image/svg+xml');
          res.setHeader('Cache-Control', 'max-age=60');
          return void res.end(frameSvg(s, f));
        }
        if (m[2] === '/elements') return json(res, 200, { frame_id: f.id, width: f.width, height: f.height, elements: f.elements });
        return json(res, 200, { id: f.id, video_id: s.id, frame_timestamp: f.t, image_description: f.description, risk_score: f.risk_score, risk_level: f.risk_level, action_type: f.action_type, image_url: `/frames/${f.id}/image`, frame_width: f.width, frame_height: f.height });
      }
      if (p === '/v2/ingest-video/' && req.method === 'POST') {
        req.resume();
        await new Promise((r) => req.on('end', r));
        const id = `mock-${Date.now()}`;
        jobs.set(id, { start: Date.now(), videoId: 1, fail: false });
        res.statusCode = 202;
        res.setHeader('Content-Type', 'application/json');
        return void res.end(JSON.stringify({ job_id: id, status: 'queued', status_url: `/jobs/${id}` }));
      }
      if ((m = p.match(/^\/jobs\/(.+)$/))) {
        const j = jobs.get(m[1]);
        if (!j) return err(res, 404, 'job_not_found', `Job ${m[1]} not found`, 'job_lookup');
        const el = (Date.now() - j.start) / 1000;
        const total = 25;
        let status = 'queued', done = 0, warnings: string[] = [];
        if (el > 1.5) status = 'extracting';
        if (el > 3.5) { status = 'analyzing'; done = Math.min(total, Math.floor((el - 3.5) * 4)); }
        if (done >= total && el > 11) status = 'summarizing';
        if (el > 13) { status = 'done'; done = total; warnings = ['1 frame was skipped after analysis retries (mock warning)']; }
        return json(res, 200, { job_id: m[1], status, stage: status, frames_done: done, frames_total: status === 'queued' || status === 'extracting' ? (status === 'queued' ? 0 : total) : total, frames_failed: status === 'done' ? 1 : 0, video_id: status === 'done' ? j.videoId : null, error: null, warnings });
      }
    } catch (e: any) {
      return err(res, 500, 'mock_error', String(e?.message || e));
    }
    next();
  };
}
